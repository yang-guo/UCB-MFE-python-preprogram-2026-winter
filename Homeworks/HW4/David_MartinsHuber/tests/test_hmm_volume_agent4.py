"""Adversarial engine-integration tests for the volume-bar HMM strategy.

These tests drive ``strategies.hmm_volume.HMMVolumeStrategy`` only through the
public ``Backtester`` interface, plus the ``DecisionContext`` contract the
engine hands a strategy.  Every fixture is synthetic; nothing here touches the
course database.

Two tests in ``DefectChecks`` are expected to fail against the current
submission and are kept as regression evidence:

* ``test_constructor_accepts_only_fittable_configurations`` --
  ``HMMVolumeStrategy.__init__`` accepts ``n_states``/``min_train_bars``
  combinations that cannot satisfy ``research.hmm.fit_hmm``'s minimum sample,
  so the *run* dies at the first refit instead of construction failing.
* ``test_no_purchase_after_the_supplied_series_ends`` -- when the benchmark /
  volume series given to the strategy stop before the backtest ends, the
  strategy keeps proposing purchases on a frozen state, with no error, no
  diagnostic, and no refit.
"""

import unittest
import warnings
from copy import deepcopy

import numpy as np
import pandas as pd

from backtesting import (
    Backtester,
    DecisionContext,
    HeldPriceGapError,
    MarketData,
    Phase,
    Portfolio,
)
from strategies.hmm_volume import HMMVolumeStrategy

# Small, fast hyper-parameters.  ``risk_on_states=(0, 1, 2)`` makes the
# strategy "always invest spare cash", which keeps trade-level assertions
# deterministic instead of depending on which HMM state happens to win.
FAST = dict(
    median_window=20,
    bucket_multiple=5.0,
    n_states=3,
    min_train_bars=30,
    refit_every_bars=10,
    n_init=1,
    n_iter=50,
    top_n=3,
    risk_on_states=(0, 1, 2),
)


def synthetic_market(periods=700, symbols=6, seed=8, start="2015-01-01"):
    """Benchmark price/volume and a long-form price panel on one calendar."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=periods)
    benchmark = 100.0 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, periods)))
    volume = pd.Series(rng.lognormal(15.0, 0.3, periods), index=dates)

    rows = []
    for symbol in [f"S{i}" for i in range(symbols)]:
        path = benchmark * np.exp(np.cumsum(rng.normal(0.0, 0.005, periods)))
        rows.append(pd.DataFrame({"date": dates, "symbol": symbol, "adj_close": path}))
    prices = pd.concat(rows, ignore_index=True)

    benchmark_frame = pd.DataFrame({"date": dates, "adj_close": benchmark})
    quarters = pd.date_range(dates[0], dates[-1], freq="QS")
    membership = pd.concat(
        [
            pd.DataFrame(
                {
                    "symbol": [f"S{i}" for i in range(symbols)],
                    "quarter_start": quarter,
                    "selection_date": quarter - pd.Timedelta(days=1),
                }
            )
            for quarter in quarters
        ],
        ignore_index=True,
    )
    return dict(
        dates=dates,
        benchmark=pd.Series(benchmark, index=dates),
        volume=volume,
        prices=prices,
        benchmark_frame=benchmark_frame,
        membership=membership,
    )


def make_model(market, benchmark=None, volume=None, **overrides):
    options = dict(FAST)
    options.update(overrides)
    return HMMVolumeStrategy(
        market["benchmark"] if benchmark is None else benchmark,
        market["volume"] if volume is None else volume,
        **options,
    )


def make_engine(market, model, prices=None, **kwargs):
    return Backtester(
        market["prices"] if prices is None else prices,
        market["benchmark_frame"],
        market["membership"],
        model,
        initial_cash=kwargs.pop("initial_cash", 100_000),
        **kwargs,
    )


def make_context(model, signal, prices, membership, *, nav=100_000.0, cash=None,
                 volume=None):
    """A hand-built decision context, for tests of input handling."""
    signal = pd.Timestamp(signal)
    execution = signal + pd.Timedelta(days=1)
    cash = nav if cash is None else cash
    if "date" in prices.columns:
        price_view = (prices.loc[prices["date"] <= signal]
                      .pivot(index="date", columns="symbol", values="adj_close"))
    else:
        price_view = prices.loc[:signal].copy()
    return DecisionContext(
        signal,
        execution,
        price_view,
        membership.copy(),
        pd.DataFrame(),
        Portfolio(cash, cash, nav, {}),
        0,
        pd.DataFrame() if volume is None else volume.loc[:signal].copy(),
        pd.DataFrame(),
    )


class EngineCase(unittest.TestCase):
    """Shared fixtures: one causality-friendly market and one baseline run."""

    @classmethod
    def setUpClass(cls):
        cls.market = synthetic_market(periods=700, seed=8)
        cls.dates = cls.market["dates"]
        cls.start, cls.end = cls.dates[200], cls.dates[-1]
        cls.cash = 100_000
        cls.bps = 5
        cls.model = make_model(cls.market)
        cls.engine = make_engine(cls.market, cls.model, fee_bps=cls.bps)
        cls.result = cls.engine.run(cls.start, cls.end)

    # ----------------------------------------------------------- assertions
    def assert_accounting(self, result, initial_cash, fee_bps=0.0):
        """The book must reconcile: no leakage, no early sale, no credit."""
        rate = fee_bps / 10_000.0
        fills = result.fills
        buys = fills[fills.side == "BUY"] if len(fills) else fills
        sells = fills[fills.side == "SELL"] if len(fills) else fills
        if len(fills):
            self.assertTrue(
                np.allclose(fills.fee, fills.notional * rate, rtol=1e-12, atol=1e-12),
                msg="fee must equal notional x fee rate on every fill",
            )
            self.assertTrue(
                (buys.due_date - buys.date).eq(pd.Timedelta(days=90)).all(),
                msg="every purchase must be scheduled 90 calendar days out",
            )
            maturity = sells[sells.trigger == "maturity"]
            self.assertTrue(
                (maturity.date >= maturity.due_date).all(),
                msg="a maturity sale must never precede its due date",
            )
            self.assertTrue(
                set(sells.trigger) <= {"maturity", "delisting"},
                msg=f"unexpected sell trigger(s): {sorted(set(sells.trigger))}",
            )
            self.assertFalse(sells.lot_id.duplicated().any(),
                             msg="a lot must be sold at most once")
            self.assertTrue(set(sells.lot_id) <= set(buys.lot_id),
                            msg="every sale must close a purchased lot")
            strategy_ids = set(fills.strategy_id) | set(result.orders.strategy_id)
            self.assertEqual(len(strategy_ids), 1, msg="one strategy owns the account")

        spent = float((buys.notional + buys.fee).sum())
        proceeds = float((sells.notional - sells.fee).sum())
        self.assertAlmostEqual(
            float(result.history.cash.iloc[-1]),
            initial_cash - spent + proceeds,
            delta=1e-6 * initial_cash,
            msg="final cash must equal initial cash minus purchases plus proceeds",
        )
        history = result.history
        self.assertTrue(np.isfinite(history.nav).all())
        self.assertTrue((history.cash >= -1e-9).all(),
                        msg="the strategy must never drive cash negative")
        self.assertTrue(
            np.allclose(history.nav, history.cash + history.holdings_value, atol=1e-6),
            msg="nav must equal cash plus marked holdings on every session",
        )
        self.assertAlmostEqual(
            float(result.pnl.realized + result.pnl.unrealized),
            float(result.pnl.total),
            delta=1e-6 * initial_cash,
            msg="realized plus unrealized must equal total PnL",
        )
        self.assertAlmostEqual(
            float(history.nav.iloc[-1]),
            initial_cash + float(result.pnl.total),
            delta=1e-6 * initial_cash,
        )
        closed = set(sells.lot_id) if len(fills) else set()
        open_ids = set(buys.lot_id) - closed if len(fills) else set()
        self.assertEqual(set(result.open_lots.lot_id), open_ids,
                         msg="open lots must match unsold purchases")


class LifecycleChecks(EngineCase):
    """Self-fitting refit lifecycle, audit, deepcopy, and identity."""

    def test_self_fitting_run_needs_no_training_panel(self):
        self.assertIs(HMMVolumeStrategy.self_fitting, True)
        # ``make_engine`` supplies no training_panel at all.
        self.assertFalse(self.result.refits.empty)
        self.assertTrue(self.result.refits.status.eq("success").all())
        # A self-fitting strategy is handed ``None``; the audit records no rows.
        self.assertTrue(self.result.refits["rows"].eq(0).all())
        self.assertTrue(self.result.refits.first_date.isna().all())
        phases = self.result.phases.phase
        self.assertIn("refitting", set(phases))
        self.assertEqual(phases.iloc[-1], "complete")

    def test_refit_audit_tracks_bar_milestones_and_model_versions(self):
        bars = self.model._bars
        signals = list(self.result.refits.signal_date)
        self.assertTrue(len(signals) > 1)
        # The run starts after the warm-up, so the first refit fires on the
        # first signal session that already has enough bars; after that the
        # cadence is exactly one refit per ``refit_every_bars`` completed bars.
        first_signal = self.dates[self.dates.get_loc(self.start) - 1]
        counts = [len(self.model._bars_at(signal)) for signal in signals]
        self.assertEqual(counts[0], len(self.model._bars_at(first_signal)),
                         msg="the first refit must fire on the first eligible session")
        self.assertGreaterEqual(counts[0], self.model.min_train_bars + 1)
        for index, (signal, count) in enumerate(zip(signals, counts)):
            self.assertEqual(count, counts[0] + index * self.model.refit_every_bars,
                             msg="each refit must wait for refit_every_bars new bars")
            self.assertEqual(signal,
                             max(first_signal, bars["end"].iloc[count - 1]),
                             msg="a refit must fire on the bar-end session, not mid-bar")
        self.assertTrue(all(a < b for a, b in zip(signals, signals[1:])))
        self.assertEqual(list(self.result.refits.model_version),
                         list(range(1, len(self.result.refits) + 1)))
        self.assertEqual(self.engine.model_version, len(self.result.refits))
        # The published settings must describe the model actually fitted.
        settings = self.result.refits.settings.iloc[-1]
        self.assertEqual(settings["n_states"], self.model.n_states)
        self.assertEqual(settings["min_train_bars"], self.model.min_train_bars)
        self.assertEqual(settings["refit_every_bars"], self.model.refit_every_bars)
        self.assertEqual(settings["seed"], self.model.seed)
        self.assertEqual(settings["risk_on_states"], list(self.model.risk_on_states))
        # Audit rows are independent copies, not aliases of the live model.
        self.assertIsNot(self.result.refits.settings.iloc[0],
                         self.result.refits.settings.iloc[1])
        fitted = self.engine.strategy
        original = fitted.n_states
        try:
            fitted.n_states = 99
            self.assertEqual(dict(self.result.refits.settings.iloc[-1])["n_states"],
                             self.model.n_states,
                             msg="the audit must snapshot, not alias, model settings")
        finally:
            fitted.n_states = original

    def test_every_fill_carries_the_model_version_in_force(self):
        refits = self.result.refits
        fills = self.result.fills
        self.assertTrue(len(fills) > 0)
        expected = [int((refits.signal_date <= signal).sum()) for signal in fills.signal_date]
        self.assertEqual(list(fills.model_version), expected,
                         msg="a fill must record the model version live at its signal")
        # A sale carries the entry lot's version, not today's.
        buys = fills[fills.side == "BUY"].set_index("lot_id").model_version
        sells = fills[fills.side == "SELL"]
        for lot_id, version in sells.set_index("lot_id").model_version.items():
            self.assertEqual(version, buys.loc[lot_id])

    def test_strategy_identity_and_refit_settings_are_stable(self):
        model = make_model(self.market, strategy_id="hmm_volume_probe")
        engine = make_engine(self.market, model, fee_bps=0)
        result = engine.run(self.start, self.end)
        self.assertEqual(engine.strategy.strategy_id, "hmm_volume_probe")
        self.assertEqual(set(result.fills.strategy_id), {"hmm_volume_probe"})
        self.assertEqual(set(result.orders.strategy_id), {"hmm_volume_probe"})
        self.assertTrue(result.fills.signal_date.lt(result.fills.date).all(),
                        msg="purchases execute strictly after their signal")

    def test_deepcopy_shares_input_frames_and_copies_model_state(self):
        fitted = self.engine.strategy
        self.assertIsNot(fitted, self.model)
        self.assertIsNotNone(fitted._fit)
        clone = deepcopy(fitted)
        # Large read-only inputs are shared by design.
        self.assertIs(clone._bars, fitted._bars)
        self.assertIs(clone.benchmark, fitted.benchmark)
        self.assertIs(clone.volume, fitted.volume)
        # Model state is not.
        self.assertIsNot(clone._fit, fitted._fit)
        clone._state = (fitted.state + 1) % fitted.n_states
        clone._state_bar_count = -1
        self.assertNotEqual(clone._state, fitted.state)
        self.assertNotEqual(clone._state_bar_count, fitted._state_bar_count)
        # Arrays are copied, not aliased: an in-place edit cannot leak back.
        expected = fitted.probabilities
        clone._probabilities[0] = -1.0
        self.assertFalse(np.array_equal(clone.probabilities, expected),
                         msg="the clone's posterior must be its own array")
        self.assertTrue(np.array_equal(fitted.probabilities, expected))
        # A refit must never mutate the shared bar table.
        before = fitted.bars
        context = make_context(fitted, fitted._bars["end"].iloc[-1],
                               self.market["prices"], self.market["membership"])
        clone.refit(None, context)
        self.assertTrue(fitted.bars.equals(before))

    def test_caller_strategy_is_untouched_and_runs_repeat_exactly(self):
        # The engine only ever refits a deep copy, so the caller's object stays
        # pristine and a second engine reproduces the run bit for bit.
        self.assertIsNone(self.model._fit)
        self.assertIsNone(self.model.state)
        self.assertIsNone(self.model.state_date)
        self.assertIsNot(self.engine.strategy, self.model)
        again = make_engine(self.market, self.model, fee_bps=self.bps)
        repeat = again.run(self.start, self.end)
        self.assertTrue(repeat.fills.equals(self.result.fills))
        self.assertTrue(repeat.orders.equals(self.result.orders))
        self.assertTrue(np.allclose(repeat.history.nav, self.result.history.nav))
        self.assertIsNone(self.model._fit)

    def test_refit_before_warmup_raises_and_leaves_the_model_unfitted(self):
        model = make_model(self.market)
        early = self.model._bars["end"].iloc[3]
        context = make_context(model, early, self.market["prices"],
                               self.market["membership"])
        self.assertFalse(model.needs_refit(context),
                         msg="the engine must not attempt a doomed refit")
        with self.assertRaises(ValueError):
            model.refit(None, context)
        # A failed fit must not leave a half-built model behind.
        self.assertIsNone(model._fit)
        self.assertIsNone(model.state)
        self.assertIsNone(model.probabilities)
        # With warm-up complete the same object fits normally.
        late = self.model._bars["end"].iloc[80]
        model.refit(None, make_context(model, late, self.market["prices"],
                                       self.market["membership"]))
        self.assertIsNotNone(model._fit)
        self.assertIn(model.state, range(model.n_states))
        self.assertEqual(model.state_date, late)


class AccountingChecks(EngineCase):
    """90-day lots, maturity exits, cash recycling, and hostile quote tapes."""

    def test_lots_hold_to_maturity_across_recycled_purchases(self):
        result = self.result
        self.assertGreater(len(result.fills), 10)
        self.assert_accounting(result, self.cash, self.bps)
        buys = result.fills[result.fills.side == "BUY"]
        sells = result.fills[result.fills.side == "SELL"]
        self.assertTrue(len(sells) > 0, msg="maturity exits must actually happen")
        self.assertTrue(sells.trigger.eq("maturity").all())
        # Cash recycling: the same name is re-bought after earlier lots mature.
        repeated = buys.symbol.value_counts()
        self.assertTrue((repeated > 1).any(),
                        msg="maturity proceeds must be recycled into new lots")
        self.assertTrue((buys.groupby("symbol").signal_date.nunique() > 1).any())
        # No purchase may reuse a signal twice for one name (no duplicate keys).
        self.assertFalse(result.orders.duplicated(["signal_date", "symbol"]).any())
        self.assertTrue(result.open_lots.due_date.gt(result.history.index[-1]).all())

    def test_missing_entry_quote_is_rejected_and_recycled(self):
        buys = self.result.fills[self.result.fills.side == "BUY"]
        gap = buys.date.min()
        gap_symbols = sorted(buys.loc[buys.date.eq(gap), "symbol"])
        self.assertTrue(gap_symbols)
        prices = self.market["prices"]
        holed = prices.loc[~((prices.date.eq(gap)) & prices.symbol.isin(gap_symbols))]
        result = make_engine(self.market, make_model(self.market), prices=holed,
                             fee_bps=0).run(self.start, self.end)
        self.assertTrue(result.diagnostics.event.eq("entry_rejected").any())
        self.assert_accounting(result, self.cash, 0)
        rejected = result.orders[result.orders.status.eq("rejected_missing_quote")]
        self.assertEqual(set(rejected.symbol), set(gap_symbols))
        # The refused budget is not lost: each blocked name is bought later.
        for symbol in gap_symbols:
            later = result.fills[(result.fills.side == "BUY")
                                 & result.fills.symbol.eq(symbol)]
            self.assertFalse(later.empty, msg=f"{symbol} was never re-proposed")
        self.assertTrue(result.phases.phase.eq("complete").any())

    def test_maturity_exit_waits_for_the_next_quote(self):
        lot = self.result.fills[self.result.fills.side == "BUY"].iloc[0]
        symbol, due = lot.symbol, lot.due_date
        gap_sessions = [day for day in self.dates
                        if due <= day <= due + pd.Timedelta(days=4)]
        self.assertTrue(gap_sessions)
        prices = self.market["prices"]
        holed = prices.loc[~((prices.symbol.eq(symbol)) & prices.date.isin(gap_sessions))]
        result = make_engine(self.market, make_model(self.market), prices=holed,
                             fee_bps=0).run(self.start, self.end)
        self.assertTrue(result.diagnostics.event.eq("maturity_exit_delayed").any())
        self.assertTrue(result.diagnostics.event.eq("stale_mark").any())
        sale = result.fills[(result.fills.side == "SELL")
                            & result.fills.symbol.eq(symbol)].iloc[0]
        self.assertGreater(sale.date, due,
                           msg="the exit must be delayed, never pulled forward")
        self.assertEqual(sale.trigger, "maturity")
        self.assert_accounting(result, self.cash, 0)

    def test_delisting_last_price_cannot_enable_forced_liquidation(self):
        with self.assertRaisesRegex(ValueError, "stale mark"):
            make_engine(self.market, make_model(self.market), delisting="last_price")

    def test_delisting_error_fails_cleanly_without_partial_bookkeeping(self):
        stop = self.dates[350]
        prices = self.market["prices"]
        dead = prices.loc[~((prices.symbol.eq("S1")) & prices.date.gt(stop))]
        engine = make_engine(self.market, make_model(self.market), prices=dead,
                             fee_bps=0, max_stale_sessions=2)
        with self.assertRaises(HeldPriceGapError):
            engine.run(self.start, self.end)
        self.assertIs(engine.phase, Phase.FAILED)
        history = engine.history_so_far
        self.assertTrue(history.index.max() < self.dates[-1])
        # Nothing was committed for the aborted session.
        fills = pd.DataFrame(engine.oms.fills)
        self.assertTrue(fills.date.le(history.index.max()).all())
        self.assertGreaterEqual(engine.oms.cash, 0.0)
        self.assertTrue(np.isfinite(history.nav).all())
        self.assertEqual(engine.phases[-1]["phase"], "failed")


class DataShapeChecks(EngineCase):
    """Input shapes, duplicates, alignment, and causal signal timing."""

    def test_dataframe_and_series_inputs_give_the_same_run(self):
        market = self.market
        frame_model = HMMVolumeStrategy(
            pd.DataFrame({"date": market["dates"], "adj_close": market["benchmark"].to_numpy()}),
            pd.DataFrame({"date": market["dates"], "volume": market["volume"].to_numpy()}),
            **FAST,
        )
        self.assertTrue(frame_model.bars.equals(self.model._bars))
        result = make_engine(market, frame_model, fee_bps=self.bps).run(self.start, self.end)
        self.assertTrue(result.fills.equals(self.result.fills))
        self.assertTrue(np.allclose(result.history.nav, self.result.history.nav))

    def test_duplicate_dates_keep_the_last_supplied_row(self):
        benchmark = self.market["benchmark"]
        volume = self.market["volume"]
        # A bar boundary is the only place a benchmark price can move a bar.
        conflict_date = self.model._bars["end"].iloc[5]
        conflict = float(benchmark.loc[conflict_date]) * 1.5
        duplicated = pd.concat([benchmark, pd.Series([conflict], index=[conflict_date])])
        model = make_model(self.market, benchmark=duplicated)
        self.assertEqual(len(model.benchmark), len(benchmark))
        self.assertAlmostEqual(float(model.benchmark.loc[conflict_date]), conflict)
        self.assertFalse(model.bars.equals(self.model._bars),
                         msg="the last duplicate must feed the bar builder")

        zeroed_date = self.model._bars["end"].iloc[8]
        volume_duplicate = pd.concat([volume, pd.Series([0.0], index=[zeroed_date])])
        zeroed = make_model(self.market, volume=volume_duplicate)
        self.assertEqual(len(zeroed.volume), len(volume))
        self.assertEqual(float(zeroed.volume.loc[zeroed_date]), 0.0)
        self.assertFalse(zeroed.bars.equals(self.model._bars),
                         msg="the last duplicate volume must feed the bar builder")

    def test_nonfinite_timezone_and_empty_overlap_inputs_are_rejected(self):
        market = self.market
        benchmark, volume = market["benchmark"], market["volume"]

        tz_benchmark = benchmark.copy()
        tz_benchmark.index = tz_benchmark.index.tz_localize("UTC")
        with self.assertRaises(ValueError):
            make_model(market, benchmark=tz_benchmark)
        tz_volume = volume.copy()
        tz_volume.index = tz_volume.index.tz_localize("America/New_York")
        with self.assertRaises(ValueError):
            make_model(market, volume=tz_volume)

        nan_volume = volume.copy()
        nan_volume.iloc[10] = np.nan
        with self.assertRaises(ValueError):
            make_model(market, volume=nan_volume)
        negative = volume.copy()
        negative.iloc[10] = -1.0
        with self.assertRaises(ValueError):
            make_model(market, volume=negative)
        nonpositive = benchmark.copy()
        nonpositive.iloc[10] = 0.0
        with self.assertRaises(ValueError):
            make_model(market, benchmark=nonpositive)

        moved = volume.copy()
        moved.index = moved.index + pd.Timedelta(days=3650)
        with self.assertRaises(ValueError):
            make_model(market, volume=moved)
        with self.assertRaises(TypeError):
            HMMVolumeStrategy(list(benchmark), list(volume), **FAST)

    def test_misaligned_volume_uses_the_intersection(self):
        market = self.market
        late = market["volume"].loc[market["volume"].index >= self.dates[400]]
        model = make_model(market, volume=late)
        self.assertEqual(len(model.volume), len(late))
        self.assertGreaterEqual(model.bars.start.iloc[0], late.index[0])
        self.assertGreaterEqual(model.bars.end.iloc[0],
                                late.index[0] + pd.Timedelta(days=1))
        early = make_model(market, volume=market["volume"].loc[: self.dates[300]])
        self.assertLessEqual(early.bars.end.iloc[-1], self.dates[300])
        # A late-starting volume series still supports a run inside its window.
        result = make_engine(market, model).run(self.dates[450], self.end)
        self.assert_accounting(result, self.cash, self.bps)

    def test_signals_inside_a_bar_do_not_advance_the_state(self):
        bars = self.model._bars
        spans = (bars["end"] - bars["start"]).dt.days.to_numpy()
        index = int(np.argmax(spans[40:80]) + 40)
        bar_end = bars["end"].iloc[index]
        position = self.dates.get_loc(bar_end)
        next_session = self.dates[position + 1]
        engine = make_engine(self.market, make_model(self.market))
        first = engine._context(bar_end, next_session, {})
        second = engine._context(next_session, self.dates[position + 2], {})

        model = engine.strategy
        model.refit(None, first)
        state, state_date = model.state, model.state_date
        self.assertEqual(state_date, bar_end)
        self.assertFalse(model.needs_refit(second),
                         msg="a mid-bar session must not trigger a refit")
        self.assertEqual(len(model._bars_at(next_session)), len(model._bars_at(bar_end)))
        self.assertLessEqual(model._bars_at(next_session)["end"].max(), next_session)
        budgets = model.propose_buys(second)
        self.assertEqual(model.state, state)
        self.assertEqual(model.state_date, state_date)
        self.assertEqual(set(budgets), set(model.propose_buys(first)))

    def test_future_input_data_cannot_change_a_completed_run(self):
        cut = self.dates[500]
        truncated = make_model(self.market,
                               benchmark=self.market["benchmark"].loc[:cut],
                               volume=self.market["volume"].loc[:cut])
        full = make_model(self.market)
        left = make_engine(self.market, truncated, fee_bps=self.bps).run(self.dates[100], cut)
        right = make_engine(self.market, full, fee_bps=self.bps).run(self.dates[100], cut)
        self.assertTrue(left.fills.equals(right.fills))
        self.assertTrue(left.orders.equals(right.orders))
        self.assertTrue(np.allclose(left.history.nav, right.history.nav))


class RobustnessChecks(EngineCase):
    """Degenerate universes, empty membership, short runs, and data plumbing."""

    def test_single_symbol_universe_uses_all_cash_in_one_order(self):
        market = synthetic_market(periods=700, symbols=1, seed=11)
        model = make_model(market, top_n=50)
        result = make_engine(market, model).run(market["dates"][200], market["dates"][-1])
        self.assertTrue(len(result.fills) > 0)
        self.assertEqual(set(result.fills.symbol), {"S0"})
        self.assertEqual(int(result.orders.groupby("signal_date").size().max()), 1)
        buys = result.fills[result.fills.side == "BUY"]
        self.assertTrue((buys.notional > 0).all())
        self.assert_accounting(result, self.cash, self.bps)

    def test_empty_membership_completes_without_orders(self):
        market = self.market
        engine = Backtester(market["prices"], market["benchmark_frame"],
                            market["membership"].iloc[0:0], make_model(market),
                            initial_cash=self.cash)
        result = engine.run(self.start, self.end)
        self.assertTrue(result.fills.empty)
        self.assertTrue(result.orders.empty)
        self.assertTrue(result.diagnostics.empty)
        # The model still fits and is audited even with nothing to buy.
        self.assertFalse(result.refits.empty)
        self.assertTrue(result.refits.status.eq("success").all())
        self.assertEqual(result.history.nav.nunique(), 1)
        self.assert_accounting(result, self.cash, 0)

    def test_run_before_warmup_is_a_clean_no_trade_run(self):
        result = make_engine(self.market, make_model(self.market)).run(
            self.dates[1], self.dates[20])
        self.assertEqual(len(result.history) - 1, 20)
        self.assertTrue(result.fills.empty)
        self.assertTrue(result.refits.empty)
        # Empty frames still carry the documented schema.
        self.assertIn("status", result.refits.columns)
        self.assertIn("model_version", result.refits.columns)
        self.assertIn("trigger", result.fills.columns)
        self.assertEqual(float(result.pnl.total), 0.0)
        self.assertEqual(result.phases.phase.iloc[-1], "complete")
        summary = result.performance()
        self.assertFalse(summary.empty)

    def test_allow_partial_clips_the_run_end(self):
        model = make_model(self.market)
        with self.assertRaises(ValueError):
            make_engine(self.market, model).run(
                self.dates[200], self.dates[-1] + pd.Timedelta(days=30))
        engine = make_engine(self.market, make_model(self.market))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = engine.run(self.dates[200],
                                self.dates[-1] + pd.Timedelta(days=30),
                                allow_partial=True)
        self.assertTrue(result.coverage.range_clipped)
        self.assertEqual(result.coverage.effective_end, self.dates[-1])
        self.assertTrue(any("Partial backtest" in str(w.message) for w in caught))
        self.assert_accounting(result, self.cash, self.bps)

    def test_market_data_volume_panel_selects_the_liquid_basket(self):
        data, bench_series, bench_volume = self._liquidity_market()
        model = make_model(self.market, benchmark=bench_series, volume=bench_volume,
                           top_n=2)
        probe = Backtester(market_data=data, strategy=model,
                           initial_cash=self.cash, fee_bps=0)
        # The engine must hand the strategy judged-so-far share volume only.
        signal = data.calendar[250]
        context = probe._context(signal, data.calendar[251], {})
        expected = data.volume_panel(end=signal)
        self.assertTrue(context.volume.equals(expected),
                        msg="context.volume must be the point-in-time panel slice")
        self.assertEqual(list(context.volume.columns), ["AAA", "BBB", "MID", "ZZZ"])
        self.assertLessEqual(context.prices.index.max(), signal)

        engine = Backtester(market_data=data,
                            strategy=make_model(self.market, benchmark=bench_series,
                                                volume=bench_volume, top_n=2),
                            initial_cash=self.cash, fee_bps=0)
        result = engine.run(data.calendar[200], data.calendar[-1])
        self.assertEqual(set(result.fills.symbol), {"ZZZ", "MID"},
                         msg="top_n must follow trailing dollar volume, not the alphabet")

        # Without a volume panel the documented alphabetical fallback applies.
        plain_market = self._alphabetical_market()
        fallback = Backtester(
            plain_market["prices"], plain_market["benchmark_frame"],
            plain_market["membership"],
            make_model(self.market, benchmark=plain_market["benchmark"],
                       volume=plain_market["volume"], top_n=2),
            initial_cash=self.cash, fee_bps=0,
        ).run(plain_market["dates"][200], plain_market["dates"][-1])
        self.assertEqual(set(fallback.fills.symbol), {"AAA", "BBB"})

    def test_partial_volume_coverage_completes_the_basket_deterministically(self):
        model = make_model(self.market, top_n=3)
        signal = self.model._bars["end"].iloc[80]
        prices = (self.market["prices"].loc[self.market["prices"].date <= signal]
                  .pivot(index="date", columns="symbol", values="adj_close"))
        volume = pd.DataFrame(1.0, index=prices.index, columns=["S4", "S5"])
        context = make_context(model, signal, prices, self.market["membership"],
                               volume=volume)
        model.refit(None, context)
        budgets = model.propose_buys(context)
        self.assertEqual(len(budgets), 3)
        self.assertTrue({"S5", "S4"} <= set(budgets),
                        msg="names with a volume panel must outrank the remainder")
        self.assertLessEqual(sum(budgets.values()), 100_000.0)
        # The remainder is filled alphabetically and the basket stays exact.
        self.assertEqual(set(budgets), {"S4", "S5", "S0"})

    def test_order_floor_governs_tiny_accounts(self):
        """``min_order_notional`` is the documented dust filter, not a bug.

        With $2 of capital split three ways each slice is below the $1 floor, so
        nothing is bought; with a $0.5 floor all three names are bought.  Either
        way the budgets stay inside cash and no order exceeds the account.
        """
        signal = self.model._bars["end"].iloc[80]
        prices = (self.market["prices"].loc[self.market["prices"].date <= signal]
                  .pivot(index="date", columns="symbol", values="adj_close"))
        floor = make_model(self.market, top_n=3, min_order_notional=1.0)
        small = make_context(floor, signal, prices, self.market["membership"], nav=2.0)
        floor.refit(None, small)
        self.assertEqual(floor.propose_buys(small), {})
        generous = make_model(self.market, top_n=3, min_order_notional=0.5)
        tiny = make_context(generous, signal, prices, self.market["membership"], nav=2.0)
        generous.refit(None, tiny)
        budgets = generous.propose_buys(tiny)
        self.assertEqual(len(budgets), 3)
        self.assertGreaterEqual(min(budgets.values()), 0.5)
        self.assertLessEqual(sum(budgets.values()), 2.0)

    # ------------------------------------------------------------- fixtures
    def _liquidity_market(self, periods=460):
        """Four names whose dollar volume order contradicts the alphabet."""
        dates = pd.bdate_range("2018-01-01", periods=periods)
        rng = np.random.default_rng(3)
        benchmark = 100.0 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, periods)))
        volumes = {"AAA": 100.0, "BBB": 1_000.0, "MID": 10_000.0, "ZZZ": 100_000.0}
        rows = []
        for symbol, base in {"AAA": 10.0, "BBB": 20.0, "MID": 30.0, "ZZZ": 40.0}.items():
            path = base * np.exp(np.cumsum(rng.normal(0.0, 0.01, periods)))
            rows.extend(
                dict(date=day, symbol=symbol, adj_close=price, close=price,
                     volume=volumes[symbol])
                for day, price in zip(dates, path)
            )
        prices = pd.DataFrame(rows)
        benchmark_frame = pd.DataFrame({"date": dates, "adj_close": benchmark})
        universe = pd.DataFrame({"symbol": sorted(volumes), "sector": "Tech"})
        quarters = pd.date_range(dates[0], dates[-1], freq="QS")
        membership = pd.concat(
            [
                pd.DataFrame(
                    {
                        "symbol": sorted(volumes),
                        "quarter_start": quarter,
                        "selection_date": quarter - pd.Timedelta(days=1),
                        "universe_rank": 1,
                    }
                )
                for quarter in quarters
            ],
            ignore_index=True,
        )
        filings = []
        for symbol in sorted(volumes):
            for quarter in pd.date_range("2017-09-30", dates[-1], freq="QS"):
                filings.append(
                    dict(
                        symbol=symbol,
                        fiscal_date=quarter,
                        period="quarter",
                        filing_date=quarter + pd.Timedelta(days=10),
                        revenue=100.0,
                        net_income=10.0,
                        operating_income=15.0,
                        total_assets=200.0,
                        total_debt=50.0,
                        cash_and_equivalents=20.0,
                        free_cash_flow=5.0,
                        operating_cash_flow=15.0,
                        reported_currency="USD",
                        shares_outstanding=100.0,
                    )
                )
        data = MarketData.from_frames(prices, pd.DataFrame(filings), universe,
                                      benchmark_frame, membership)
        return (data, pd.Series(benchmark, index=dates),
                pd.Series(sum(volumes.values()), index=dates))

    def _alphabetical_market(self, periods=460):
        """The same universe on its own calendar, for the no-volume fallback."""
        dates = pd.bdate_range("2018-01-01", periods=periods)
        rng = np.random.default_rng(3)
        benchmark = 100.0 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, periods)))
        rows = []
        for symbol, base in {"AAA": 10.0, "BBB": 20.0, "MID": 30.0, "ZZZ": 40.0}.items():
            path = base * np.exp(np.cumsum(rng.normal(0.0, 0.01, periods)))
            rows.append(pd.DataFrame({"date": dates, "symbol": symbol, "adj_close": path}))
        quarters = pd.date_range(dates[0], dates[-1], freq="QS")
        membership = pd.concat(
            [
                pd.DataFrame(
                    {
                        "symbol": ["AAA", "BBB", "MID", "ZZZ"],
                        "quarter_start": quarter,
                        "selection_date": quarter - pd.Timedelta(days=1),
                    }
                )
                for quarter in quarters
            ],
            ignore_index=True,
        )
        return dict(
            dates=dates,
            benchmark=pd.Series(benchmark, index=dates),
            volume=pd.Series(111_110.0, index=dates),
            prices=pd.concat(rows, ignore_index=True),
            benchmark_frame=pd.DataFrame({"date": dates, "adj_close": benchmark}),
            membership=membership,
        )


class DefectChecks(EngineCase):
    """Regression evidence for defects found by this round; both must fail."""

    def test_constructor_accepts_only_fittable_configurations(self):
        """A configuration the constructor accepts must survive a run.

        ``research.hmm.fit_hmm`` needs ``max(20, 5 * n_states)`` observations,
        and ``refit`` always fits on exactly ``min_train_bars`` of them, but
        ``HMMVolumeStrategy.__init__`` only enforces ``min_train_bars >= 20``.
        A configuration that is accepted therefore dies in the middle of the
        run at the first refit (``backtesting/engine.py:202`` ->
        ``strategies/hmm_volume.py:230`` -> ``research/hmm.py:76``).
        """
        for n_states, min_train_bars in [(5, 20), (5, 24), (6, 29), (3, 30)]:
            with self.subTest(n_states=n_states, min_train_bars=min_train_bars):
                try:
                    model = make_model(
                        self.market,
                        n_states=n_states,
                        min_train_bars=min_train_bars,
                        risk_on_states=tuple(range(n_states)),
                    )
                except ValueError:
                    continue  # Rejecting at construction is the acceptable fix.
                engine = make_engine(self.market, model)
                try:
                    result = engine.run(self.dates[150], self.end)
                except Exception as exc:  # noqa: BLE001 - the defect is the crash
                    self.fail(
                        f"n_states={n_states}, min_train_bars={min_train_bars} is accepted "
                        f"by the constructor but the run died at the first refit: "
                        f"{type(exc).__name__}: {exc} "
                        "(backtesting/engine.py:202 -> strategies/hmm_volume.py:227 "
                        "-> research/hmm.py:77; the fit needs "
                        f"max(20, 5*{n_states}) observations, refit supplies "
                        f"{min_train_bars})"
                    )
                self.assertFalse(
                    result.refits.empty,
                    msg=(f"n_states={n_states}, min_train_bars={min_train_bars} "
                         "was accepted but never refitted"),
                )

    def test_no_purchase_after_the_supplied_series_ends(self):
        """A strategy must not trade on dates it was given no data for.

        The engine keeps running past the end of the strategy's own benchmark
        and volume series; ``_bars_at`` then returns the same bars forever, so
        ``needs_refit`` is ``False`` and ``propose_buys`` keeps emitting buys
        from a state frozen months earlier, with no warning and no diagnostic
        (``strategies/hmm_volume.py:205-218`` and ``:256-262``).
        """
        cut = self.dates[350]
        model = make_model(self.market,
                           benchmark=self.market["benchmark"].loc[:cut],
                           volume=self.market["volume"].loc[:cut])
        early = make_engine(self.market, model, fee_bps=0).run(self.dates[300], cut)
        buys = early.fills[early.fills.side == "BUY"]
        self.assertTrue(len(buys) > 0, msg="the run must actually trade before the cut")
        self.assertTrue(len(buys[buys.signal_date <= cut]) > 0)

        model = make_model(self.market,
                           benchmark=self.market["benchmark"].loc[:cut],
                           volume=self.market["volume"].loc[:cut])
        with self.assertRaisesRegex(ValueError, "ended before the signal date"):
            make_engine(self.market, model, fee_bps=0).run(self.dates[300], self.end)


if __name__ == "__main__":
    unittest.main()
