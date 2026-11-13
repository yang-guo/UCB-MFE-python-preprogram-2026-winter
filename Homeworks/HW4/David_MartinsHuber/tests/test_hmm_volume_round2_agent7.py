"""Round-2 adversarial integration tests on the real top-2000 database.

These tests run :class:`strategies.hmm_volume.HMMVolumeStrategy` end to end on
real point-in-time data:

* the benchmark price and universe volume come from
  :func:`research.market.build_market_series` (the research definitions);
* the stock panel comes from the verbatim course loaders in ``data_loading``
  (``load_data`` / ``load_membership``) and, separately, from
  :class:`backtesting.market_data.MarketData`;
* the run is 2012-01-03 .. 2014-12-31 over 15 large caps that are members in
  every quarter of the window, plus shorter windows for the causal and
  gating checks.

What is attacked
----------------
1. refit lifecycle, model-version/audit coherence, recorded settings;
2. fills, 90-calendar-day maturity, no early sales, cash/NAV reconciliation,
   fill prices against the raw panel;
3. the strategy's per-session states against
   :func:`research.hmm.walk_forward_states` on the same bars, including the
   refit lifecycle and version offset;
4. no future data: perturbing the real benchmark price, universe volume,
   stock prices and stock volume strictly after a cutoff leaves the earlier
   run bit-identical;
5. the default risk-on gate on real data, including crisis-state sessions in
   2011 and an explicit ``risk_on_states=(2,)`` policy that blocks state 1;
6. the engine's explicit-frame path versus the ``market_data`` path, both
   without a volume panel (must be identical) and with ``top_n`` smaller than
   the universe (the documented alphabetical fallback vs. the dollar-volume
   ranking).

Runtime
-------
Measured with ``../.venv/bin/python`` on the real database (330 MB) from the
submission folder: 97 s warm and 135 s cold for the whole module. The dominant
costs are one ~17 s ``walk_forward_states`` reference run and several 5-13 s
backtests; the expensive shared artefacts are built once in ``setUpClass``.

The module skips cleanly when ``data/equities_top2000.duckdb`` is absent.
"""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

DATABASE = Path(__file__).resolve().parents[1] / "data" / "equities_top2000.duckdb"
DATA_END = "2015-06-30"

# Large caps that are in the point-in-time universe in every quarter of
# 2011-10-01 .. 2014-12-31 and quote on every benchmark session from 2000.
SYMBOLS = (
    "AAPL", "CVX", "GE", "IBM", "JNJ", "JPM", "KO",
    "MRK", "MSFT", "ORCL", "PFE", "PG", "WFC", "WMT", "XOM",
)

HOLD_DAYS = 90
INITIAL_CASH = 250_000.0
FEE_BPS = 5.0

MAIN_START, MAIN_END = "2012-01-03", "2014-12-31"
CUTOFF = pd.Timestamp("2013-06-28")


@unittest.skipUnless(
    DATABASE.is_file(),
    f"real database not available at {DATABASE}",
)
class HMMVolumeRealDataIntegration(unittest.TestCase):
    """End-to-end attacks against the real DuckDB delivery."""

    @classmethod
    def setUpClass(cls):
        from backtesting import Backtester
        from backtesting.market_data import MarketData
        from data_loading import load_data, load_membership
        from research.hmm import walk_forward_states
        from research.market import build_market_series
        from strategies.hmm_volume import HMMVolumeStrategy

        cls.database = DATABASE
        cls.market = build_market_series(DATABASE, DATA_END)
        cls.benchmark_price, cls.benchmark_volume = cls.market["benchmark"].aligned()

        prices, fundamentals, universe, benchmark = load_data(
            DATABASE, end=DATA_END, symbols=list(SYMBOLS)
        )
        membership = load_membership(DATABASE, end=DATA_END)
        membership = membership.loc[membership.symbol.isin(SYMBOLS)].reset_index(drop=True)
        cls.prices = prices
        cls.fundamentals = fundamentals
        cls.universe = universe
        cls.benchmark = benchmark
        cls.membership = membership
        cls.calendar = pd.DatetimeIndex(benchmark.date).sort_values()
        cls.market_data = MarketData.from_frames(
            prices, fundamentals, universe, benchmark, membership
        )

        # The strategy's own bar builder, the research reference states, and
        # the main recorded run are all shared by several tests.
        cls.bars = HMMVolumeStrategy(
            cls.benchmark_price, cls.benchmark_volume, top_n=8
        ).bars
        cls.reference = walk_forward_states(
            cls.bars.log_return.to_numpy(dtype=float),
            cls.bars.end,
            min_train=250,
            refit_every=13,
            n_states=3,
            n_init=3,
            seed=7,
            n_iter=200,
        )
        cls.main_records, cls.main_result, cls.main_engine = cls.drive(
            cls.make_engine(explicit=True, top_n=8), MAIN_START, MAIN_END
        )

    # ------------------------------------------------------------- helpers
    @classmethod
    def make_engine(cls, *, explicit, top_n=8, risk_on_states=None):
        from strategies.hmm_volume import HMMVolumeStrategy

        from backtesting import Backtester

        strategy = HMMVolumeStrategy(
            cls.benchmark_price,
            cls.benchmark_volume,
            top_n=top_n,
            risk_on_states=risk_on_states,
        )
        if explicit:
            return Backtester(
                cls.prices,
                cls.benchmark,
                cls.membership,
                strategy,
                initial_cash=INITIAL_CASH,
                fee_bps=FEE_BPS,
            )
        return Backtester(
            market_data=cls.market_data,
            strategy=strategy,
            initial_cash=INITIAL_CASH,
            fee_bps=FEE_BPS,
        )

    @classmethod
    def drive(cls, engine, start, end):
        """Run one session at a time and record the decision-time view.

        ``cash_before`` is read before ``prepare`` so it shows the cash the
        strategy could have deployed; ``state`` is read after ``prepare`` so
        it is the state the strategy actually gated on.
        """
        rows = []
        engine.begin(start, end)
        while engine.next_session is not None:
            day = engine.next_session
            cash_before = engine.oms.available_cash
            engine.prepare()
            position = cls.calendar.get_loc(day)
            signal = cls.calendar[position - 1]
            strategy = engine.strategy
            rows.append(
                dict(
                    day=day,
                    signal=signal,
                    state=strategy.state,
                    state_date=strategy.state_date,
                    model_version=engine.model_version,
                    fit_bar_count=strategy._fit_bar_count,
                    cash_before=cash_before,
                    proposals=dict(engine.raw_proposals),
                )
            )
            engine.advance()
        result = engine.finish()
        records = pd.DataFrame(rows).set_index("day")
        return records, result, engine

    def reference_state_at(self, signal):
        """Walk-forward state of the last bar completed on or before signal."""
        ends = self.bars.end.to_numpy()
        i = int(np.searchsorted(ends, np.datetime64(signal), side="right")) - 1
        if i < 0:
            return -1
        return int(self.reference.states.iloc[i])

    def allowed_symbols(self, signal, execution):
        """Point-in-time eligible universe at one decision, independently."""
        quarter = pd.Timestamp(execution).to_period("Q").start_time
        members = set(
            self.membership.loc[
                self.membership.quarter_start.eq(quarter)
                & self.membership.selection_date.le(pd.Timestamp(signal)),
                "symbol",
            ]
        )
        last_prices = self.market_data.price_panel(end=signal).iloc[-1]
        return sorted(
            symbol
            for symbol in members
            if symbol in last_prices.index
            and np.isfinite(last_prices[symbol])
            and last_prices[symbol] > 0
        )

    def expected_liquidity_basket(self, signal, top_n):
        """Recompute the documented trailing dollar-volume ranking."""
        prices = self.market_data.price_panel(end=signal)
        volumes = self.market_data.volume_panel(end=signal).reindex(prices.index)
        window = min(63, len(prices))
        allowed = self.allowed_symbols(
            signal, self.calendar[self.calendar.get_loc(signal) + 1]
        )
        columns = [symbol for symbol in allowed if symbol in volumes.columns]
        dollar = (prices[columns].tail(window) * volumes[columns].tail(window)).mean()
        dollar = dollar.replace([np.inf, -np.inf], np.nan).dropna()
        ranked = dollar.sort_values(ascending=False, kind="stable").index.tolist()
        remainder = [symbol for symbol in allowed if symbol not in set(ranked)]
        return (ranked + remainder)[:top_n]

    # ---------------------------------------------------- series and panel
    def test_real_market_series_and_panel_are_consistent(self):
        price, volume = self.benchmark_price, self.benchmark_volume
        self.assertGreater(len(price), 3000)
        self.assertEqual(price.index.min(), pd.Timestamp("2000-01-03"))
        self.assertEqual(price.index.max(), pd.Timestamp(DATA_END))
        self.assertTrue(price.index.is_monotonic_increasing)
        self.assertTrue(price.index.equals(volume.index))
        self.assertTrue(np.isfinite(price.to_numpy()).all() and (price > 0).all())
        self.assertTrue(np.isfinite(volume.to_numpy()).all() and (volume >= 0).all())

        # The engine calendar and the strategy benchmark must agree exactly on
        # the run window, otherwise "the same bars" is not well defined.
        window = self.calendar[
            (self.calendar >= pd.Timestamp(MAIN_START))
            & (self.calendar <= pd.Timestamp(MAIN_END))
        ]
        self.assertTrue(window.isin(price.index).all())

        self.assertEqual(set(self.prices.symbol), set(SYMBOLS))
        coverage = self.prices.groupby("symbol").date.agg(["min", "max", "count"])
        self.assertTrue((coverage["min"] == pd.Timestamp("2000-01-03")).all())
        self.assertTrue((coverage["max"] == pd.Timestamp(DATA_END)).all())
        self.assertTrue((coverage["count"] == len(self.calendar)).all())
        self.assertTrue(self.prices.date.isin(self.calendar).all())

        # Every symbol is a member of every quarter of the attack window.
        quarters = pd.date_range("2012-01-01", "2014-10-01", freq="QS")
        counts = (
            self.membership.loc[self.membership.quarter_start.isin(quarters)]
            .groupby("quarter_start")
            .symbol.nunique()
        )
        self.assertTrue((counts.reindex(quarters) == len(SYMBOLS)).all())

        # The strategy builds its bars with the research rule (90 / 5).
        from research.bars import volume_bars

        rebuilt = volume_bars(
            self.benchmark_price, self.benchmark_volume, median_window=90,
            bucket_multiple=5.0,
        ).bars
        pd.testing.assert_frame_equal(self.bars, rebuilt)
        self.assertGreater(len(self.bars), 250)
        self.assertTrue((self.bars.end <= self.benchmark_price.index[-1]).all())

        self.assertEqual(self.main_result.coverage.initial_capital_date,
                         self.calendar[self.calendar.get_loc(pd.Timestamp(MAIN_START)) - 1])

    # -------------------------------------------------------- refit audit
    def test_refits_succeed_with_coherent_versions_and_audit(self):
        result, engine = self.main_result, self.main_engine
        refits = result.refits
        self.assertGreaterEqual(len(refits), 5, "too few refits to audit")
        self.assertTrue(refits.status.eq("success").all())
        self.assertEqual(refits.model_version.tolist(),
                         list(range(1, len(refits) + 1)))
        self.assertTrue(refits.signal_date.is_monotonic_increasing)
        self.assertTrue(refits.signal_date.is_unique)
        self.assertTrue(refits.signal_date.between(
            pd.Timestamp("2011-12-30"), pd.Timestamp(MAIN_END)).all())
        # Self-fitting strategy: no engine training panel is consumed.
        self.assertTrue((refits.rows == 0).all())
        for column in ("first_date", "last_date", "latest_label_end", "latest_available"):
            self.assertTrue(refits[column].isna().all())

        from strategies.hmm_volume import HMMVolumeStrategy

        baseline_settings = HMMVolumeStrategy(
            self.benchmark_price, self.benchmark_volume, top_n=8
        ).refit_settings()
        for settings in refits.settings:
            self.assertEqual(settings, baseline_settings)
        self.assertEqual(baseline_settings["median_window"], 90)
        self.assertEqual(baseline_settings["bucket_multiple"], 5.0)
        self.assertEqual(baseline_settings["n_states"], 3)
        self.assertEqual(baseline_settings["min_train_bars"], 250)
        self.assertEqual(baseline_settings["refit_every_bars"], 13)
        self.assertEqual(baseline_settings["risk_on_states"], [1, 2])

        # The recorded model version is the count of completed refits.
        records = self.main_records
        expected = pd.Series(
            [int((refits.signal_date <= signal).sum()) for signal in records.signal],
            index=records.index,
        )
        self.assertTrue(records.model_version.equals(expected))
        self.assertEqual(engine.model_version, len(refits))

        # Each fill carries the version in force when its order was decided.
        fills = result.fills
        expected_fill_versions = [
            int((refits.signal_date <= signal).sum()) for signal in fills.signal_date
        ]
        self.assertEqual(fills.model_version.tolist(), expected_fill_versions)

        # The fitted model is a real, finite Gaussian HMM.
        final = engine.strategy
        self.assertIsNotNone(final._fit)
        self.assertTrue(np.isfinite(final._fit.log_likelihood))
        self.assertEqual(final._fit.state_order.shape, (3,))
        self.assertLessEqual(final._fit_bar_count, len(self.bars))
        probabilities = final.probabilities
        self.assertEqual(probabilities.shape, (3,))
        self.assertAlmostEqual(float(probabilities.sum()), 1.0, places=10)
        self.assertIn(final.state, range(3))
        self.assertEqual(final.bars.shape, self.bars.shape)

    # ------------------------------------------------------------ ledger
    def test_fills_are_real_quotes_held_to_90_day_maturity(self):
        result = self.main_result
        fills = result.fills
        self.assertGreater(len(fills), 20, "run produced too few fills")
        self.assertEqual(set(fills.side), {"BUY", "SELL"})
        self.assertTrue(fills.strategy_id.eq("hmm_volume").all())
        self.assertTrue(fills.date.between(pd.Timestamp(MAIN_START), pd.Timestamp(MAIN_END)).all())
        self.assertTrue(fills.date.isin(self.calendar).all())

        # Every fill is the raw adjusted close of that symbol on that session.
        quotes = self.prices.set_index(["date", "symbol"]).adj_close
        for row in fills.itertuples():
            self.assertEqual(row.price, float(quotes.loc[(row.date, row.symbol)]),
                             f"fill price is not the panel quote on {row.date} {row.symbol}")

        buys = fills.loc[fills.side.eq("BUY")]
        sells = fills.loc[fills.side.eq("SELL")]
        self.assertFalse(buys.empty)
        self.assertFalse(sells.empty)

        # Buy accounting: notional + fee == cash spent, and units * price == notional.
        self.assertTrue(np.allclose(buys.notional + buys.fee, -buys.cash_change,
                                    rtol=0, atol=1e-8))
        self.assertTrue(np.allclose(buys.units * buys.price, buys.notional,
                                    rtol=1e-12, atol=0))
        self.assertTrue((buys.units > 0).all())

        # Maturity: every sale is a maturity sale, never before 90 calendar days.
        self.assertTrue(sells.trigger.eq("maturity").all())
        self.assertTrue((sells.date >= sells.due_date).all())
        delay_days = (sells.date - sells.due_date).dt.days
        self.assertTrue((delay_days >= 0).all())
        # due_date is fixed at entry, not at exit: entry + 90 calendar days.
        entries = buys.set_index("lot_id").date
        expected_due = entries.reindex(sells.lot_id).to_numpy() + pd.Timedelta(days=HOLD_DAYS)
        self.assertTrue((sells.due_date.to_numpy() == expected_due).all())
        # Every held name quotes daily here, so the exit is the first session
        # on or after maturity.
        first_session = self.calendar[self.calendar.searchsorted(sells.due_date.to_numpy())]
        self.assertTrue((first_session == sells.date.to_numpy()).all())

        # No lot is sold twice, every sale closes a real buy, and open lots are
        # exactly the purchases that have not matured yet.
        self.assertTrue(sells.lot_id.is_unique)
        self.assertTrue(set(sells.lot_id).issubset(set(buys.lot_id)))
        open_lots = set(result.open_lots.lot_id)
        self.assertEqual(set(buys.lot_id) - set(sells.lot_id), open_lots)
        # This three-year run ends exactly on the last maturity date, so every
        # lot opened in the window is closed inside it.
        self.assertTrue(result.open_lots.empty)

        # Buy/sell cash flows close the ledger.
        expected_cash = INITIAL_CASH + fills.cash_change.sum()
        final_cash = result.history.cash.iloc[-1]
        self.assertAlmostEqual(final_cash, expected_cash, places=6)
        self.assertTrue(np.allclose(sells.notional - sells.fee, sells.cash_change,
                                    rtol=0, atol=1e-8))

    def test_history_reconciles_cash_nav_and_pnl(self):
        result = self.main_result
        history, fills = result.history, result.fills
        self.assertEqual(history.index[0], pd.Timestamp("2011-12-30"))
        self.assertTrue(history.index.is_monotonic_increasing)
        self.assertTrue(np.allclose(history.cash + history.holdings_value, history.nav,
                                    rtol=0, atol=1e-8))
        self.assertTrue((history.cash >= 0).all())
        self.assertTrue(np.isfinite(history.nav).all() and (history.nav > 0).all())
        self.assertTrue((history.stale_marks == 0).all())
        self.assertTrue(result.diagnostics.empty)

        # Session cash equals opening capital plus every cash flow to date.
        flows = fills.groupby("date").cash_change.sum()
        replay = INITIAL_CASH + flows.reindex(history.index).fillna(0.0).cumsum()
        self.assertTrue(np.allclose(replay.to_numpy(), history.cash.to_numpy(),
                                    rtol=0, atol=1e-6))

        self.assertAlmostEqual(result.pnl.realized, fills.realized_pnl.sum(), places=6)
        self.assertAlmostEqual(result.pnl.unrealized, 0.0, places=6)
        self.assertAlmostEqual(result.pnl.total,
                               history.nav.iloc[-1] - INITIAL_CASH, places=6)
        self.assertAlmostEqual(result.pnl.total, result.pnl.realized, places=6)

        # One share of the benchmark is bought with the opening capital on the
        # first executed session, so its normalised NAV there is untouched.
        benchmark_nav = history.benchmark_nav
        self.assertTrue(np.isfinite(benchmark_nav).all() and (benchmark_nav > 0).all())
        first_day = pd.Timestamp(MAIN_START)
        self.assertEqual(history.loc[pd.Timestamp("2011-12-30"), "benchmark_nav"],
                         INITIAL_CASH)
        self.assertEqual(history.loc[first_day, "benchmark_nav"], INITIAL_CASH)
        # On the entry session the account NAV is the capital net of entry fees;
        # it is not out of line with the benchmark through any other channel.
        self.assertAlmostEqual(history.loc[first_day, "nav"],
                               INITIAL_CASH - history.loc[first_day, "fees"], places=6)
        self.assertGreater(history.fees.max(), 0.0)

    # -------------------------------------------- research state agreement
    def test_states_match_research_walk_forward_in_common_window(self):
        records = self.main_records
        expected = pd.Series(
            [self.reference_state_at(signal) for signal in records.signal],
            index=records.index,
        )
        self.assertTrue((expected >= 0).all(), "reference never fitted in window")
        mismatches = records.state.to_numpy() != expected.to_numpy()
        self.assertEqual(int(mismatches.sum()), 0,
                         f"state mismatches at:\n{records.loc[mismatches, ['state', 'state_date']]}")

        # The strategy never looks past the signal date for a state.
        self.assertTrue((records.state_date <= records.signal).all())
        self.assertTrue(records.state_date.isin(set(self.bars.end)).all())
        self.assertTrue(records.state_date.is_monotonic_increasing)
        self.assertEqual(int(records.state.max()), 2)

        # The refit lifecycle: the first fit needs one bar more than min_train
        # and every later fit happens exactly refit_every_bars new bars on.
        refits = self.main_result.refits
        ends = self.bars.end.to_numpy()
        refit_indices = [
            int(np.searchsorted(ends, np.datetime64(signal), side="right")) - 1
            for signal in refits.signal_date
        ]
        self.assertTrue(all(i >= 250 for i in refit_indices))
        self.assertEqual(np.diff(refit_indices).tolist(),
                         [13] * (len(refit_indices) - 1),
                         "refits must happen every refit_every_bars new bars")
        for signal, i in zip(refits.signal_date, refit_indices):
            self.assertGreaterEqual(signal, self.bars.end.iloc[i])
            if i + 1 < len(self.bars):
                self.assertLess(signal, self.bars.end.iloc[i + 1])

        # The stored model is exactly what the research fitter produces on the
        # bars strictly before the last scored bar.
        from research.hmm import fit_hmm, infer_state

        last_index = refit_indices[-1]
        final = self.main_engine.strategy._fit
        self.assertEqual(self.main_engine.strategy._fit_bar_count, last_index + 1)
        training = self.bars.log_return.to_numpy(dtype=float)[:last_index]
        expected_fit = fit_hmm(training, n_states=3, n_init=3, seed=7, n_iter=200)
        np.testing.assert_allclose(final.mean, expected_fit.mean)
        np.testing.assert_allclose(final.scale, expected_fit.scale)
        np.testing.assert_allclose(final.model.means_, expected_fit.model.means_)
        np.testing.assert_allclose(final.model.covars_, expected_fit.model.covars_)
        np.testing.assert_allclose(final.model.transmat_, expected_fit.model.transmat_)
        np.testing.assert_allclose(final.log_likelihood, expected_fit.log_likelihood)

        # After the last refit the strategy keeps scoring new completed bars
        # with that same model, so the published state is the score of every
        # bar completed on or before the final signal date.
        last_signal = records.signal.iloc[-1]
        completed = self.bars.loc[self.bars.end <= last_signal, "log_return"].to_numpy(
            dtype=float
        )
        np.testing.assert_allclose(
            self.main_engine.strategy._bar_returns, completed, rtol=0, atol=0
        )
        state, probabilities = infer_state(final, completed)
        self.assertEqual(state, self.main_engine.strategy.state)
        np.testing.assert_allclose(probabilities,
                                   self.main_engine.strategy.probabilities)

    # ------------------------------------------------------ future data
    def test_perturbing_real_data_after_cutoff_cannot_change_earlier_run(self):
        perturbed_price = self.benchmark_price.copy()
        perturbed_volume = self.benchmark_volume.copy()
        future = perturbed_price.index > CUTOFF
        self.assertGreater(int(future.sum()), 0)
        perturbed_price.loc[future] = perturbed_price.loc[future] * 1.7
        perturbed_volume.loc[future] = perturbed_volume.loc[future] * 3.0 + 1e9
        self.assertFalse(np.allclose(perturbed_price.loc[future],
                                     self.benchmark_price.loc[future]))
        self.assertFalse(np.allclose(perturbed_volume.loc[future],
                                     self.benchmark_volume.loc[future]))

        perturbed_prices = self.prices.copy()
        stock_future = perturbed_prices.date > CUTOFF
        self.assertGreater(int(stock_future.sum()), 0)
        for column, factor in (("adj_close", 0.55), ("close", 0.55), ("volume", 7.0)):
            perturbed_prices.loc[stock_future, column] = (
                perturbed_prices.loc[stock_future, column] * factor
            )

        from backtesting import Backtester
        from strategies.hmm_volume import HMMVolumeStrategy

        baseline = Backtester(
            self.prices, self.benchmark, self.membership,
            HMMVolumeStrategy(self.benchmark_price, self.benchmark_volume, top_n=8),
            initial_cash=INITIAL_CASH, fee_bps=FEE_BPS,
        ).run("2012-01-03", "2013-06-28")
        perturbed = Backtester(
            perturbed_prices, self.benchmark, self.membership,
            HMMVolumeStrategy(perturbed_price, perturbed_volume, top_n=8),
            initial_cash=INITIAL_CASH, fee_bps=FEE_BPS,
        ).run("2012-01-03", "2013-06-28")

        self.assertGreater(len(baseline.fills), 0, "inert baseline run")
        self.assertLessEqual(baseline.fills.date.max(), CUTOFF)
        pd.testing.assert_frame_equal(baseline.history, perturbed.history)
        pd.testing.assert_frame_equal(baseline.fills, perturbed.fills)
        pd.testing.assert_frame_equal(baseline.refits, perturbed.refits)
        pd.testing.assert_frame_equal(baseline.orders, perturbed.orders)
        self.assertEqual(baseline.history.nav.iloc[-1], perturbed.history.nav.iloc[-1])

    # ------------------------------------------------------------ gating
    def test_default_gate_skips_crisis_state_sessions_in_2011(self):
        records, result, _ = self.drive(
            self.make_engine(explicit=True, top_n=8), "2011-01-03", "2011-12-31"
        )
        self.assertGreater(len(result.fills), 0)
        self.assertIn(0, set(records.state), "crisis state never occurred; gate untested")

        crisis = records.loc[records.state.eq(0)]
        self.assertFalse(crisis.empty)
        self.assertTrue(all(not proposals for proposals in crisis.proposals))
        self.assertTrue((pd.Series(
            [self.reference_state_at(signal) for signal in crisis.signal]
        ) == 0).all())

        buys = result.fills.loc[result.fills.side.eq("BUY")]
        self.assertFalse(buys.empty)
        signal_state = records.set_index("signal").state
        bought_on = signal_state.loc[buys.signal_date.unique()]
        self.assertTrue(bought_on.isin([1, 2]).all())
        # No purchase can execute after a signal the strategy gated off.
        self.assertEqual(set(buys.signal_date) & set(crisis.signal), set())

    def test_explicit_policy_only_trades_the_top_state_on_real_data(self):
        records, result, engine = self.drive(
            self.make_engine(explicit=True, top_n=8, risk_on_states=(2,)),
            "2012-01-03", "2012-12-31",
        )
        self.assertEqual(engine.strategy.risk_on_states, (2,))
        self.assertEqual(set(records.state), {1, 2})
        self.assertGreater(len(result.fills), 0)

        blocked = records.loc[records.state.ne(2) & records.cash_before.gt(100.0)]
        self.assertFalse(blocked.empty, "no funded risk-off session; gate untested")
        self.assertTrue(all(not proposals for proposals in blocked.proposals))
        self.assertTrue(blocked.fit_bar_count.gt(0).all())

        funded_risk_on = records.loc[records.state.eq(2) & records.cash_before.gt(100.0)]
        self.assertFalse(funded_risk_on.empty)
        self.assertTrue(all(bool(proposals) for proposals in funded_risk_on.proposals))

        buys = result.fills.loc[result.fills.side.eq("BUY")]
        self.assertFalse(buys.empty)
        signal_state = records.set_index("signal").state
        self.assertTrue(signal_state.loc[buys.signal_date.unique()].eq(2).all())
        self.assertEqual(set(buys.signal_date) & set(blocked.signal), set())

    # ------------------------------------- engine explicit vs market_data
    def test_explicit_frames_and_market_data_agree_without_volume_ranking(self):
        explicit = self.make_engine(explicit=True, top_n=None)
        via_market_data = self.make_engine(explicit=False, top_n=None)
        self.assertIsNone(explicit._volume)
        self.assertIsNotNone(via_market_data._volume)
        self.assertFalse(via_market_data._volume.empty)

        left = explicit.run("2013-01-02", "2013-12-31")
        right = via_market_data.run("2013-01-02", "2013-12-31")
        self.assertGreater(len(left.fills), 0)
        pd.testing.assert_frame_equal(left.history, right.history)
        pd.testing.assert_frame_equal(left.fills, right.fills)
        pd.testing.assert_frame_equal(left.orders, right.orders)
        pd.testing.assert_frame_equal(left.refits, right.refits)

        # With top_n=None every eligible name is bought, so both paths traded
        # the whole 15-name universe.
        buys = left.fills.loc[left.fills.side.eq("BUY")]
        self.assertEqual(buys.groupby("signal_date").symbol.nunique().max(), len(SYMBOLS))

    def test_volume_panel_only_changes_the_documented_liquidity_basket(self):
        top_n = 6
        explicit = self.make_engine(explicit=True, top_n=top_n).run(
            "2013-01-02", "2013-06-28"
        )
        via_market_data = self.make_engine(explicit=False, top_n=top_n).run(
            "2013-01-02", "2013-06-28"
        )
        self.assertGreater(len(explicit.fills), 0)
        self.assertGreater(len(via_market_data.fills), 0)

        explicit_buys = explicit.fills.loc[explicit.fills.side.eq("BUY")]
        market_buys = via_market_data.fills.loc[via_market_data.fills.side.eq("BUY")]
        explicit_baskets = explicit_buys.groupby("signal_date").symbol.apply(sorted)
        market_baskets = market_buys.groupby("signal_date").symbol.apply(sorted)
        self.assertTrue(len(explicit_baskets) > 0)

        for signal, basket in explicit_baskets.items():
            execution = self.calendar[self.calendar.get_loc(signal) + 1]
            allowed = self.allowed_symbols(signal, execution)
            self.assertEqual(basket, sorted(allowed)[:top_n])
            self.assertEqual(len(basket), top_n)

        for signal, basket in market_baskets.items():
            expected = self.expected_liquidity_basket(signal, top_n)
            self.assertEqual(basket, sorted(expected))
            self.assertEqual(len(basket), top_n)

        self.assertNotEqual(set(explicit_baskets.iloc[0]), set(market_baskets.iloc[0]))


if __name__ == "__main__":
    unittest.main()
