"""Adversarial lookahead / causality audit for ``strategies.hmm_volume``.

Every test in this module is written to FAIL if the strategy can see, train on,
or be influenced by any observation dated after ``context.signal_date``.  The
tests deliberately go beyond the happy path: they rebuild the strategy from
truncated history, perturb future benchmark prices, benchmark volume, stock
prices and stock volume, spy on the exact arrays handed to the HMM, and compare
the state path against an independent walk-forward implementation.

Nothing here is allowed to be weakened to make the implementation pass; a red
test is a defect report, not a test to relax.
"""

from __future__ import annotations

import unittest
from unittest import mock

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from backtesting import Backtester, DecisionContext, MarketData, Portfolio
from research.hmm import fit_hmm, infer_state, walk_forward_states
from strategies import hmm_volume as hmm_volume_module
from strategies.hmm_volume import HMMVolumeStrategy


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------
SYMBOLS = ("S0", "S1", "S2", "S3", "S4")


def synthetic_market(periods=900, symbols=SYMBOLS, seed=11):
    """A small but realistic market: benchmark, volume, stocks, membership."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-01", periods=periods)
    benchmark = pd.Series(
        100 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, periods))), index=dates
    )
    volume = pd.Series(rng.lognormal(15, 0.3, periods), index=dates)
    price_rows = []
    for symbol in symbols:
        prices = benchmark.to_numpy() * np.exp(
            np.cumsum(rng.normal(0, 0.005, periods))
        )
        stock_volume = rng.lognormal(13, 0.4, periods)
        price_rows.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "symbol": symbol,
                    "adj_close": prices,
                    "close": prices,
                    "volume": stock_volume,
                }
            )
        )
    prices = pd.concat(price_rows, ignore_index=True)
    quarters = pd.date_range(dates[0], dates[-1], freq="QS")
    membership = pd.concat(
        [
            pd.DataFrame(
                {
                    "symbol": list(symbols),
                    "quarter_start": quarter,
                    "selection_date": quarter - pd.Timedelta(days=1),
                }
            )
            for quarter in quarters
        ],
        ignore_index=True,
    )
    universe = pd.DataFrame({"symbol": list(symbols), "sector": "Tech"})
    fundamentals = pd.DataFrame(
        {
            "symbol": list(symbols),
            "fiscal_date": dates[0],
            "filing_date": dates[0],
            "period": "quarter",
            "revenue": 1.0,
            "net_income": 0.1,
            "operating_income": 0.1,
            "total_assets": 2.0,
            "total_debt": 0.5,
            "cash_and_equivalents": 0.3,
            "free_cash_flow": 0.1,
            "operating_cash_flow": 0.1,
        }
    )
    return dict(
        dates=dates,
        benchmark=benchmark,
        volume=volume,
        prices=prices,
        benchmark_frame=pd.DataFrame(
            {"date": dates, "adj_close": benchmark.to_numpy()}
        ),
        membership=membership,
        universe=universe,
        fundamentals=fundamentals,
    )


OPTIONS = dict(
    median_window=20,
    bucket_multiple=5.0,
    n_states=3,
    min_train_bars=40,
    refit_every_bars=7,
    n_init=1,
    n_iter=60,
    seed=3,
    top_n=3,
    risk_on_states=(1, 2),
)


def build(market, *, benchmark=None, volume=None, **overrides):
    options = dict(OPTIONS)
    options.update(overrides)
    return HMMVolumeStrategy(
        market["benchmark"] if benchmark is None else benchmark,
        market["volume"] if volume is None else volume,
        **options,
    )


def price_panel(prices, signal):
    return (
        prices.loc[prices["date"] <= pd.Timestamp(signal)]
        .pivot(index="date", columns="symbol", values="adj_close")
    )


def membership_at(membership, signal, execution):
    quarter = pd.Timestamp(execution).to_period("Q").start_time
    return membership.loc[
        membership["quarter_start"].eq(quarter)
        & membership["selection_date"].le(pd.Timestamp(signal))
    ].copy()


def next_session(dates, signal):
    position = dates.get_loc(pd.Timestamp(signal))
    if position + 1 < len(dates):
        return dates[position + 1]
    return pd.Timestamp(signal) + pd.Timedelta(days=1)


def make_context(market, signal, *, cash=100_000.0, volume_panel=None):
    signal = pd.Timestamp(signal)
    execution = next_session(market["dates"], signal)
    prices = price_panel(market["prices"], signal)
    if volume_panel is None:
        volume = pd.DataFrame()
    else:
        volume = volume_panel.loc[volume_panel.index <= signal].copy()
    return DecisionContext(
        signal,
        execution,
        prices,
        membership_at(market["membership"], signal, execution),
        pd.DataFrame(),
        Portfolio(cash, cash, cash, {}),
        0,
        volume,
        pd.DataFrame(),
    )


def drive(model, market, signals, *, volume_panel=None, cash=100_000.0):
    """Replay the engine loop (needs_refit -> refit -> propose) in order."""
    history = []
    for signal in signals:
        context = make_context(market, signal, cash=cash, volume_panel=volume_panel)
        refit = bool(model.needs_refit(context))
        if refit:
            model.refit(None, context)
        budgets = model.propose_buys(context)
        history.append(
            dict(
                signal=pd.Timestamp(signal),
                refit=refit,
                state=model.state,
                state_date=model.state_date,
                probabilities=model.probabilities,
                budgets=dict(budgets),
            )
        )
    return history


class HMMVolumeCausalityChecks(unittest.TestCase):
    """Every check attacks the "no future data" claim from a different side."""

    # -- 1. the precomputed bar table -------------------------------------
    def test_completed_bars_are_bit_identical_under_future_perturbations(self):
        """Future price/volume must not move a single past bar number."""
        market = synthetic_market()
        base = build(market)
        self.assertGreater(len(base.bars), 120)
        for index in (20, 45, 70, 100):
            signal = base.bars["end"].iloc[index]
            for factor in (1e-6, 1e6):
                with self.subTest(index=index, signal=signal.date(), factor=factor):
                    changed_benchmark = market["benchmark"].copy()
                    changed_volume = market["volume"].copy()
                    future = changed_benchmark.index > signal
                    changed_benchmark.loc[future] *= factor
                    changed_volume.loc[future] *= factor
                    other = build(market, benchmark=changed_benchmark, volume=changed_volume)
                    assert_frame_equal(
                        other._bars_at(signal), base._bars_at(signal), check_exact=True
                    )

    def test_future_volume_cannot_change_completed_bar_boundaries(self):
        """A bar's start/end/target must be fixed by data at its own start."""
        market = synthetic_market()
        base = build(market)
        cut = base.bars["end"].iloc[60]
        changed = market["volume"].copy()
        changed.loc[changed.index > cut] = 0.0
        other = build(market, volume=changed)
        prefix = other._bars_at(cut)
        assert_frame_equal(prefix, base._bars_at(cut), check_exact=True)
        self.assertTrue((prefix["end"] <= cut).all())
        self.assertTrue((prefix["start"] <= prefix["end"]).all())

    def test_bars_built_from_truncated_history_equal_the_full_prefix(self):
        """The full-history bar table is a causal extension of the truncated one."""
        market = synthetic_market()
        base = build(market)
        for index in (25, 50, 75, 110):
            signal = base.bars["end"].iloc[index]
            with self.subTest(index=index):
                truncated = build(
                    market,
                    benchmark=market["benchmark"].loc[:signal],
                    volume=market["volume"].loc[:signal],
                )
                assert_frame_equal(
                    truncated._bars, base._bars_at(signal), check_exact=True
                )

    # -- 2. which bars a signal date may see ------------------------------
    def test_signal_date_only_uses_bars_completed_on_or_before_it(self):
        market = synthetic_market()
        model = build(market)
        bars = model.bars
        for index in (30, 60, 90, 120):
            end = bars["end"].iloc[index]
            # A session strictly inside the *next* bar: nothing new completed.
            inside = bars["start"].iloc[index + 1]
            with self.subTest(index=index):
                inside_bars = model._bars_at(inside)
                assert_frame_equal(
                    inside_bars, model._bars_at(end), check_exact=True
                )
                self.assertTrue((inside_bars["end"] <= inside).all())
                self.assertNotIn(bars["bar_id"].iloc[index + 1], set(inside_bars["bar_id"]))
                self.assertEqual(
                    len(model._bars_at(end)), int((bars["end"] <= end).sum())
                )

    def test_signal_before_first_bar_and_after_last_bar(self):
        market = synthetic_market()
        model = build(market)
        first_start = model.bars["start"].iloc[0]
        early = next_session(market["dates"], market["dates"][0])  # well before bars
        self.assertLess(early, first_start)
        context = make_context(market, early)
        self.assertEqual(len(model._bars_at(early)), 0)
        self.assertFalse(model.needs_refit(context))
        self.assertEqual(model.propose_buys(context), {})
        with self.assertRaises(ValueError):
            model.refit(None, context)  # the engine never calls this, the guard holds

        late = market["dates"][-1] + pd.Timedelta(days=30)
        self.assertGreater(late, model.bars["end"].iloc[-1])
        assert_frame_equal(model._bars_at(late), model.bars, check_exact=True)

    # -- 3. what the HMM is actually trained on ---------------------------
    def test_fit_hmm_receives_only_returns_strictly_before_current_bar(self):
        """Spy on the real call sites: no future bar can reach the estimator."""
        market = synthetic_market()
        model = build(market)
        recorded = {"signal": None, "fit": [], "infer": []}
        real_fit, real_infer = hmm_volume_module.fit_hmm, hmm_volume_module.infer_state

        def spy_fit(values, **kwargs):
            recorded["fit"].append(
                (recorded["signal"], np.array(values, dtype=float, copy=True))
            )
            return real_fit(values, **kwargs)

        def spy_infer(fit, values):
            recorded["infer"].append(
                (recorded["signal"], np.array(values, dtype=float, copy=True))
            )
            return real_infer(fit, values)

        signals = [
            signal
            for signal in market["dates"][::9]
            if model.bars["end"].iloc[0] <= signal <= model.bars["end"].iloc[-1]
        ]
        with mock.patch.object(hmm_volume_module, "fit_hmm", spy_fit), mock.patch.object(
            hmm_volume_module, "infer_state", spy_infer
        ):
            for signal in signals:
                recorded["signal"] = pd.Timestamp(signal)
                context = make_context(market, signal)
                if model.needs_refit(context):
                    model.refit(None, context)
                model.propose_buys(context)

        self.assertTrue(recorded["fit"], "the run never refitted; test is vacuous")
        self.assertTrue(recorded["infer"], "the run never inferred; test is vacuous")
        for signal, values in recorded["fit"]:
            expected = model._bars_at(signal)["log_return"].to_numpy()
            self.assertEqual(values.size, expected.size - 1)
            np.testing.assert_array_equal(values, expected[:-1])
        for signal, values in recorded["infer"]:
            expected = model._bars_at(signal)["log_return"].to_numpy()
            np.testing.assert_array_equal(values, expected)

    def test_fitted_model_equals_a_fit_on_all_but_the_current_bar(self):
        """One-step-ahead, not in-sample and not two-step-ahead."""
        market = synthetic_market()
        model = build(market)
        signal = model.bars["end"].iloc[70]
        model.refit(None, make_context(market, signal))
        returns = model._bars_at(signal)["log_return"].to_numpy()
        np.testing.assert_array_equal(model._bar_returns, returns)

        # NOTE: hmmlearn's own EM is not bit-reproducible (two fits of a
        # byte-identical array differ by ~7e-15 in means on this machine), so a
        # re-fit is compared with a tolerance far tighter than any real data
        # leak could produce (an in-sample fit moves the means by O(1)).
        expected = fit_hmm(
            returns[:-1], n_states=3, n_init=1, seed=3, n_iter=60
        )
        np.testing.assert_allclose(
            model._fit.model.means_, expected.model.means_, rtol=1e-9, atol=1e-9
        )
        np.testing.assert_allclose(
            model._fit.model.covars_, expected.model.covars_, rtol=1e-9, atol=1e-9
        )
        np.testing.assert_allclose(
            model._fit.model.transmat_, expected.model.transmat_, rtol=1e-9, atol=1e-9
        )
        self.assertAlmostEqual(
            model._fit.log_likelihood, expected.log_likelihood, places=6
        )

        in_sample = fit_hmm(returns, n_states=3, n_init=1, seed=3, n_iter=60)
        self.assertFalse(
            np.allclose(in_sample.model.means_, model._fit.model.means_, atol=1e-3),
            "in-sample and walk-forward fits coincide; the test would not detect "
            "an in-sample fit",
        )
        reference_state, reference_probabilities = infer_state(expected, returns)
        self.assertEqual(int(model.state), int(reference_state))
        np.testing.assert_allclose(
            model.probabilities, reference_probabilities, rtol=0, atol=1e-9
        )

    def test_changing_the_current_bar_close_cannot_change_the_fitted_model(self):
        """Only the *scored* bar may react to its own close; training must not."""
        market = synthetic_market()
        model = build(market)
        signal = model.bars["end"].iloc[70]
        context = make_context(market, signal)

        # A price shock exactly on the bar-end date changes only the return of
        # that bar: bars are bucketed by volume, not price.
        shocked = market["benchmark"].copy()
        shocked.loc[signal] *= 1.5
        other = build(market, benchmark=shocked)

        model.refit(None, context)
        other.refit(None, context)

        self.assertNotEqual(model._bar_returns[-1], other._bar_returns[-1])
        np.testing.assert_array_equal(
            model._bar_returns[:-1], other._bar_returns[:-1]
        )
        # See the note in test_fitted_model_equals_a_fit_on_all_but_the_current_bar:
        # hmmlearn's EM is not bit-reproducible, so identical inputs are compared
        # with a tolerance that is still ~12 orders of magnitude tighter than the
        # shift an in-sample (leaky) fit would introduce.
        np.testing.assert_allclose(
            model._fit.model.means_, other._fit.model.means_, rtol=1e-9, atol=1e-9
        )
        np.testing.assert_allclose(
            model._fit.model.covars_, other._fit.model.covars_, rtol=1e-9, atol=1e-9
        )
        np.testing.assert_allclose(
            model._fit.model.transmat_, other._fit.model.transmat_, rtol=1e-9, atol=1e-9
        )
        self.assertAlmostEqual(
            model._fit.log_likelihood, other._fit.log_likelihood, places=6
        )
        # ... while the scored state may legitimately move.
        self.assertEqual(other.state_date, signal)

    # -- 4. the state path -------------------------------------------------
    def test_state_path_matches_an_independent_walk_forward_reference(self):
        """The state at bar t must equal a reference fit on bars < t only."""
        market = synthetic_market(periods=1200, seed=21)
        model = build(market)
        bars = model.bars
        reference = walk_forward_states(
            bars["log_return"].to_numpy(),
            bars["end"],
            min_train=OPTIONS["min_train_bars"],
            refit_every=OPTIONS["refit_every_bars"],
            n_states=OPTIONS["n_states"],
            n_init=OPTIONS["n_init"],
            seed=OPTIONS["seed"],
            n_iter=OPTIONS["n_iter"],
        )
        for index in range(len(bars)):
            signal = bars["end"].iloc[index]
            context = make_context(market, signal)
            if model.needs_refit(context):
                model.refit(None, context)
            model.propose_buys(context)
            expected = int(reference.states.iloc[index])
            if expected < 0:
                self.assertIsNone(model.state)
            else:
                self.assertEqual(
                    model.state, expected, msg=f"bar index {index} @ {signal}"
                )

    def test_future_data_cannot_change_past_states_or_proposals(self):
        """Replay the whole run with the future rewritten; the past must match."""
        market = synthetic_market(seed=31)
        model = build(market)
        signals = [
            signal
            for signal in market["dates"][::3]
            if signal >= model.bars["end"].iloc[45]
            and signal <= model.bars["end"].iloc[110]
        ]
        self.assertGreater(len(signals), 10)
        baseline = drive(model, market, signals)

        for cut in (signals[len(signals) // 3], signals[2 * len(signals) // 3]):
            with self.subTest(cut=cut):
                changed_benchmark = market["benchmark"].copy()
                changed_volume = market["volume"].copy()
                future = changed_benchmark.index > cut
                changed_benchmark.loc[future] *= 1e6
                changed_volume.loc[future] *= 1e6
                changed_market = dict(market)
                changed_market["benchmark"] = changed_benchmark
                changed_market["volume"] = changed_volume
                changed_model = build(changed_market)
                replay = drive(changed_model, changed_market, signals)
                for before, after in zip(baseline, replay):
                    if before["signal"] > cut:
                        break
                    self.assertEqual(before["refit"], after["refit"])
                    self.assertEqual(before["state"], after["state"])
                    self.assertEqual(before["state_date"], after["state_date"])
                    self.assertEqual(before["budgets"], after["budgets"])
                    if before["probabilities"] is not None:
                        # Tolerance only covers hmmlearn's ~1e-16 numerical
                        # drift; a real leak shifts posteriors by ~1e-2.
                        np.testing.assert_allclose(
                            after["probabilities"],
                            before["probabilities"],
                            rtol=0,
                            atol=1e-9,
                        )

    def test_truncated_history_replay_equals_full_history_replay(self):
        """Rebuild the strategy from data <= cut; every decision must match."""
        market = synthetic_market(seed=41)
        model = build(market)
        first = model.bars["end"].iloc[45]
        last = model.bars["end"].iloc[110]
        signals = [
            signal
            for signal in market["dates"][::4]
            if first <= signal <= last
        ]
        full = drive(model, market, signals)

        for cut in (signals[len(signals) // 2], signals[-3]):
            with self.subTest(cut=cut):
                truncated_market = dict(market)
                truncated_market["benchmark"] = market["benchmark"].loc[:cut]
                truncated_market["volume"] = market["volume"].loc[:cut]
                truncated_model = build(truncated_market)
                truncated_signals = [signal for signal in signals if signal <= cut]
                replay = drive(truncated_model, truncated_market, truncated_signals)
                for before, after in zip(full, replay):
                    self.assertEqual(before["refit"], after["refit"])
                    self.assertEqual(before["state"], after["state"])
                    self.assertEqual(before["state_date"], after["state_date"])
                    self.assertEqual(before["budgets"], after["budgets"])

    def test_future_stock_volume_cannot_reorder_the_liquidity_basket(self):
        """``_rank_by_liquidity`` must read a panel that stops at the signal."""
        market = synthetic_market(seed=51)
        model = build(market, top_n=2)
        signal = model.bars["end"].iloc[80]
        # Panels differ only after the signal date.
        base_panel = pd.DataFrame(
            {symbol: 1.0 for symbol in SYMBOLS}, index=market["dates"]
        )
        base_panel["S0"] = 500.0
        base_panel["S1"] = 100.0
        changed_panel = base_panel.copy()
        changed_panel.loc[changed_panel.index > signal, "S4"] = 1e9
        changed_panel.loc[changed_panel.index > signal, "S3"] = 1e9

        model._fit = "placeholder"  # bypass the HMM; only ranking is under test
        model._state = 2
        model._state_bar_count = len(model.bars)
        base = model.propose_buys(
            make_context(market, signal, volume_panel=base_panel)
        )
        changed = model.propose_buys(
            make_context(market, signal, volume_panel=changed_panel)
        )
        self.assertEqual(base, changed)
        self.assertEqual(set(base), {"S0", "S1"})

    def test_state_timeline_never_runs_ahead_of_the_signal_date(self):
        """A state may only advance when a new bar has completed on or before it."""
        market = synthetic_market(seed=81)
        model = build(market)
        bars = model.bars
        signals = [
            signal
            for signal in market["dates"]
            if bars["end"].iloc[40] <= signal <= bars["end"].iloc[110]
        ]
        previous_state_date = None
        previous_state = None
        completed = set(bars["end"])
        moved = 0
        for signal in signals:
            context = make_context(market, signal)
            if model.needs_refit(context):
                model.refit(None, context)
            model.propose_buys(context)
            if model.state_date is None:
                continue
            self.assertLessEqual(model.state_date, signal)
            self.assertIn(model.state_date, completed)
            if previous_state_date is not None and model.state_date != previous_state_date:
                moved += 1
                # A change of state_date requires a session that completed a bar.
                self.assertGreater(model.state_date, previous_state_date)
            if model.state_date == previous_state_date:
                self.assertEqual(model.state, previous_state)
            previous_state_date, previous_state = model.state_date, model.state
        self.assertGreater(moved, 20)

    # -- 5. the engine must not hand future rows to the strategy -----------
    def test_engine_never_exposes_future_rows_to_the_strategy(self):
        market = synthetic_market(seed=61)
        model = build(market)

        class AuditingStrategy(HMMVolumeStrategy):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.audit = []

            def propose_buys(self, context):
                budgets = super().propose_buys(context)
                self.audit.append(
                    dict(
                        signal=context.signal_date,
                        execution=context.execution_date,
                        prices_end=context.prices.index.max(),
                        volume_end=(
                            context.volume.index.max()
                            if not context.volume.empty
                            else pd.NaT
                        ),
                        selection_end=context.membership["selection_date"].max(),
                        state_date=self.state_date,
                        budgets=dict(budgets),
                    )
                )
                return budgets

        audited = AuditingStrategy(
            market["benchmark"], market["volume"], **OPTIONS
        )
        engine = Backtester(
            market["prices"],
            market["benchmark_frame"],
            market["membership"],
            audited,
            initial_cash=100_000,
            fee_bps=0,
        )
        engine.run(market["dates"][350], market["dates"][800])
        audit = engine.strategy.audit
        self.assertGreater(len(audit), 100)
        signals_seen = [row["signal"] for row in audit]
        self.assertTrue(all(
            later > earlier
            for earlier, later in zip(signals_seen, signals_seen[1:])
        ), "the engine must call the strategy in strictly increasing signal order")
        for row in audit:
            signal = row["signal"]
            self.assertLess(row["execution"], signal + pd.Timedelta(days=5))
            self.assertGreater(row["execution"], signal)
            self.assertLessEqual(row["prices_end"], signal)
            if not pd.isna(row["volume_end"]):
                self.assertLessEqual(row["volume_end"], signal)
            if not pd.isna(row["selection_end"]):
                self.assertLessEqual(row["selection_end"], signal)
            if row["state_date"] is not None:
                self.assertLessEqual(row["state_date"], signal)

    # -- 6. end-to-end invariance -----------------------------------------
    def test_backtest_is_invariant_to_future_benchmark_and_stock_prices(self):
        market = synthetic_market(seed=71)
        start, end = market["dates"][300], market["dates"][700]
        cut = end

        def run(benchmark, prices):
            model = build(
                market, benchmark=benchmark, min_train_bars=40, top_n=3
            )
            benchmark_frame = pd.DataFrame(
                {"date": benchmark.index, "adj_close": benchmark.to_numpy()}
            )
            return Backtester(
                prices,
                benchmark_frame,
                market["membership"],
                model,
                initial_cash=100_000,
                fee_bps=5,
            ).run(start, end)

        baseline = run(market["benchmark"], market["prices"])
        self.assertFalse(baseline.fills.empty)

        changed_benchmark = market["benchmark"].copy()
        changed_benchmark.loc[changed_benchmark.index > cut] *= 1e3
        changed_prices = market["prices"].copy()
        future = changed_prices["date"] > cut
        changed_prices.loc[future, "adj_close"] *= 1e3
        changed_prices.loc[future, "close"] *= 1e3
        replay = run(changed_benchmark, changed_prices)

        assert_frame_equal(baseline.fills, replay.fills, check_exact=True)
        assert_frame_equal(baseline.orders, replay.orders, check_exact=True)
        assert_frame_equal(baseline.history, replay.history, check_exact=True)
        assert_frame_equal(baseline.refits, replay.refits, check_exact=True)

    def test_backtest_is_invariant_to_future_stock_volume_panel(self):
        market = synthetic_market(seed=81)
        start, end = market["dates"][300], market["dates"][700]
        cut = end

        def market_data(prices):
            return MarketData.from_frames(
                prices,
                market["fundamentals"],
                market["universe"],
                market["benchmark_frame"],
                market["membership"],
                drop_non_usd=False,
            )

        def run(prices):
            model = build(market, min_train_bars=40, top_n=3)
            return Backtester(
                market_data=market_data(prices),
                strategy=model,
                initial_cash=100_000,
                fee_bps=5,
            ).run(start, end)

        baseline = run(market["prices"])
        self.assertFalse(baseline.fills.empty)

        changed_prices = market["prices"].copy()
        future = changed_prices["date"] > cut
        changed_prices.loc[future, "volume"] *= 1e6
        replay = run(changed_prices)

        assert_frame_equal(baseline.fills, replay.fills, check_exact=True)
        assert_frame_equal(baseline.orders, replay.orders, check_exact=True)
        assert_frame_equal(baseline.history, replay.history, check_exact=True)

    def test_needs_refit_sequence_is_independent_of_future_data(self):
        """The refit calendar up to a date cannot depend on later observations."""
        market = synthetic_market(seed=91)
        base = build(market)
        signals = [
            signal
            for signal in market["dates"][::2]
            if signal <= base.bars["end"].iloc[110]
        ]
        cut = signals[-1]
        changed_benchmark = market["benchmark"].copy()
        changed_volume = market["volume"].copy()
        changed_benchmark.loc[changed_benchmark.index > cut] *= 1e6
        changed_volume.loc[changed_volume.index > cut] *= 1e6
        changed_market = dict(market)
        changed_market["benchmark"] = changed_benchmark
        changed_market["volume"] = changed_volume
        other = build(changed_market)

        # Lockstep replay: identical decisions must keep the two models in the
        # same internal state, so the refit schedule stays comparable.
        decisions = []
        for signal in signals:
            context = make_context(market, signal)
            changed_context = make_context(changed_market, signal)
            left = bool(base.needs_refit(context))
            right = bool(other.needs_refit(changed_context))
            if signal <= cut:
                decisions.append((left, right))
                self.assertEqual(left, right, msg=f"refit flag at {signal}")
            if left:
                base.refit(None, context)
            if right:
                other.refit(None, changed_context)
            base.propose_buys(context)
            other.propose_buys(changed_context)
            if signal <= cut:
                self.assertEqual(base.state, other.state, msg=f"state at {signal}")
        # Sanity: the schedule actually refit more than once before the cut.
        self.assertGreater(sum(left for left, _ in decisions), 1)


if __name__ == "__main__":
    unittest.main()
