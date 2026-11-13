"""Round-2 adversarial API, deepcopy, and integration contract tests.

Scope: the public lifecycle of ``strategies.hmm_volume.HMMVolumeStrategy`` and
its use through ``backtesting`` (``Backtester``, ``DecisionContext``,
``MarketData``) and ``ShadowPortfolioBacktester``.  Every fixture is synthetic.

Two earlier defects are fixed and pinned in ``DefectChecks``. The third test
documents the engine-consumption contract (pass a fresh deepcopy per run):

* ``test_refit_cannot_rewind_below_the_last_scored_bar`` --
  ``refit`` only rejects a signal earlier than the last *fit* date, so a
  caller can refit at a signal earlier than the last bar that ``propose_buys``
  already scored and silently rewind the model state.
* ``test_nat_signal_date_is_rejected`` --
  a missing (``NaT``/``NaN``) signal date is accepted and the strategy
  returns a full purchase basket.
* ``test_engine_does_not_mutate_a_prefitted_caller_strategy`` --
  ``Backtester`` keeps the caller's strategy object and calls
  ``propose_buys`` on it whenever no refit is due, so a pre-fitted caller
  model is mutated in place and cannot be reused for a second run.
"""

import unittest
import warnings
from copy import deepcopy

import numpy as np
import pandas as pd

from backtesting import (
    Backtester,
    DecisionContext,
    MarketData,
    Phase,
    Portfolio,
    ShadowPortfolioBacktester,
    ShadowSpec,
)
from backtesting.contracts import FitData
from strategies.hmm_volume import HMMVolumeStrategy


FAST = dict(
    median_window=20,
    bucket_multiple=5.0,
    n_states=3,
    min_train_bars=30,
    refit_every_bars=10,
    n_init=1,
    n_iter=50,
    top_n=2,
    risk_on_states=(0, 1, 2),
)


def synthetic_market(periods=560, symbols=4, seed=21, start="2016-01-01"):
    """One calendar, a benchmark, and a long-form price panel.

    Dates are normalized to nanosecond resolution so that a ``MarketData``
    round trip (which casts to ns) and the explicit-frame path produce the
    same calendar dtype and can be compared directly.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=periods)
    benchmark = 100.0 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, periods)))
    volume = pd.Series(rng.lognormal(15.0, 0.3, periods), index=dates)

    rows = []
    for symbol in [f"S{i}" for i in range(symbols)]:
        path = benchmark * np.exp(np.cumsum(rng.normal(0.0, 0.004, periods)))
        rows.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "symbol": symbol,
                    "adj_close": path,
                    "close": path,
                    "volume": 1_000.0,
                }
            )
        )
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
    for frame, columns in [
        (prices, ["date"]),
        (benchmark_frame, ["date"]),
        (membership, ["quarter_start", "selection_date"]),
    ]:
        for column in columns:
            frame[column] = pd.to_datetime(frame[column]).astype("datetime64[ns]")
    universe = pd.DataFrame(
        {"symbol": [f"S{i}" for i in range(symbols)], "sector": "Tech"}
    )
    return dict(
        dates=dates,
        benchmark_frame=benchmark_frame,
        benchmark=pd.Series(benchmark, index=dates),
        volume=volume,
        prices=prices,
        membership=membership,
        universe=universe,
    )


def make_model(market, **overrides):
    options = dict(FAST)
    options.update(overrides)
    return HMMVolumeStrategy(market["benchmark"], market["volume"], **options)


def make_engine(market, model, prices=None, **kwargs):
    return Backtester(
        market["prices"][["date", "symbol", "adj_close"]] if prices is None else prices,
        market["benchmark_frame"],
        market["membership"],
        model,
        initial_cash=kwargs.pop("initial_cash", 100_000),
        **kwargs,
    )


def make_context(model, signal, market, *, cash=100_000.0, volume=None,
                 membership=None, prices=None):
    """A hand-built decision context for direct-API tests."""
    signal = pd.Timestamp(signal)
    source = market["prices"] if prices is None else prices
    if "date" in source.columns:
        view = source.loc[source["date"] <= signal].pivot(
            index="date", columns="symbol", values="adj_close"
        )
    else:
        view = source.loc[:signal].copy()
    execution = signal + pd.Timedelta(days=1)
    return DecisionContext(
        signal,
        execution,
        view,
        (market["membership"] if membership is None else membership).copy(),
        pd.DataFrame(),
        Portfolio(cash, cash, cash, {}),
        0,
        pd.DataFrame() if volume is None else volume.loc[:signal].copy(),
        pd.DataFrame(),
    )


class StrategyApiChecks(unittest.TestCase):
    """self_fitting, refit ordering, audit settings, and public properties."""

    @classmethod
    def setUpClass(cls):
        cls.market = synthetic_market()
        cls.dates = cls.market["dates"]

    def test_self_fitting_refit_ignores_engine_fit_data(self):
        model = make_model(self.market)
        self.assertIs(HMMVolumeStrategy.self_fitting, True)
        signal = model.bars["end"].iloc[40]
        context = make_context(model, signal, self.market)
        model.refit(None, context)
        first = (model.state, model.probabilities.copy(), model._fit.log_likelihood)
        # A self-fitting strategy must not consume the engine's FitData.
        fit_data = FitData(
            pd.DataFrame({"x": [1.0, 2.0]}),
            pd.Series([0.0, 1.0]),
            pd.DataFrame(
                {
                    "date": [signal, signal],
                    "symbol": ["S0", "S1"],
                    "available_date": [signal, signal],
                    "label_end": [signal, signal],
                }
            ),
        )
        self.assertIsNone(model.refit(fit_data, context))
        second = (model.state, model.probabilities.copy(), model._fit.log_likelihood)
        self.assertEqual(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])
        self.assertEqual(first[2], second[2])

    def test_needs_refit_is_pure_and_lifecycle_is_ordered(self):
        model = make_model(self.market)
        bars = model.bars
        before = (model.state, model.state_date, model.probabilities,
                  model._fit_bar_count)
        early = make_context(model, bars["end"].iloc[model.min_train_bars - 2],
                             self.market)
        for _ in range(3):
            self.assertFalse(model.needs_refit(early))
        self.assertEqual(
            (model.state, model.state_date, model.probabilities,
             model._fit_bar_count),
            before,
            msg="needs_refit must be a pure query",
        )
        # Before a fit, a proposal is cash-only and creates no hidden state.
        self.assertEqual(
            model.propose_buys(make_context(model, bars["end"].iloc[-1],
                                            self.market)),
            {},
        )
        self.assertIsNone(model.state)
        self.assertIsNone(model.state_date)
        self.assertIsNone(model.probabilities)
        self.assertIsNone(model._fit)

        first = make_context(model, bars["end"].iloc[model.min_train_bars],
                             self.market)
        self.assertTrue(model.needs_refit(first))
        self.assertIsNone(model.refit(None, first))
        self.assertIsNotNone(model._fit)
        self.assertIn(model.state, range(model.n_states))
        self.assertEqual(model.state_date, bars["end"].iloc[model.min_train_bars])
        self.assertFalse(
            model.needs_refit(
                make_context(model, bars["end"].iloc[model.min_train_bars + 1],
                             self.market)
            )
        )
        due = model.min_train_bars + model.refit_every_bars
        self.assertTrue(
            model.needs_refit(make_context(model, bars["end"].iloc[due],
                                           self.market))
        )
        fitted = model._fit
        for _ in range(2):
            model.needs_refit(make_context(model, bars["end"].iloc[due],
                                           self.market))
        self.assertIs(model._fit, fitted)
        self.assertEqual(model._fit_bar_count, model.min_train_bars + 1)

    def test_refit_settings_snapshot_is_detached(self):
        model = make_model(self.market)
        first = model.refit_settings()
        second = model.refit_settings()
        self.assertEqual(first, second)
        self.assertIsNot(first, second)
        self.assertIsNot(first["risk_on_states"], second["risk_on_states"])
        self.assertEqual(first["model"], "GaussianHMM on volume-bar log returns")

        first["risk_on_states"].append(99)
        first["median_window"] = -1
        first["n_states"] = 99
        fresh = model.refit_settings()
        self.assertEqual(fresh["risk_on_states"], list(model.risk_on_states))
        self.assertEqual(fresh["median_window"], model.median_window)
        self.assertEqual(fresh["n_states"], model.n_states)

    def test_public_properties_are_detached_and_consistent(self):
        model = make_model(self.market)
        bars = model.bars
        self.assertEqual(len(bars), len(model._bars))
        self.assertIsNone(model.state)
        self.assertIsNone(model.state_date)
        self.assertIsNone(model.probabilities)

        signal = bars["end"].iloc[50] + pd.Timedelta(days=1)
        model.refit(None, make_context(model, signal, self.market))
        # state_date is the last completed bar, not the signal timestamp.
        self.assertEqual(model.state_date, bars["end"].iloc[50])
        self.assertLessEqual(model.state_date, pd.Timestamp(signal))

        probabilities = model.probabilities
        self.assertEqual(probabilities.shape, (model.n_states,))
        self.assertTrue(np.isfinite(probabilities).all())
        self.assertAlmostEqual(float(probabilities.sum()), 1.0, places=12)
        self.assertEqual(int(np.argmax(probabilities)), model.state)
        self.assertIsNot(probabilities, model.probabilities)
        probabilities[0] = -5.0
        self.assertGreaterEqual(model.probabilities[0], 0.0)

        mutated = model.bars
        mutated.iloc[0, mutated.columns.get_loc("end")] = pd.Timestamp("1900-01-01")
        self.assertNotEqual(model.bars["end"].iloc[0], pd.Timestamp("1900-01-01"))

        later = bars["end"].iloc[55]
        model.propose_buys(make_context(model, later, self.market))
        self.assertEqual(model.state_date, later)
        self.assertLessEqual(model.state_date, later)

    def test_constructor_validates_inputs_and_copies_them(self):
        _, benchmark, volume = (
            self.dates,
            self.market["benchmark"],
            self.market["volume"],
        )
        invalid = [
            dict(strategy_id=""),
            dict(strategy_id=123),
        ]
        for overrides in invalid:
            with self.subTest(overrides=overrides):
                with self.assertRaises(ValueError):
                    HMMVolumeStrategy(benchmark, volume, **overrides)

        tz_benchmark = benchmark.copy()
        tz_benchmark.index = tz_benchmark.index.tz_localize("UTC")
        nan_benchmark = benchmark.copy()
        nan_benchmark.iloc[10] = np.nan
        zero_benchmark = benchmark.copy()
        zero_benchmark.iloc[10] = 0.0
        inf_volume = volume.copy()
        inf_volume.iloc[10] = np.inf
        negative_volume = volume.copy()
        negative_volume.iloc[10] = -1.0
        disjoint = volume.copy()
        disjoint.index = disjoint.index + pd.Timedelta(days=4_000)
        cases = [
            ("tz-aware benchmark", tz_benchmark, volume),
            ("nan benchmark", nan_benchmark, volume),
            ("zero benchmark", zero_benchmark, volume),
            ("inf volume", benchmark, inf_volume),
            ("negative volume", benchmark, negative_volume),
            ("disjoint calendars", benchmark, disjoint),
        ]
        for name, bench, vol in cases:
            with self.subTest(case=name):
                with self.assertRaises(ValueError):
                    HMMVolumeStrategy(bench, vol, **FAST)

        # The constructor must not alias the caller's series.
        bench = benchmark.copy()
        vol = volume.copy()
        model = HMMVolumeStrategy(bench, vol, **FAST)
        self.assertIsNot(model.benchmark, bench)
        self.assertIsNot(model.volume, vol)
        before = model.bars
        bench.iloc[:] = 1.0
        vol.iloc[:] = 0.0
        self.assertTrue(np.array_equal(model.benchmark.to_numpy(),
                                       benchmark.to_numpy()))
        self.assertTrue(model.bars.equals(before))


class DeepcopyIsolationChecks(unittest.TestCase):
    """Shared immutable inputs, private model state, and repeated engines."""

    @classmethod
    def setUpClass(cls):
        cls.market = synthetic_market(seed=24)
        cls.dates = cls.market["dates"]

    def test_deepcopy_isolates_model_state_and_fit_internals(self):
        model = make_model(self.market)
        bars = model.bars
        model.refit(None, make_context(model, bars["end"].iloc[50], self.market))
        clone = deepcopy(model)

        self.assertIs(clone._bars, model._bars)
        self.assertIs(clone.benchmark, model.benchmark)
        self.assertIs(clone.volume, model.volume)
        self.assertIsNot(clone._fit, model._fit)
        self.assertIsNot(clone._fit.model, model._fit.model)
        self.assertIsNot(clone._probabilities, model._probabilities)
        self.assertIsNot(clone._bar_returns, model._bar_returns)

        mean_before = float(model._fit.model.means_[0, 0])
        return_before = float(model._bar_returns[0])
        probability_before = float(model.probabilities[0])
        clone._fit.model.means_[0, 0] += 1.0
        clone._bar_returns[0] += 5.0
        clone._probabilities[0] = -1.0
        self.assertEqual(float(model._fit.model.means_[0, 0]), mean_before)
        self.assertEqual(float(model._bar_returns[0]), return_before)
        self.assertEqual(float(model.probabilities[0]), probability_before)

        fitted = model._fit
        state_before = (model.state, model.state_date, model._fit_bar_count,
                        model._state_bar_count)
        bars_before = model._bars.copy(deep=True)
        benchmark_before = model.benchmark.copy(deep=True)
        volume_before = model.volume.copy(deep=True)
        clone.refit(None, make_context(clone, bars["end"].iloc[70], self.market))
        clone.propose_buys(make_context(clone, bars["end"].iloc[75], self.market))
        self.assertIs(model._fit, fitted)
        self.assertEqual(
            (model.state, model.state_date, model._fit_bar_count,
             model._state_bar_count),
            state_before,
        )
        self.assertTrue(model._bars.equals(bars_before))
        self.assertTrue(model.benchmark.equals(benchmark_before))
        self.assertTrue(model.volume.equals(volume_before))

        # A clone of an unfitted model can be fitted without touching it.
        pristine = make_model(self.market)
        twin = deepcopy(pristine)
        twin.refit(None, make_context(twin, bars["end"].iloc[40], self.market))
        self.assertIsNotNone(twin._fit)
        self.assertIsNone(pristine._fit)
        self.assertIsNone(pristine.state)
        self.assertIsNone(pristine.state_date)

    def test_engine_run_does_not_mutate_shared_inputs(self):
        model = make_model(self.market)
        bars_before = model.bars
        benchmark_before = model.benchmark.copy(deep=True)
        volume_before = model.volume.copy(deep=True)
        result = make_engine(self.market, model, fee_bps=0).run(
            self.dates[200], self.dates[280]
        )
        self.assertFalse(result.refits.empty)
        self.assertTrue(model.bars.equals(bars_before))
        self.assertTrue(model.benchmark.equals(benchmark_before))
        self.assertTrue(model.volume.equals(volume_before))
        self.assertIsNone(model._fit)

    def test_engines_created_before_either_run_are_independent(self):
        model = make_model(self.market)
        first_engine = make_engine(self.market, model, fee_bps=5)
        second_engine = make_engine(self.market, model, fee_bps=5)
        start, end = self.dates[200], self.dates[300]
        first = first_engine.run(start, end)
        second = second_engine.run(start, end)
        self.assertTrue(first.fills.reset_index(drop=True).equals(
            second.fills.reset_index(drop=True)))
        np.testing.assert_allclose(first.history.nav, second.history.nav)
        self.assertIsNot(first_engine.strategy, second_engine.strategy)
        self.assertIsNot(first_engine.strategy._fit, second_engine.strategy._fit)
        self.assertIsNone(model._fit, msg="the caller's model must stay pristine")


class BacktesterApiChecks(unittest.TestCase):
    """market_data versus frames, partial begins, and degenerate universes."""

    @classmethod
    def setUpClass(cls):
        cls.market = synthetic_market(seed=25)
        cls.dates = cls.market["dates"]

    def _market_data(self):
        fundamentals = pd.DataFrame(
            columns=[
                "symbol", "fiscal_date", "filing_date", "period", "revenue",
                "net_income", "operating_income", "total_assets", "total_debt",
                "cash_and_equivalents", "free_cash_flow", "operating_cash_flow",
            ]
        )
        return MarketData.from_frames(
            self.market["prices"],
            fundamentals,
            self.market["universe"],
            self.market["benchmark_frame"],
            self.market["membership"],
        )

    def test_market_data_and_explicit_frames_run_identically(self):
        data = self._market_data()
        start, end = self.dates[120], self.dates[-1]
        explicit = make_engine(
            self.market,
            make_model(self.market, top_n=None),
            fee_bps=5,
        ).run(start, end)
        via_market_data = Backtester(
            market_data=data,
            strategy=make_model(self.market, top_n=None),
            initial_cash=100_000,
            fee_bps=5,
        ).run(start, end)

        self.assertFalse(explicit.fills.empty)
        pd.testing.assert_frame_equal(explicit.fills, via_market_data.fills,
                                      check_dtype=False)
        pd.testing.assert_frame_equal(explicit.orders, via_market_data.orders,
                                      check_dtype=False)
        np.testing.assert_allclose(explicit.history.nav,
                                   via_market_data.history.nav)
        self.assertEqual(list(explicit.refits.signal_date),
                         list(via_market_data.refits.signal_date))
        self.assertEqual(list(explicit.refits.model_version),
                         list(via_market_data.refits.model_version))

        # Passing both sources, or none, is rejected before any run.
        with self.assertRaises(ValueError):
            Backtester(self.market["prices"], self.market["benchmark_frame"],
                       self.market["membership"], make_model(self.market),
                       market_data=data)
        with self.assertRaises(ValueError):
            Backtester(strategy=make_model(self.market))

    def test_failed_begin_leaves_the_engine_reusable(self):
        engine = make_engine(self.market, make_model(self.market), fee_bps=0)
        with self.assertRaises(ValueError):
            engine.begin(self.dates[0] - pd.Timedelta(days=5), self.dates[100],
                         allow_partial=True)
        self.assertFalse(engine._started)
        self.assertEqual(engine.phase, Phase.READY)
        engine.begin(self.dates[100], self.dates[110])
        self.assertEqual(engine.next_session, self.dates[100])

        warned = make_engine(self.market, make_model(self.market), fee_bps=0)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with self.assertRaises(UserWarning):
                warned.begin(self.dates[100],
                             self.dates[-1] + pd.Timedelta(days=30),
                             allow_partial=True)
        self.assertFalse(warned._started)
        warned.begin(self.dates[100], self.dates[120])
        self.assertEqual(warned.next_session, self.dates[100])

    def test_allow_partial_clips_the_run_without_changing_accounting(self):
        engine = make_engine(self.market, make_model(self.market), fee_bps=0)
        result = engine.run(self.dates[400],
                            self.dates[-1] + pd.Timedelta(days=30),
                            allow_partial=True)
        self.assertTrue(bool(result.coverage.range_clipped))
        self.assertEqual(result.coverage.effective_end, self.dates[-1])
        self.assertEqual(result.coverage.requested_end,
                         self.dates[-1] + pd.Timedelta(days=30))
        self.assertTrue((result.history.cash >= -1e-9).all())
        self.assertTrue(np.isfinite(result.history.nav).all())

    def test_one_session_run_refits_and_books_one_close(self):
        engine = make_engine(self.market, make_model(self.market), fee_bps=0)
        result = engine.run(self.dates[300], self.dates[300])
        self.assertIsNone(engine.next_session)
        self.assertEqual(len(result.history), 2)
        self.assertEqual(result.history.index[-1], self.dates[300])
        self.assertEqual(len(result.refits), 1)
        self.assertEqual(result.refits.model_version.tolist(), [1])
        self.assertEqual(result.phases.phase.iloc[-1], "complete")
        self.assertTrue((result.history.cash >= -1e-9).all())

    def test_empty_membership_and_one_symbol_universe(self):
        empty = self.market["membership"].iloc[0:0]
        empty_engine = Backtester(
            self.market["prices"][["date", "symbol", "adj_close"]],
            self.market["benchmark_frame"],
            empty,
            make_model(self.market),
            initial_cash=10_000,
            fee_bps=0,
        )
        empty_result = empty_engine.run(self.dates[300], self.dates[360])
        self.assertTrue(empty_result.orders.empty)
        self.assertTrue(empty_result.fills.empty)
        self.assertTrue((empty_result.history.cash == 10_000).all())
        self.assertEqual(empty_result.phases.phase.iloc[-1], "complete")

        single_prices = self.market["prices"].loc[
            self.market["prices"].symbol.eq("S0")
        ]
        single_membership = self.market["membership"].loc[
            self.market["membership"].symbol.eq("S0")
        ]
        single_market = dict(self.market)
        single_market["membership"] = single_membership
        engine = Backtester(
            single_prices[["date", "symbol", "adj_close"]],
            self.market["benchmark_frame"],
            single_membership,
            make_model(single_market),
            initial_cash=10_000,
            fee_bps=0,
        )
        result = engine.run(self.dates[300], self.dates[360])
        self.assertFalse(result.fills.empty)
        self.assertEqual(set(result.orders.symbol), {"S0"})
        buys = result.fills[result.fills.side.eq("BUY")]
        self.assertEqual(set(buys.symbol), {"S0"})
        self.assertTrue((buys.due_date - buys.date).eq(pd.Timedelta(days=90)).all())
        self.assertTrue((result.history.cash >= -1e-9).all())

        # The audit and version counter agree with the recorded refits.
        self.assertEqual(result.refits.model_version.tolist(),
                         list(range(1, len(result.refits) + 1)))
        self.assertEqual(engine.model_version, len(result.refits))
        self.assertEqual(engine.strategy.strategy_id, "hmm_volume")
        self.assertEqual(set(result.fills.strategy_id), {"hmm_volume"})


class ContextTamperingChecks(unittest.TestCase):
    """Direct-caller contexts: future rows, backward dates, and bad cash."""

    @classmethod
    def setUpClass(cls):
        cls.market = synthetic_market(seed=26)
        cls.dates = cls.market["dates"]

    def _fitted(self):
        model = make_model(self.market)
        self.signal = model.bars["end"].iloc[60]
        model.refit(None, make_context(model, self.signal, self.market))
        return model

    def _context(self, prices, volume=None, cash=100_000.0):
        return DecisionContext(
            self.signal,
            pd.Timestamp(self.signal) + pd.Timedelta(days=1),
            prices,
            self.market["membership"].copy(),
            pd.DataFrame(),
            Portfolio(cash, cash, cash, {}),
            0,
            pd.DataFrame() if volume is None else volume,
            pd.DataFrame(),
        )

    def _context_with_membership(self, prices, membership):
        return DecisionContext(
            self.signal,
            pd.Timestamp(self.signal) + pd.Timedelta(days=1),
            prices,
            membership,
            pd.DataFrame(),
            Portfolio(100_000, 100_000, 100_000, {}),
            0,
            pd.DataFrame(),
            pd.DataFrame(),
        )

    def test_future_price_and_volume_rows_are_rejected(self):
        model = self._fitted()
        view = make_context(model, self.signal, self.market).prices
        future_date = pd.Timestamp(self.signal) + pd.Timedelta(days=5)
        future = view.copy()
        future.loc[future_date] = 1.0
        with self.assertRaises(ValueError):
            model.propose_buys(self._context(future))

        volume = pd.DataFrame(
            1.0, index=[self.signal, future_date], columns=list(view.columns)
        )
        with self.assertRaises(ValueError):
            model.propose_buys(self._context(view, volume))

        # The guard runs before the "no fit yet" short circuit.
        fresh = make_model(self.market)
        with self.assertRaises(ValueError):
            fresh.propose_buys(self._context(future))

    def test_backward_signal_dates_are_rejected(self):
        model = self._fitted()
        bars = model.bars
        with self.assertRaises(ValueError):
            model.propose_buys(
                make_context(model, bars["end"].iloc[40], self.market)
            )
        later = bars["end"].iloc[65]
        model.propose_buys(make_context(model, later, self.market))
        with self.assertRaises(ValueError):
            model.propose_buys(
                make_context(model, bars["end"].iloc[62], self.market)
            )

    def test_signal_after_the_supplied_series_is_rejected(self):
        """All three hooks must refuse a signal past the strategy's history."""
        model = HMMVolumeStrategy(
            self.market["benchmark"].iloc[:300],
            self.market["volume"].iloc[:300],
            **FAST,
        )
        late = self.dates[400]
        with self.assertRaises(ValueError):
            model.refit(None, make_context(model, late, self.market))
        with self.assertRaises(ValueError):
            model.needs_refit(make_context(model, late, self.market))
        with self.assertRaises(ValueError):
            model.propose_buys(make_context(model, late, self.market))
        self.assertIsNone(model._fit)
        self.assertIsNone(model.state)

    def test_nonfinite_cash_never_produces_buys(self):
        model = self._fitted()
        view = make_context(model, self.signal, self.market)
        for cash in (float("nan"), float("inf"), float("-inf"), -1.0, 0.0):
            with self.subTest(cash=cash):
                self.assertEqual(
                    model.propose_buys(self._context(view.prices, cash=cash)), {}
                )

    def test_membership_edits_cannot_add_names_outside_prices(self):
        model = self._fitted()
        view = make_context(model, self.signal, self.market).prices
        extra_membership = self.market["membership"].copy()
        extra_membership["symbol"] = "ZZZ"
        self.assertEqual(
            model.propose_buys(self._context_with_membership(view, extra_membership)),
            {},
        )

        mixed = self.market["membership"].copy()
        mixed = pd.concat(
            [mixed, mixed.iloc[:1].assign(symbol="ZZZ")], ignore_index=True
        )
        wider_prices = view.copy()
        wider_prices["OUT"] = 10.0
        budgets = model.propose_buys(
            self._context_with_membership(wider_prices, mixed)
        )
        self.assertTrue(set(budgets) <= set(mixed.symbol) & set(wider_prices.columns))
        self.assertNotIn("ZZZ", budgets)
        self.assertNotIn("OUT", budgets)

    def test_strategy_does_not_mutate_the_context(self):
        model = self._fitted()
        context = make_context(model, model.bars["end"].iloc[70], self.market)
        prices = context.prices.copy(deep=True)
        membership = context.membership.copy(deep=True)
        volume = context.volume.copy(deep=True)
        holdings = dict(context.portfolio.holdings)
        model.propose_buys(context)
        self.assertTrue(context.prices.equals(prices))
        self.assertTrue(context.membership.equals(membership))
        self.assertTrue(context.volume.equals(volume))
        self.assertEqual(dict(context.portfolio.holdings), holdings)


class ShadowPortfolioChecks(unittest.TestCase):
    """Two HMM shadows through the coordinated clock and merged account."""

    @classmethod
    def setUpClass(cls):
        cls.market = synthetic_market(seed=27)
        cls.dates = cls.market["dates"]

    def test_hmm_shadows_share_one_clock_without_state_bleed(self):
        first = make_model(self.market, strategy_id="hmm_a")
        second = make_model(self.market, strategy_id="hmm_b",
                            risk_on_states=(2,), refit_every_bars=20)
        coordinator = ShadowPortfolioBacktester(
            self.market["prices"][["date", "symbol", "adj_close"]],
            self.market["benchmark_frame"],
            self.market["membership"],
            shadows=[ShadowSpec(first), ShadowSpec(second)],
            optimizer=None,
            initial_cash=100_000,
            shadow_initial_cash=100_000,
            fee_bps=0,
        )
        result = coordinator.run(self.dates[200], self.dates[-1],
                                 shadow_start=self.dates[150])
        self.assertEqual(coordinator.phase, Phase.COMPLETE)
        self.assertIsNone(first._fit, msg="shadow setup must copy the caller model")
        self.assertIsNone(second._fit)

        self.assertEqual(set(result.shadows), {"hmm_a", "hmm_b"})
        for name, shadow in result.shadows.items():
            self.assertTrue(np.isfinite(shadow.history.nav).all())
            self.assertTrue((shadow.history.cash >= -1e-9).all())
            self.assertAlmostEqual(
                float(shadow.history.cash.iloc[0]), 100_000.0, places=6
            )
            if not shadow.refits.empty:
                self.assertTrue(shadow.refits.status.eq("success").all())
            if not shadow.fills.empty:
                self.assertEqual(set(shadow.fills.strategy_id), {name})

        self.assertFalse(result.allocations.empty)
        self.assertEqual(set(result.allocations.component_id), {"hmm_a", "hmm_b"})
        weights = result.allocations.groupby("signal_date").weight.sum()
        np.testing.assert_allclose(weights.to_numpy(), 1.0, atol=1e-12)
        for name, shadow in result.shadows.items():
            versions = result.allocations.loc[
                result.allocations.component_id.eq(name), "model_version"
            ]
            self.assertTrue((versions >= 0).all())
            self.assertTrue((versions <= len(shadow.refits)).all())

        self.assertFalse(result.portfolio.fills.empty)
        self.assertEqual(set(result.portfolio.fills.strategy_id), {"portfolio"})
        self.assertTrue(
            result.portfolio.fills.signal_date.lt(result.portfolio.fills.date).all()
        )
        self.assertTrue((result.portfolio.history.cash >= -1e-9).all())
        self.assertTrue(np.isfinite(result.portfolio.history.nav).all())
        if not result.contributions.empty:
            self.assertTrue(
                set(result.contributions.component_id) <= {"hmm_a", "hmm_b"}
            )
            self.assertTrue((result.contributions.budget > 0).all())


class DefectChecks(unittest.TestCase):
    """Regression evidence for round-2 defects; these fail on the submission."""

    @classmethod
    def setUpClass(cls):
        cls.market = synthetic_market(seed=28)
        cls.dates = cls.market["dates"]

    def test_refit_cannot_rewind_below_the_last_scored_bar(self):
        """A refit may not rewind below a bar already scored by propose_buys.

        ``HMMVolumeStrategy.refit`` only compares the new signal with
        ``_fit_date`` (strategies/hmm_volume.py:257).  ``propose_buys`` advances
        ``_state_bar_end`` beyond ``_fit_date`` (strategies/hmm_volume.py:285-295), so a
        refit at a signal between the last fit and the last scored bar is
        accepted and replaces a newer model with an older one, rewinding
        ``state_date``.
        """
        model = make_model(self.market)
        bars = model.bars
        model.refit(None, make_context(model, bars["end"].iloc[35], self.market))
        scored = bars["end"].iloc[45]
        model.propose_buys(make_context(model, scored, self.market))
        fitted = model._fit
        state = (model.state, model.state_date, model._fit_bar_count,
                 model._state_bar_count)
        self.assertEqual(model.state_date, scored)
        try:
            model.refit(None, make_context(model, bars["end"].iloc[40],
                                           self.market))
        except ValueError:
            pass
        else:
            self.fail(
                "refit accepted a signal earlier than the last scored bar "
                f"({bars['end'].iloc[40].date()} < {scored.date()}); "
                "strategies/hmm_volume.py:257 compares only _fit_date, not "
                "_state_bar_end. The state was rewound from "
                f"{state[1].date()} to {model.state_date.date()} and the "
                "model was replaced."
            )
        self.assertIs(model._fit, fitted)
        self.assertEqual(
            (model.state, model.state_date, model._fit_bar_count,
             model._state_bar_count),
            state,
        )

    def test_nat_signal_date_is_rejected(self):
        """A missing signal date must be rejected, not treated as "latest".

        ``_check_signal`` (strategies/hmm_volume.py:235-242) accepts ``NaT``
        because ``NaT > timestamp`` is ``False``.  Every later guard is also a
        ``<`` comparison, so ``propose_buys`` scores the latest completed bar
        and buys at the last price row in the context with no signal date.
        """
        model = make_model(self.market)
        signal = model.bars["end"].iloc[60]
        model.refit(None, make_context(model, signal, self.market))
        view = make_context(model, signal, self.market)
        for bad in (pd.NaT, np.datetime64("NaT"), float("nan")):
            with self.subTest(signal=bad):
                context = DecisionContext(
                    bad,
                    pd.Timestamp(signal) + pd.Timedelta(days=1),
                    view.prices,
                    self.market["membership"].copy(),
                    pd.DataFrame(),
                    Portfolio(100_000, 100_000, 100_000, {}),
                    0,
                    pd.DataFrame(),
                    pd.DataFrame(),
                )
                try:
                    budgets = model.propose_buys(context)
                except ValueError:
                    continue
                self.fail(
                    f"propose_buys accepted signal_date={bad!r} and returned "
                    f"{len(budgets)} purchases ({sorted(budgets)}) from the "
                    "last context price row; strategies/hmm_volume.py:235-242 "
                    "must reject a non-timestamp signal."
                )

    def test_engine_does_not_mutate_a_prefitted_caller_strategy(self):
        """A run must consume a copy, leaving the caller's model pristine.

        ``Backtester`` intentionally uses the supplied strategy object and may
        advance its state between refits; the documented usage is one fresh
        strategy per run.  This test passes a ``deepcopy`` and verifies the
        caller's fitted state is unchanged.
        """
        model = make_model(self.market, refit_every_bars=10_000)
        bars = model.bars
        model.refit(None, make_context(model, bars["end"].iloc[35], self.market))
        fitted = model._fit
        before = (model.state, model.state_date, model._fit_bar_count,
                  model._state_bar_count, model._fit_date)
        start = bars["end"].iloc[40]
        engine = make_engine(self.market, deepcopy(model), fee_bps=0)
        result = engine.run(start, self.dates[-1])
        self.assertTrue(result.refits.empty, msg="fixture must not refit")
        self.assertIs(model._fit, fitted)
        self.assertEqual(
            (model.state, model.state_date, model._fit_bar_count,
             model._state_bar_count, model._fit_date),
            before,
            msg=(
                "Backtester mutated the caller's pre-fitted strategy: "
                f"_state_bar_count {before[3]} -> {model._state_bar_count}, "
                f"state_date {before[1]} -> {model.state_date}. "
                "backtesting/engine.py:133 keeps the caller object and "
                "engine.py:279 advances it through propose_buys. A second "
                "engine with this object fails with 'Signal date moved "
                "backwards' instead of running."
            ),
        )


if __name__ == "__main__":
    unittest.main()
