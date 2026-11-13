"""Baseline tests for the volume-bar HMM market-timing strategy."""

import unittest

import numpy as np
import pandas as pd

from backtesting import Backtester, DecisionContext, Portfolio
from strategies.hmm_volume import HMMVolumeStrategy


def synthetic_market(periods=800, symbols=6, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-01", periods=periods)
    benchmark = 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, periods)))
    volume = pd.Series(rng.lognormal(15, 0.3, periods), index=dates)
    price_rows = []
    for symbol in [f"S{i}" for i in range(symbols)]:
        prices = benchmark * np.exp(np.cumsum(rng.normal(0, 0.005, periods)))
        price_rows.append(pd.DataFrame({"date": dates, "symbol": symbol, "adj_close": prices}))
    prices = pd.concat(price_rows, ignore_index=True)
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
    return (
        dates,
        pd.Series(benchmark, index=dates),
        volume,
        prices,
        benchmark_frame,
        membership,
    )


def context(strategy, signal, prices, membership, *, nav=100_000.0, cash=None,
            holdings=None, volume=None):
    signal = pd.Timestamp(signal)
    execution = signal + pd.Timedelta(days=1)
    cash = nav if cash is None else cash
    if "date" in prices.columns:
        price_view = (
            prices.loc[prices["date"] <= signal]
            .pivot(index="date", columns="symbol", values="adj_close")
        )
    else:
        price_view = prices.loc[:signal].copy()
    return DecisionContext(
        signal,
        execution,
        price_view,
        membership.copy(),
        pd.DataFrame(),
        Portfolio(cash, cash, nav, dict(holdings or {})),
        0,
        pd.DataFrame() if volume is None else volume.loc[:signal].copy(),
        pd.DataFrame(),
    )


def strategy(seed=0, **overrides):
    dates, benchmark, volume, prices, benchmark_frame, membership = synthetic_market(seed=seed)
    options = dict(
        median_window=20,
        bucket_multiple=5.0,
        n_states=3,
        min_train_bars=30,
        refit_every_bars=10,
        n_init=1,
        n_iter=50,
        top_n=4,
        risk_on_states=(1, 2),
    )
    options.update(overrides)
    model = HMMVolumeStrategy(benchmark, volume, **options)
    return model, dates, prices, membership


class HMMVolumeChecks(unittest.TestCase):
    def test_constructor_validation(self):
        _, benchmark, volume, _, _, _ = synthetic_market(periods=400)
        invalid = [
            dict(median_window=1),
            dict(bucket_multiple=0),
            dict(n_states=1),
            dict(min_train_bars=3),
            dict(refit_every_bars=0),
            dict(n_init=0),
            dict(n_iter=0),
            dict(risk_on_states=()),
            dict(risk_on_states=(3,)),
            dict(risk_on_states=(1, 1)),
            dict(top_n=0),
            dict(market_cap=pd.DataFrame({'symbol': ['A']})),
            dict(min_order_notional=-1),
        ]
        for options in invalid:
            with self.subTest(options=options):
                with self.assertRaises(ValueError):
                    HMMVolumeStrategy(benchmark, volume, **options)
        with self.assertRaises(ValueError):
            HMMVolumeStrategy(benchmark.iloc[:10], volume, median_window=20)

    def test_dataframe_inputs_are_normalized(self):
        dates, benchmark, volume, _, _, _ = synthetic_market(periods=400)
        model = HMMVolumeStrategy(
            pd.DataFrame({"date": dates, "adj_close": benchmark.to_numpy()}),
            pd.DataFrame({"date": dates, "volume": volume.to_numpy()}),
            median_window=20,
            min_train_bars=30,
            n_init=1,
            n_iter=50,
        )
        self.assertEqual(len(model.benchmark), len(dates))

    def test_needs_refit_waits_for_minimum_and_cadence(self):
        model, dates, prices, membership = strategy()
        bars = model._bars
        first = model.min_train_bars
        self.assertFalse(model.needs_refit(context(model, bars["end"].iloc[first - 1], prices, membership)))
        first_context = context(model, bars["end"].iloc[first], prices, membership)
        self.assertTrue(model.needs_refit(first_context))
        model.refit(None, first_context)
        self.assertEqual(model._fit_bar_count, first + 1)
        self.assertFalse(model.needs_refit(context(model, bars["end"].iloc[first + 1], prices, membership)))
        due = first + 1 + model.refit_every_bars
        self.assertTrue(model.needs_refit(context(model, bars["end"].iloc[due], prices, membership)))

    def test_refit_scores_the_current_bar_and_stores_state(self):
        model, dates, prices, membership = strategy()
        bars = model._bars
        signal = bars["end"].iloc[model.min_train_bars]
        model.refit(None, context(model, signal, prices, membership))
        self.assertIsNotNone(model._fit)
        self.assertIn(model.state, range(model.n_states))
        self.assertEqual(model.state_date, signal)
        self.assertEqual(len(model._bar_returns), model.min_train_bars + 1)

    def test_risk_off_returns_no_buys(self):
        model, dates, prices, membership = strategy(risk_on_states=(2,))
        bars = model._bars
        model._fit = object()
        model._state = 0
        model._state_bar_count = len(bars)
        ctx = context(model, bars["end"].iloc[-1], prices, membership)
        self.assertEqual(model.propose_buys(ctx), {})

    def test_risk_on_buys_equal_weight_basket_within_cash(self):
        model, dates, prices, membership = strategy(top_n=4)
        bars = model._bars
        model._fit = object()
        model._state = 2
        model._state_bar_count = len(bars)
        ctx = context(model, bars["end"].iloc[-1], prices, membership, nav=1000.0)
        budgets = model.propose_buys(ctx)
        self.assertEqual(len(budgets), 4)
        self.assertLessEqual(sum(budgets.values()), 1000.0 + 1e-8)
        self.assertTrue(all(symbol in set(membership.symbol) for symbol in budgets))
        self.assertTrue(all(value > 0 for value in budgets.values()))

    def test_no_cash_or_empty_universe_returns_no_buys(self):
        model, dates, prices, membership = strategy()
        bars = model._bars
        model._fit = object()
        model._state = 2
        model._state_bar_count = len(bars)
        signal = bars["end"].iloc[-1]
        self.assertEqual(model.propose_buys(context(model, signal, prices, membership, cash=0)), {})
        empty = membership.iloc[0:0]
        self.assertEqual(model.propose_buys(context(model, signal, prices, empty)), {})

    def test_market_cap_ranking_and_weights(self):
        dates, benchmark, volume, prices, benchmark_frame, membership = synthetic_market(seed=3)
        caps = pd.DataFrame({
            "symbol": [f"S{i}" for i in range(6)],
            "quarter_start": pd.Timestamp("2015-01-01"),
            "selection_date": pd.Timestamp("2014-12-31"),
            "market_cap_proxy": [10.0, 20.0, 30.0, 40.0, 500.0, 1000.0],
        })
        model = HMMVolumeStrategy(
            benchmark, volume, market_cap=caps, median_window=20,
            min_train_bars=30, refit_every_bars=10, n_init=1, n_iter=50,
            top_n=2,
        )
        bars = model._bars
        model._fit = object()
        model._state = 2
        model._state_bar_count = len(bars)
        signal = bars["end"].iloc[-1]
        budgets = model.propose_buys(context(model, signal, prices, membership, nav=1000.0))
        self.assertEqual(set(budgets), {"S5", "S4"})
        self.assertAlmostEqual(budgets["S5"], 1000.0 * 1000.0 / 1500.0, delta=1e-3)
        self.assertAlmostEqual(budgets["S4"], 1000.0 * 500.0 / 1500.0, delta=1e-3)

    def test_future_data_does_not_change_past_state_or_proposals(self):
        dates, benchmark, volume, prices, benchmark_frame, membership = synthetic_market(seed=3)
        model = HMMVolumeStrategy(
            benchmark, volume, median_window=20, min_train_bars=30,
            refit_every_bars=10, n_init=1, n_iter=50, top_n=4,
        )
        bars = model._bars
        signal = bars["end"].iloc[60]
        ctx = context(model, signal, prices, membership)
        model.refit(None, ctx)
        first_state = model.state
        first_budgets = model.propose_buys(ctx)

        changed_benchmark = benchmark.copy()
        changed_volume = volume.copy()
        changed_benchmark.loc[changed_benchmark.index > signal] *= 2
        changed_volume.loc[changed_volume.index > signal] *= 3
        changed = HMMVolumeStrategy(
            changed_benchmark, changed_volume, median_window=20, min_train_bars=30,
            refit_every_bars=10, n_init=1, n_iter=50, top_n=4,
        )
        changed.refit(None, ctx)
        self.assertEqual(changed.state, first_state)
        self.assertEqual(changed.propose_buys(ctx), first_budgets)

    def test_backtester_integration_holds_to_maturity(self):
        dates, benchmark, volume, prices, benchmark_frame, membership = synthetic_market(seed=5)
        model = HMMVolumeStrategy(
            benchmark, volume, median_window=20, min_train_bars=30,
            refit_every_bars=10, n_init=1, n_iter=50, top_n=3,
        )
        engine = Backtester(
            prices, benchmark_frame, membership, model,
            initial_cash=100_000, fee_bps=0,
        )
        result = engine.run(dates[200], dates[-1])
        self.assertFalse(result.refits.empty)
        self.assertTrue(result.refits.status.eq("success").all())
        if not result.fills.empty:
            sells = result.fills.loc[result.fills.side.eq("SELL")]
            self.assertTrue((sells.date >= sells.due_date).all())
        self.assertTrue((result.history.cash >= -1e-8).all())
        self.assertTrue(np.isfinite(result.history.nav).all())

    def test_refit_settings_are_recorded(self):
        model, _, _, _ = strategy()
        settings = model.refit_settings()
        self.assertEqual(settings["n_states"], 3)
        self.assertEqual(settings["median_window"], 20)
        self.assertIn("risk_on_states", settings)


if __name__ == "__main__":
    unittest.main()
