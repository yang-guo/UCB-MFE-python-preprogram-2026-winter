"""Shared-clock shadow histories, capital allocation, and causal boundaries."""

import unittest

import numpy as np
import pandas as pd

from backtesting import (AllocationContext, Backtester, EqualWeightOptimizer,
                         MinimumVarianceOptimizer, Phase, Portfolio,
                         PortfolioOptimizer, PortfolioStrategy, ShadowPortfolioBacktester,
                         ShadowSpec, Strategy)
from tests.test_backtesting import inputs


class FractionStrategy(Strategy):
    def __init__(self, strategy_id, symbol="A"):
        super().__init__(strategy_id)
        self.symbol = symbol

    def propose_buys(self, context):
        assert context.prices.index.max() <= context.signal_date
        budget = context.portfolio.available_cash / 4
        return {self.symbol: budget} if budget > 1e-8 else {}


class RefittingFraction(FractionStrategy):
    def __init__(self, strategy_id, lookback):
        super().__init__(strategy_id)
        self.lookback = lookback

    def needs_refit(self, context):
        return True

    def refit(self, data, context):
        assert data.audit.label_end.max() < context.signal_date
        assert data.audit.available_date.le(data.audit.date).all()
        rows = data.audit.date >= context.signal_date - pd.Timedelta(days=self.lookback)
        self.mean = data.y.loc[rows].mean()
        self.fitting_rows = int(rows.sum())

    def refit_settings(self):
        return {"lookback_days": self.lookback, "random_seed": 0}

    def propose_buys(self, context):
        return super().propose_buys(context) if self.mean > 0 else {}


class InspectOptimizer(PortfolioOptimizer):
    def weights(self, context):
        assert context.execution_date > context.signal_date
        assert context.shadow_returns.index.max() <= context.signal_date if len(context.shadow_returns) else True
        # Mutating supplied data must not change published source histories.
        names = list(context.shadow_returns.columns)
        context.shadow_returns.iloc[:, :] = 123
        return dict.fromkeys(names, 1 / len(names))


class PortfolioChecks(unittest.TestCase):
    def test_single_shadow_matches_standalone_account(self):
        dates, p, b, m = inputs(12)
        shadow = FractionStrategy("one")
        actual = ShadowPortfolioBacktester(p, b, m, [shadow], initial_cash=100,
                                          shadow_initial_cash=100, fee_bps=10).run(dates[1], dates[-1])
        standalone = Backtester(p, b, m, FractionStrategy("one"), initial_cash=100, fee_bps=10).run(dates[1], dates[-1])
        pd.testing.assert_frame_equal(actual.portfolio.history, standalone.history)
        pd.testing.assert_frame_equal(actual.shadows["one"].history, standalone.history)
        self.assertEqual(set(actual.portfolio.fills.strategy_id), {"portfolio"})
        self.assertEqual(set(actual.shadows["one"].fills.strategy_id), {"one"})

    def test_rejected_component_buy_is_not_approved(self):
        dates, prices, benchmark, membership = inputs(8)
        missing = prices.loc[~((prices.date.eq(dates[1]))
                                & (prices.symbol.eq("A")))]
        result = ShadowPortfolioBacktester(
            missing, benchmark, membership,
            [FractionStrategy("a"), FractionStrategy("b")],
            initial_cash=100, shadow_initial_cash=100, fee_bps=0).run(
                dates[1], dates[-1])
        first = result.approved_contributions.loc[
            result.approved_contributions.execution_date.eq(dates[1])]
        self.assertFalse(first.empty)
        self.assertTrue(first.approved_budget.eq(0).all())

    def test_shared_cutoff_warmup_and_overlapping_orders(self):
        dates, p, b, m = inputs(12)
        runner = ShadowPortfolioBacktester(p, b, m, [FractionStrategy("a"), FractionStrategy("b")],
                                          InspectOptimizer(), initial_cash=100, shadow_initial_cash=100)
        result = runner.run(dates[5], dates[-1], shadow_start=dates[1])
        self.assertEqual(result.allocations.iloc[0].signal_date, dates[4])
        self.assertEqual(result.allocations.iloc[0].return_observations, 4)
        self.assertTrue(result.allocations.return_cutoff.le(result.allocations.signal_date).all())
        self.assertEqual(len(result.portfolio.orders), len(result.portfolio.history) - 1)
        budgets = result.contributions.groupby(["execution_date", "symbol"]).budget.sum()
        for order in result.portfolio.orders.itertuples():
            self.assertAlmostEqual(order.budget, budgets.loc[order.execution_date, order.symbol])
        self.assertTrue(result.portfolio.history.cash.ge(0).all())
        self.assertEqual(runner.phase, Phase.COMPLETE)

    def test_changes_after_cutoff_cannot_change_earlier_decisions(self):
        dates, p, b, m = inputs(15)
        cutoff = dates[7]
        def run(prices, benchmark):
            return ShadowPortfolioBacktester(prices, benchmark, m,
                       [FractionStrategy("a"), FractionStrategy("b")],
                       MinimumVarianceOptimizer(lookback=5, min_observations=2)).run(dates[1], dates[-1])
        original = run(p, b)
        changed_p, changed_b = p.copy(), b.copy()
        changed_p.loc[p.date.gt(cutoff), "adj_close"] *= np.arange(2, 2 + p.date.gt(cutoff).sum())
        changed_b.loc[b.date.gt(cutoff), "adj_close"] *= 2
        changed = run(changed_p, changed_b)
        pd.testing.assert_frame_equal(original.portfolio.history.loc[:cutoff], changed.portfolio.history.loc[:cutoff])
        for key in original.shadows:
            pd.testing.assert_frame_equal(original.shadows[key].history.loc[:cutoff], changed.shadows[key].history.loc[:cutoff])
        for name in ["allocations", "contributions"]:
            first, second = getattr(original, name), getattr(changed, name)
            pd.testing.assert_frame_equal(first.loc[first.signal_date.le(cutoff)].reset_index(drop=True),
                                          second.loc[second.signal_date.le(cutoff)].reset_index(drop=True))
        # A full run's prefix must also equal an actually truncated run.
        truncated = ShadowPortfolioBacktester(p.loc[p.date.le(cutoff)], b.loc[b.date.le(cutoff)], m,
                    [FractionStrategy("a"), FractionStrategy("b")],
                    MinimumVarianceOptimizer(lookback=5, min_observations=2)).run(dates[1], cutoff)
        pd.testing.assert_frame_equal(original.portfolio.history.loc[:cutoff], truncated.portfolio.history)

    def test_different_refit_windows_and_future_labels(self):
        dates, p, b, m = inputs(12)
        training_dates = pd.date_range("2019-12-20", "2020-01-12")
        panel = pd.DataFrame({"date": training_dates, "symbol": "A", "available_date": training_dates,
                              "label_end": training_dates + pd.Timedelta(days=2),
                              "target": .1, "feature": 1.0})
        cutoff = dates[5]
        def run(frame):
            specs = [ShadowSpec(RefittingFraction("short", 7), frame, ("feature",)),
                     ShadowSpec(RefittingFraction("long", 30), frame, ("feature",))]
            return ShadowPortfolioBacktester(p, b, m, specs).run(dates[1], dates[-1])
        first = run(panel)
        modified = panel.copy()
        modified.loc[modified.label_end.ge(cutoff), "target"] = -100
        second = run(modified)
        for name in first.shadows:
            fits = first.shadows[name].refits
            self.assertTrue(fits.latest_label_end.lt(fits.signal_date).all())
            self.assertTrue(first.shadows[name].fills.model_version.ge(1).all())
            a, z = first.shadows[name].fills, second.shadows[name].fills
            pd.testing.assert_frame_equal(a.loc[a.signal_date.le(cutoff)].reset_index(drop=True),
                                          z.loc[z.signal_date.le(cutoff)].reset_index(drop=True))
        a, z = first.allocations, second.allocations
        pd.testing.assert_frame_equal(a.loc[a.signal_date.le(cutoff)].reset_index(drop=True),
                                      z.loc[z.signal_date.le(cutoff)].reset_index(drop=True))
        self.assertEqual(first.shadows["short"].refits.iloc[0].settings["lookback_days"], 7)
        self.assertEqual(first.shadows["long"].refits.iloc[0].settings["lookback_days"], 30)

    def test_published_history_is_detached_and_cannot_be_rewritten(self):
        strategy = PortfolioStrategy("parent", EqualWeightOptimizer())
        dates = pd.bdate_range("2020-01-01", periods=3)
        history = pd.DataFrame({"a": [100., 101., 102.]}, index=dates)
        strategy.observe(dates[1], history.iloc[:2], {"a": {}}, {"a": 0})
        history.iloc[0, 0] = 0
        with self.assertRaises(ValueError):
            strategy.observe(dates[2], history, {"a": {}}, {"a": 0})
        self.assertEqual(strategy._navs.iloc[0, 0], 100)

    def test_minimum_variance_solution_and_bad_weights(self):
        dates = pd.bdate_range("2020-01-01", periods=4)
        returns = pd.DataFrame({"a": [-.01, -.01, .01, .01], "b": [-.02, .02, -.02, .02]}, index=dates)
        context = AllocationContext(dates[-1], dates[-1] + pd.Timedelta(days=1), Portfolio(100, 100, 100, {}), returns)
        weights = MinimumVarianceOptimizer(lookback=4, min_observations=2, shrinkage=0).weights(context)
        self.assertAlmostEqual(weights["a"], .8, places=5)
        self.assertAlmostEqual(weights["b"], .2, places=5)
        with self.assertRaises(ValueError):
            MinimumVarianceOptimizer(max_weight=.4).weights(context)
        class Bad(PortfolioOptimizer):
            def weights(self, context):
                return {"a": -1.0}
        dates, p, b, m = inputs(5)
        runner = ShadowPortfolioBacktester(p, b, m, [FractionStrategy("a")], Bad())
        with self.assertRaises(ValueError):
            runner.run(dates[1], dates[-1])
        self.assertEqual(runner.phase, Phase.FAILED)
        self.assertFalse(runner._main.oms.fills)

    def test_shadow_failure_blocks_optimizer_and_main_execution(self):
        class Broken(FractionStrategy):
            def propose_buys(self, context):
                raise RuntimeError("component failed")
        dates, p, b, m = inputs(5)
        runner = ShadowPortfolioBacktester(p, b, m, [FractionStrategy("a"), Broken("b")])
        with self.assertRaises(RuntimeError):
            runner.run(dates[1], dates[-1])
        self.assertFalse(runner._main.oms.fills)
        self.assertFalse(runner._shadows["a"].oms.fills)
        self.assertFalse(runner._main.strategy.allocations)
