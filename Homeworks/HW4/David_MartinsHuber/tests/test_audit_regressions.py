"""Regression expectations derived from the first independent audit findings."""

from copy import deepcopy
from types import SimpleNamespace
import unittest
import warnings

import numpy as np
import pandas as pd

from backtesting import (Backtester, BacktestAnalytics, OMS, Strategy, Phase,
                         DecisionContext, Portfolio, PortfolioStrategy, EqualWeightOptimizer)
from tests.test_backtesting import inputs


class StayInCash(Strategy):
    def propose_buys(self, context):
        return {}


class AuditRegressionChecks(unittest.TestCase):
    def test_missing_requested_coverage_fails_before_starting(self):
        dates, p, b, m = inputs(8)
        engine = Backtester(p, b, m, StayInCash("cash"))
        with self.assertRaisesRegex(ValueError, "coverage"):
            engine.run(dates[1], "2026-09-30")
        self.assertFalse(engine._started)
        self.assertFalse(engine.oms.fills)

    def test_explicit_partial_run_warns_and_discloses_actual_dates(self):
        dates, p, b, m = inputs(8)
        engine = Backtester(p, b, m, StayInCash("cash"))
        with self.assertWarnsRegex(UserWarning, "Partial backtest"):
            result = engine.run(dates[1], "2026-09-30", allow_partial=True)
        self.assertEqual(result.coverage.requested_end, pd.Timestamp("2026-09-30"))
        self.assertEqual(result.coverage.effective_end, dates[-1])
        self.assertTrue(result.coverage.range_clipped)

    def test_weekend_end_inside_supplied_calendar_is_supported(self):
        dates, p, b, m = inputs(8)  # Source includes Monday January 13.
        result = Backtester(p, b, m, StayInCash("cash")).run(dates[1], "2020-01-12")
        self.assertEqual(result.coverage.effective_end, pd.Timestamp("2020-01-10"))
        self.assertFalse(result.coverage.range_clipped)

    def test_incremental_methods_require_begin(self):
        _, p, b, m = inputs(8)
        for method in ["prepare", "advance", "finish"]:
            with self.subTest(method=method):
                engine = Backtester(p, b, m, StayInCash("cash"))
                with self.assertRaisesRegex(ValueError, "begin"):
                    getattr(engine, method)()

    def test_unsupported_fee_configuration_is_rejected(self):
        for fees in [10000.1, 20000, np.inf, -1]:
            with self.subTest(fees=fees), self.assertRaises(ValueError):
                OMS(100, fee_bps=fees)

    def test_nonrepresentable_buy_and_sale_reject_atomically(self):
        oms = OMS(100, fee_bps=5)
        order = oms.submit_buy("s", "A", 100, "2020-01-02", "2020-01-03")
        before = deepcopy((oms.cash, oms.orders, oms.lots, oms.fills, oms.available_cash))
        with self.assertRaisesRegex(ValueError, "finite"):
            oms.fill(order, symbol="A", quote_date="2020-01-03",
                     price=np.nextafter(0, 1), session="2020-01-03")
        self.assertEqual(before, (oms.cash, oms.orders, oms.lots, oms.fills, oms.available_cash))
        oms.fill(order, symbol="A", quote_date="2020-01-03", price=.01, session="2020-01-03")
        sale = oms.submit_maturity_sale(order, "2020-04-03")
        before = deepcopy((oms.cash, oms.orders, oms.lots, oms.fills, oms.available_cash))
        with self.assertRaisesRegex(ValueError, "finite"):
            oms.fill(sale, symbol="A", quote_date="2020-04-03", price=1e308, session="2020-04-03")
        self.assertEqual(before, (oms.cash, oms.orders, oms.lots, oms.fills, oms.available_cash))

    def test_nonfinite_derived_returns_and_cagr_are_rejected(self):
        dates = pd.bdate_range("2020-01-01", periods=3)
        history = pd.DataFrame({"nav": [1e-300, 1e300, 1e300],
                                "benchmark_nav": [100., 101., 102.]}, index=dates)
        with self.assertRaisesRegex(ValueError, "Derived daily returns"):
            BacktestAnalytics(SimpleNamespace(history=history))
        # All daily returns are finite; annualizing this two-day growth overflows.
        history["nav"] = [100., 1e100, 1e200]
        analysis = BacktestAnalytics(SimpleNamespace(history=history))
        with np.errstate(over="ignore", invalid="ignore"), self.assertRaises(ValueError):
            analysis.summary()

    def test_partial_warning_as_error_does_not_start_or_corrupt_engine(self):
        dates, p, b, m = inputs(8)
        engine = Backtester(p, b, m, StayInCash("cash"))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with self.assertRaises(UserWarning):
                engine.begin(dates[1], "2026-09-30", allow_partial=True)
        self.assertFalse(engine._started)
        with self.assertRaisesRegex(ValueError, "begin"):
            _ = engine.history_so_far
        self.assertEqual(engine.run(dates[1], dates[-1]).coverage.effective_end, dates[-1])

    def test_benchmark_normalization_and_failed_finish(self):
        dates, p, b, m = inputs(8)
        b["adj_close"] = 1e308
        result = Backtester(p, b, m, StayInCash("cash"), initial_cash=1000).run(dates[1], dates[-1])
        self.assertTrue(result.history.benchmark_nav.eq(1000).all())
        b.loc[b.date.eq(dates[1]), "adj_close"] = 1e-308
        engine = Backtester(p, b, m, StayInCash("cash"))
        with self.assertRaisesRegex(ValueError, "Benchmark normalization"):
            engine.run(dates[1], dates[-1])
        self.assertEqual(engine.phase, Phase.FAILED)

    def test_full_budget_large_capital_sizes_conservatively(self):
        for capital in [100_000_000, 1_000_000_000]:
            for price in [958.2097977234747, 1.01, 10000.03]:
                with self.subTest(capital=capital, price=price):
                    oms = OMS(capital, fee_bps=5)
                    order = oms.submit_buy("s", "A", capital, "2020-01-02", "2020-01-03")
                    oms.fill(order, symbol="A", quote_date="2020-01-03", price=price, session="2020-01-03")
                    fill = oms.fills[0]
                    self.assertLessEqual(fill["notional"] + fill["fee"], capital)
                    self.assertGreater(fill["units"], 0)
                    self.assertGreaterEqual(oms.cash, 0)
                    self.assertEqual(oms.cash, capital + fill["cash_change"])

    def test_large_finite_portfolio_deployment_is_not_silently_dropped(self):
        dates = pd.bdate_range("2020-01-01", periods=2)
        strategy = PortfolioStrategy("parent", EqualWeightOptimizer())
        strategy.observe(dates[-1], pd.DataFrame({"child": [1e200, 1e200]}, index=dates),
                         {"child": {"A": 1e200}}, {"child": 0})
        context = DecisionContext(dates[-1], dates[-1] + pd.Timedelta(days=1),
                                  pd.DataFrame(), pd.DataFrame(), pd.DataFrame(),
                                  Portfolio(1e200, 1e200, 1e200, {}), 0)
        self.assertEqual(strategy.propose_buys(context), {"A": 1e200})

    def test_shadow_return_overflow_rejected_before_publication(self):
        dates = pd.bdate_range("2020-01-01", periods=2)
        strategy = PortfolioStrategy("parent", EqualWeightOptimizer())
        with self.assertRaisesRegex(ValueError, "shadow returns"):
            strategy.observe(dates[-1], pd.DataFrame({"child": [1e-200, 1e200]}, index=dates),
                             {"child": {}}, {"child": 0})
        self.assertIsNone(strategy._cutoff)

    def test_comparison_percentage_point_overflow_is_rejected(self):
        history = pd.DataFrame({"nav": [1., 1e307], "benchmark_nav": [1., 1.]},
                               index=pd.bdate_range("2020-01-01", periods=2))
        analysis = BacktestAnalytics(SimpleNamespace(history=history))
        with self.assertRaisesRegex(ValueError, "comparison arithmetic"):
            analysis.comparison()
