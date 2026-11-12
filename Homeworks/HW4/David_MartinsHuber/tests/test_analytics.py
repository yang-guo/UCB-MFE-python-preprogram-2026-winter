"""Known-return examples for portfolio metrics and benchmark regression."""

from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd

from backtesting import BacktestAnalytics


def result_from_returns(strategy, benchmark):
    values = np.vstack([np.ones(2), np.cumprod(1 + np.column_stack([strategy, benchmark]), axis=0)])
    history = pd.DataFrame(100 * values, columns=["nav", "benchmark_nav"],
                           index=pd.bdate_range("2020-01-01", periods=len(values)))
    return SimpleNamespace(history=history)


class AnalyticsChecks(unittest.TestCase):
    def test_known_alpha_beta_and_benchmark_identity(self):
        rf = 1.02 ** (1 / 252) - 1
        market = np.array([-.02, .01, .03, -.01, .015])
        strategy = rf + .0002 + 1.5 * (market - rf)
        report = BacktestAnalytics(result_from_returns(strategy, market)).summary()
        self.assertAlmostEqual(report.loc["beta", "Strategy"], 1.5)
        self.assertAlmostEqual(report.loc["annualized_alpha", "Strategy"], .0002 * 252)
        self.assertAlmostEqual(report.loc["beta", "S&P 500 total return"], 1)
        self.assertAlmostEqual(report.loc["annualized_alpha", "S&P 500 total return"], 0)

    def test_var_drawdown_sharpe_and_final_value(self):
        returns = np.array([-.1, -.05, 0, .04, .1])
        result = result_from_returns(returns, returns)
        report = BacktestAnalytics(result, annual_rf=0).summary()
        self.assertAlmostEqual(report.loc["VaR_95_daily", "Strategy"], .09)
        self.assertAlmostEqual(report.loc["VaR_99_daily", "Strategy"], .098)
        self.assertAlmostEqual(report.loc["max_drawdown", "Strategy"], -.145)
        self.assertAlmostEqual(report.loc["final_value", "Strategy"], 100 * np.prod(1 + returns))
        self.assertAlmostEqual(report.loc["Sharpe", "Strategy"], np.sqrt(252) * returns.mean() / returns.std(ddof=1))
        years = (result.history.index[-1] - result.history.index[0]).days / 365.25
        self.assertAlmostEqual(report.loc["CAGR", "Strategy"], np.prod(1 + returns) ** (1 / years) - 1)
        self.assertTrue(BacktestAnalytics(result).cumulative_returns().iloc[0].eq(0).all())

    def test_constant_benchmark_and_gain_quantiles(self):
        report = BacktestAnalytics(result_from_returns([.01, .02, .03], [0, 0, 0])).summary()
        self.assertTrue(np.isnan(report.loc["beta", "Strategy"]))
        self.assertTrue(np.isnan(report.loc["annualized_alpha", "Strategy"]))
        self.assertTrue(np.isnan(report.loc["Sharpe", "S&P 500 total return"]))
        self.assertLess(report.loc["VaR_95_daily", "Strategy"], 0)

    def test_invalid_nav_and_benchmark_comparison(self):
        result = result_from_returns([.01, .02], [0, 0])
        comparison = BacktestAnalytics(result).comparison()
        self.assertAlmostEqual(comparison.final_value_difference, 3.02)
        self.assertAlmostEqual(comparison.total_return_difference_pp, 3.02)
        self.assertTrue(comparison.beat_benchmark)
        result.history.iloc[1, 0] = np.nan
        with self.assertRaises(ValueError):
            BacktestAnalytics(result)
