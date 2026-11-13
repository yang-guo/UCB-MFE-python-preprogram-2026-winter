"""Minimum-variance, maximum-Sharpe, and maximum-diversification checks."""

from unittest import mock
import unittest

import numpy as np
import pandas as pd
from sklearn.covariance import ledoit_wolf

from backtesting import (AllocationContext, MaximumDiversificationOptimizer,
                         MaximumSharpeOptimizer, MDPOptimizer, MinimumVarianceOptimizer,
                         MSPOptimizer, Portfolio, PortfolioOptimizationError,
                         ledoit_wolf_covariance)


def context(returns):
    dates = returns.index
    return AllocationContext(dates[-1], dates[-1] + pd.Timedelta(days=1),
                             Portfolio(100, 100, 100, {}), returns)


def sample_returns(seed=0, periods=500):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=periods)
    return pd.DataFrame({
        "a": 0.001 + 0.01 * rng.normal(size=periods),
        "b": 0.002 + 0.02 * rng.normal(size=periods),
        "c": 0.0005 + 0.015 * rng.normal(size=periods),
    }, index=dates)


class PortfolioOptimizerChecks(unittest.TestCase):
    def test_ledoit_wolf_matches_sklearn(self):
        returns = sample_returns(periods=120)
        covariance, intensity = ledoit_wolf_covariance(returns)
        expected, expected_intensity = ledoit_wolf(
            returns.to_numpy(), assume_centered=False)
        np.testing.assert_allclose(covariance, expected)
        self.assertAlmostEqual(intensity, expected_intensity)
        self.assertTrue(0 <= intensity <= 1)

    def test_minimum_variance_defaults_to_ledoit_wolf(self):
        returns = sample_returns(periods=120)
        optimizer = MinimumVarianceOptimizer(lookback=120, min_observations=20)
        self.assertEqual(optimizer.covariance, "ledoit_wolf")
        self.assertIsNone(optimizer.shrinkage)
        weights = optimizer.weights(context(returns))
        self.assertAlmostEqual(sum(weights.values()), 1)
        self.assertTrue(all(value >= 0 for value in weights.values()))
        covariance, _ = ledoit_wolf_covariance(returns)
        variance = np.array(list(weights.values())) @ covariance @ np.array(
            list(weights.values()))
        self.assertGreater(variance, 0)

    def test_explicit_shrinkage_keeps_diagonal_estimator(self):
        optimizer = MinimumVarianceOptimizer(shrinkage=0.0)
        self.assertEqual(optimizer.covariance, "diagonal")
        self.assertEqual(optimizer.shrinkage, 0.0)

    def test_maximum_sharpe_beats_equal_and_grid(self):
        returns = sample_returns(periods=500)
        optimizer = MSPOptimizer(lookback=500, min_observations=20, annual_rf=0)
        weights = optimizer.weights(context(returns))
        values = np.array([weights["a"], weights["b"], weights["c"]])
        self.assertAlmostEqual(values.sum(), 1)
        self.assertTrue((values >= 0).all())
        covariance, _ = ledoit_wolf_covariance(returns)
        expected = returns.mean().to_numpy()
        sharpe = (values @ expected) / np.sqrt(values @ covariance @ values)
        grid = np.linspace(0, 1, 101)
        best = -np.inf
        for first in grid:
            for second in grid:
                third = 1 - first - second
                if third < -1e-12:
                    continue
                w = np.array([first, second, max(third, 0.0)])
                if w.sum() <= 0:
                    continue
                w = w / w.sum()
                value = (w @ expected) / np.sqrt(w @ covariance @ w)
                best = max(best, value)
        self.assertGreaterEqual(sharpe, best - 2e-3)
        equal = np.full(3, 1 / 3)
        equal_sharpe = (equal @ expected) / np.sqrt(equal @ covariance @ equal)
        self.assertGreaterEqual(sharpe, equal_sharpe - 1e-9)
        self.assertEqual(optimizer.last_result["status"], "converged")
        self.assertAlmostEqual(optimizer.last_result["sharpe"], sharpe)

    def test_maximum_sharpe_respects_cap_and_warmup(self):
        returns = sample_returns(periods=200)
        optimizer = MSPOptimizer(lookback=200, min_observations=20,
                                 annual_rf=0, max_weight=0.6)
        weights = optimizer.weights(context(returns))
        self.assertTrue(all(value <= 0.6 + 1e-9 for value in weights.values()))
        self.assertAlmostEqual(sum(weights.values()), 1)
        warm = MSPOptimizer(lookback=5, min_observations=3)
        self.assertEqual(warm.weights(context(returns.iloc[:2])),
                         {"a": 1 / 3, "b": 1 / 3, "c": 1 / 3})

    def test_maximum_sharpe_failure_reports_diagnostics(self):
        returns = sample_returns(periods=200)
        optimizer = MSPOptimizer(lookback=200, min_observations=20)
        class Failed:
            success = False
            status = 9
            message = "iteration limit"
            nit = 1000
            fun = 1.0
            x = np.array([1 / 3, 1 / 3, 1 / 3])
        with mock.patch("backtesting.portfolio.minimize", return_value=Failed()):
            with self.assertRaises(PortfolioOptimizationError) as caught:
                optimizer.weights(context(returns))
        self.assertEqual(caught.exception.status, "max_sharpe_failed")
        self.assertEqual(caught.exception.iterations, 1000)
        self.assertIn("iteration limit", str(caught.exception))
        self.assertEqual(optimizer.last_result["status"], "max_sharpe_failed")

    def test_maximum_diversification_beats_equal_and_grid(self):
        returns = sample_returns(periods=500)
        optimizer = MDPOptimizer(lookback=500, min_observations=20)
        weights = optimizer.weights(context(returns))
        values = np.array([weights["a"], weights["b"], weights["c"]])
        self.assertAlmostEqual(values.sum(), 1)
        self.assertTrue((values >= 0).all())
        covariance, _ = ledoit_wolf_covariance(returns)
        volatility = np.sqrt(np.diag(covariance))
        ratio = (values @ volatility) / np.sqrt(values @ covariance @ values)
        grid = np.linspace(0, 1, 101)
        best = -np.inf
        for first in grid:
            for second in grid:
                third = 1 - first - second
                if third < -1e-12:
                    continue
                w = np.array([first, second, max(third, 0.0)])
                if w.sum() <= 0:
                    continue
                w = w / w.sum()
                value = (w @ volatility) / np.sqrt(w @ covariance @ w)
                best = max(best, value)
        self.assertGreaterEqual(ratio, best - 2e-3)
        equal = np.full(3, 1 / 3)
        equal_ratio = (equal @ volatility) / np.sqrt(equal @ covariance @ equal)
        self.assertGreaterEqual(ratio, equal_ratio - 1e-9)
        self.assertEqual(optimizer.last_result["status"], "converged")
        self.assertAlmostEqual(optimizer.last_result["diversification_ratio"], ratio)

    def test_maximum_diversification_zero_variance_and_failure(self):
        dates = pd.bdate_range("2020-01-01", periods=100)
        rng = np.random.default_rng(3)
        returns = pd.DataFrame({"flat": np.zeros(100),
                                "risky": rng.normal(0, .01, 100),
                                "risky2": rng.normal(0, .02, 100)}, index=dates)
        optimizer = MDPOptimizer(lookback=100, min_observations=20)
        weights = optimizer.weights(context(returns))
        self.assertEqual(weights["flat"], 0.0)
        self.assertAlmostEqual(weights["risky"] + weights["risky2"], 1)

        failing = MDPOptimizer(lookback=100, min_observations=20)
        class Failed:
            success = False
            status = 4
            message = "singular matrix"
            nit = 3
            fun = 1.0
            x = np.array([.5, .5, .5])
        with mock.patch("backtesting.portfolio.minimize", return_value=Failed()):
            with self.assertRaises(PortfolioOptimizationError) as caught:
                failing.weights(context(returns))
        self.assertEqual(caught.exception.status, "max_diversification_failed")
        self.assertEqual(failing.last_result["status"], "max_diversification_failed")

    def test_aliases_and_settings(self):
        self.assertIs(MSPOptimizer, MaximumSharpeOptimizer)
        self.assertIs(MDPOptimizer, MaximumDiversificationOptimizer)
        msp = MSPOptimizer(annual_rf=0.03, covariance="ledoit_wolf")
        self.assertEqual(msp.settings()["annual_rf"], 0.03)
        self.assertEqual(msp.settings()["covariance"], "ledoit_wolf")
        mdp = MDPOptimizer(shrinkage=0.1)
        self.assertEqual(mdp.settings()["covariance"], "diagonal")
        self.assertEqual(mdp.settings()["shrinkage"], 0.1)


if __name__ == "__main__":
    unittest.main()
