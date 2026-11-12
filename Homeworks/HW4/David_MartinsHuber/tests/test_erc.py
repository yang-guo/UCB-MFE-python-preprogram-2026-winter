"""Equal-risk-contribution allocation, long-only constraints, and failure reporting."""

from unittest import mock
import unittest

import numpy as np
import pandas as pd

from backtesting import (AllocationContext, ERCOptimizer, Phase, Portfolio,
                         PortfolioOptimizationError, ShadowPortfolioBacktester)
from tests.test_backtesting import inputs
from tests.test_portfolio import FractionStrategy


def context(returns):
    dates = returns.index
    return AllocationContext(dates[-1], dates[-1] + pd.Timedelta(days=1),
                             Portfolio(100, 100, 100, {}), returns)


def risk_contributions(weights, covariance):
    weights = np.asarray(weights, dtype=float)
    sigma = np.sqrt(weights @ covariance @ weights)
    return weights * (covariance @ weights) / sigma


class ERCChecks(unittest.TestCase):
    def test_two_asset_solution_is_inverse_volatility(self):
        dates = pd.bdate_range("2020-01-01", periods=4)
        returns = pd.DataFrame({"a": [.01, -.01, .01, -.01],
                                "b": [.02, .02, -.02, -.02]}, index=dates)
        optimizer = ERCOptimizer(lookback=4, min_observations=4, shrinkage=0)
        weights = optimizer.weights(context(returns))
        self.assertAlmostEqual(weights["a"], 2 / 3, places=10)
        self.assertAlmostEqual(weights["b"], 1 / 3, places=10)
        self.assertEqual(optimizer.last_result["status"], "converged")
        self.assertLess(optimizer.last_result["dispersion"], 1e-12)

    def test_three_asset_risk_contributions_are_equal(self):
        rng = np.random.default_rng(7)
        dates = pd.bdate_range("2020-01-01", periods=160)
        returns = pd.DataFrame(rng.normal(0, [.01, .02, .03], size=(160, 3)),
                               columns=["a", "b", "c"], index=dates)
        optimizer = ERCOptimizer(lookback=160, min_observations=20, shrinkage=.1)
        weights = optimizer.weights(context(returns))
        values = np.array([weights[n] for n in ["a", "b", "c"]])
        covariance = returns.cov().to_numpy()
        covariance = .9 * covariance + .1 * np.diag(np.diag(covariance))
        contributions = risk_contributions(values, covariance)
        target = contributions.mean()
        dispersion = np.max(np.abs(contributions - target)) / target
        self.assertGreaterEqual(values.min(), 0)
        self.assertAlmostEqual(values.sum(), 1)
        self.assertLess(dispersion, 1e-6)
        self.assertLess(optimizer.last_result["dispersion"], 1e-6)

    def test_max_weight_does_not_bind_and_weights_stay_capped(self):
        rng = np.random.default_rng(11)
        dates = pd.bdate_range("2020-01-01", periods=160)
        returns = pd.DataFrame(rng.normal(0, [.01, .02, .03], size=(160, 3)),
                               columns=["a", "b", "c"], index=dates)
        optimizer = ERCOptimizer(lookback=160, min_observations=20, max_weight=.9)
        weights = optimizer.weights(context(returns))
        self.assertTrue(all(0 <= w <= .9 + 1e-9 for w in weights.values()))
        self.assertAlmostEqual(sum(weights.values()), 1)

    def test_binding_max_weight_reports_non_erc_instead_of_silently_returning(self):
        rng = np.random.default_rng(11)
        dates = pd.bdate_range("2020-01-01", periods=160)
        returns = pd.DataFrame(rng.normal(0, [.01, .02, .03], size=(160, 3)),
                               columns=["a", "b", "c"], index=dates)
        optimizer = ERCOptimizer(lookback=160, min_observations=20, max_weight=.4)
        with self.assertRaises(PortfolioOptimizationError) as caught:
            optimizer.weights(context(returns))
        self.assertIn("exceeds max_weight", str(caught.exception))
        self.assertEqual(caught.exception.status, "cap_binding")
        self.assertGreater(max(caught.exception.weights), .4)
        self.assertIsNotNone(caught.exception.dispersion)
        self.assertIsNotNone(optimizer.last_result["dispersion"])

    def test_negative_correlations_are_solved_not_rejected(self):
        rng = np.random.default_rng(0)
        covariance = np.array([[0.140689, -0.505913, 0.046299],
                               [-0.505913, 6.941588, -0.179317],
                               [0.046299, -0.179317, 0.022276]])
        returns = pd.DataFrame(
            rng.normal(size=(5000, 3)) @ np.linalg.cholesky(covariance).T,
            columns=["a", "b", "c"],
            index=pd.bdate_range("2000-01-01", periods=5000))
        optimizer = ERCOptimizer(lookback=5000, min_observations=20, shrinkage=0)
        weights = optimizer.weights(context(returns))
        self.assertEqual(optimizer.last_result["status"], "converged")
        self.assertLess(optimizer.last_result["dispersion"], 1e-8)
        self.assertAlmostEqual(sum(weights.values()), 1)
        self.assertTrue(all(value >= 0 for value in weights.values()))

    def test_two_cycle_covariance_converges(self):
        rng = np.random.default_rng(0)
        correlation = np.array([[1, .3, -.5], [.3, 1, -.4], [-.5, -.4, 1]])
        volatility = np.array([1., 1.5, 2.])
        covariance = np.diag(volatility) @ correlation @ np.diag(volatility)
        returns = pd.DataFrame(
            rng.normal(size=(5000, 3)) @ np.linalg.cholesky(covariance).T,
            columns=["a", "b", "c"],
            index=pd.bdate_range("2000-01-01", periods=5000))
        optimizer = ERCOptimizer(lookback=5000, min_observations=20, shrinkage=0)
        weights = optimizer.weights(context(returns))
        self.assertEqual(optimizer.last_result["status"], "converged")
        self.assertLess(optimizer.last_result["dispersion"], 1e-8)
        self.assertAlmostEqual(sum(weights.values()), 1)

    def test_tiny_but_positive_variance_is_not_silently_zeroed(self):
        dates = pd.bdate_range("2020-01-01", periods=4)
        returns = pd.DataFrame({"low": [1e-6, -1e-6, 1e-6, -1e-6],
                                "high": [1., -1., 1., -1.]}, index=dates)
        optimizer = ERCOptimizer(lookback=4, min_observations=2, shrinkage=0)
        weights = optimizer.weights(context(returns))
        self.assertGreater(weights["low"], .999)
        self.assertGreater(weights["high"], 0)
        self.assertEqual(optimizer.last_result["status"], "converged")
        self.assertLess(optimizer.last_result["dispersion"], 1e-8)

    def test_tiny_return_scale_is_not_treated_as_zero_variance(self):
        dates = pd.bdate_range("2020-01-01", periods=4)
        returns = pd.DataFrame({"a": [.01, -.01, .01, -.01],
                                "b": [.02, .02, -.02, -.02]}, index=dates) * 1e-12
        optimizer = ERCOptimizer(lookback=4, min_observations=2, shrinkage=0)
        weights = optimizer.weights(context(returns))
        self.assertAlmostEqual(weights["a"], 2 / 3, places=10)
        self.assertAlmostEqual(weights["b"], 1 / 3, places=10)
        self.assertEqual(optimizer.last_result["status"], "converged")

    def test_last_result_weights_and_contributions_share_one_universe(self):
        dates = pd.bdate_range("2020-01-01", periods=4)
        returns = pd.DataFrame({"flat": [0., 0., 0., 0.],
                                "a": [.01, -.01, .01, -.01],
                                "b": [.02, .02, -.02, -.02]}, index=dates)
        optimizer = ERCOptimizer(lookback=4, min_observations=2, shrinkage=0)
        weights = optimizer.weights(context(returns))
        result = optimizer.last_result
        self.assertEqual(len(result["weights"]), 3)
        self.assertEqual(len(result["risk_contributions"]), 3)
        self.assertEqual(result["active"], [False, True, True])
        self.assertEqual(weights["flat"], 0.0)
        self.assertLess(result["dispersion"], 1e-8)

    def test_warmup_respects_an_infeasible_cap(self):
        dates = pd.bdate_range("2020-01-01", periods=2)
        returns = pd.DataFrame(np.ones((2, 4)) * .01,
                               columns=list("abcd"), index=dates)
        with self.assertRaisesRegex(ValueError, "infeasible"):
            ERCOptimizer(lookback=3, min_observations=2,
                         max_weight=.2).weights(context(returns))

    def test_duplicate_strategy_labels_are_rejected(self):
        dates = pd.bdate_range("2020-01-01", periods=4)
        returns = pd.DataFrame(np.random.default_rng(2).normal(size=(4, 3)),
                               columns=["a", "a", "b"], index=dates)
        with self.assertRaisesRegex(ValueError, "unique"):
            ERCOptimizer(lookback=4, min_observations=2).weights(context(returns))

    def test_nonfinite_input_has_structured_diagnostics(self):
        dates = pd.bdate_range("2020-01-01", periods=3)
        returns = pd.DataFrame({"a": [.01, np.nan, .01],
                                "b": [.01, .02, .03]}, index=dates)
        optimizer = ERCOptimizer(lookback=3, min_observations=2)
        with self.assertRaises(PortfolioOptimizationError) as caught:
            optimizer.weights(context(returns))
        self.assertEqual(caught.exception.status, "invalid_returns")
        self.assertEqual(caught.exception.iterations, 0)
        self.assertEqual(optimizer.last_result["status"], "invalid_returns")

    def test_maxiter_accepts_an_already_converged_point(self):
        rng = np.random.default_rng(0)
        correlation = np.array([[1, .3, -.5], [.3, 1, -.4], [-.5, -.4, 1]])
        volatility = np.array([1., 1.5, 2.])
        covariance = np.diag(volatility) @ correlation @ np.diag(volatility)
        returns = pd.DataFrame(
            rng.normal(size=(5000, 3)) @ np.linalg.cholesky(covariance).T,
            columns=["a", "b", "c"],
            index=pd.bdate_range("2000-01-01", periods=5000))
        optimizer = ERCOptimizer(lookback=5000, min_observations=20,
                                 shrinkage=0, maxiter=4)
        optimizer.weights(context(returns))
        self.assertEqual(optimizer.last_result["status"], "converged")
        self.assertLess(optimizer.last_result["dispersion"], 1e-8)

    def test_large_covariance_scale_does_not_cause_false_failure(self):
        rng = np.random.default_rng(0)
        correlation = np.array([[1, .3, -.5], [.3, 1, -.4], [-.5, -.4, 1]])
        volatility = np.array([1., 1.5, 2.])
        covariance = np.diag(volatility) @ correlation @ np.diag(volatility)
        returns = pd.DataFrame(
            rng.normal(size=(5000, 3)) @ np.linalg.cholesky(covariance).T,
            columns=["a", "b", "c"],
            index=pd.bdate_range("2000-01-01", periods=5000)) * 1e7
        optimizer = ERCOptimizer(lookback=5000, min_observations=20, shrinkage=.1)
        optimizer.weights(context(returns))
        self.assertEqual(optimizer.last_result["status"], "converged")
        self.assertLess(optimizer.last_result["dispersion"], 1e-8)

    def test_single_active_strategy_respects_max_weight(self):
        rng = np.random.default_rng(6)
        dates = pd.bdate_range("2020-01-01", periods=40)
        returns = pd.DataFrame({"flat": np.zeros(40),
                                "risky": rng.normal(0, .01, 40)}, index=dates)
        with self.assertRaisesRegex(ValueError, "infeasible"):
            ERCOptimizer(lookback=40, min_observations=3,
                         max_weight=.5, shrinkage=0).weights(context(returns))

    def test_mislabeled_converged_solve_is_reported_as_dispersion_check(self):
        rng = np.random.default_rng(3)
        dates = pd.bdate_range("2020-01-01", periods=10)
        returns = pd.DataFrame(rng.normal(size=(10, 3)),
                               columns=list("abc"), index=dates)
        bad = (np.array([.7, .2, .1]), 3, .1, "converged", "ok")
        with mock.patch.object(ERCOptimizer, "_solve", return_value=bad):
            with self.assertRaises(PortfolioOptimizationError) as caught:
                ERCOptimizer(lookback=10, min_observations=2).weights(context(returns))
        self.assertEqual(caught.exception.status, "dispersion_check")

    def test_invalid_solver_vectors_are_rejected(self):
        rng = np.random.default_rng(4)
        dates = pd.bdate_range("2020-01-01", periods=10)
        returns = pd.DataFrame(rng.normal(size=(10, 3)),
                               columns=list("abc"), index=dates)
        for vector in (np.array([.5, .5, .5]), np.array([.9, -.3, .4]),
                       np.array([.5, .5])):
            with mock.patch.object(ERCOptimizer, "_solve",
                                   return_value=(vector, 1, 1.0, "converged", "ok")):
                with self.assertRaises(PortfolioOptimizationError) as caught:
                    ERCOptimizer(lookback=10, min_observations=2).weights(context(returns))
            self.assertEqual(caught.exception.status, "invalid_weights")

    def test_invalid_covariance_populates_last_result(self):
        dates = pd.bdate_range("2020-01-01", periods=40)
        returns = pd.DataFrame({"a": np.linspace(-1, 1, 40) * 1e160,
                                "b": np.linspace(-1, 1, 40) * 1e160 * 1.0001},
                               index=dates)
        optimizer = ERCOptimizer(lookback=40, min_observations=2, shrinkage=0)
        with self.assertRaises(PortfolioOptimizationError) as caught:
            optimizer.weights(context(returns))
        self.assertEqual(caught.exception.status, "invalid_covariance")
        self.assertEqual(optimizer.last_result["status"], "invalid_covariance")

    def test_impossible_tolerance_is_not_reported_as_invalid_direction(self):
        rng = np.random.default_rng(5)
        dates = pd.bdate_range("2020-01-01", periods=10)
        returns = pd.DataFrame(rng.normal(size=(10, 2)),
                               columns=list("ab"), index=dates)
        optimizer = ERCOptimizer(lookback=10, min_observations=2,
                                 shrinkage=0, rc_tolerance=1e-18)
        try:
            optimizer.weights(context(returns))
        except PortfolioOptimizationError as error:
            self.assertNotEqual(error.status, "invalid_direction")

    def test_warmup_single_asset_and_zero_variance(self):
        dates = pd.bdate_range("2020-01-01", periods=4)
        returns = pd.DataFrame({"a": [.01, .02, .03, .04],
                                "b": [.01, .02, .03, .04]}, index=dates)
        warm = ERCOptimizer(lookback=5, min_observations=3)
        self.assertEqual(warm.weights(context(returns.iloc[:2])), {"a": .5, "b": .5})
        self.assertEqual(warm.last_result["solver_message"], "insufficient observations")

        one = ERCOptimizer(lookback=4, min_observations=2)
        self.assertEqual(one.weights(context(returns[["a"]])), {"a": 1.0})

        flat = pd.DataFrame({"a": [0., 0., 0., 0.], "b": [0., 0., 0., 0.]}, index=dates)
        zero = ERCOptimizer(lookback=4, min_observations=2, shrinkage=0)
        self.assertEqual(zero.weights(context(flat)), {"a": .5, "b": .5})

    def test_zero_variance_strategy_gets_zero_weight(self):
        dates = pd.bdate_range("2020-01-01", periods=4)
        returns = pd.DataFrame({"flat": [0., 0., 0., 0.],
                                "risky": [.01, -.01, .01, -.01]}, index=dates)
        optimizer = ERCOptimizer(lookback=4, min_observations=2, shrinkage=0)
        weights = optimizer.weights(context(returns))
        self.assertEqual(weights, {"flat": 0.0, "risky": 1.0})

    def test_nonconvergence_carries_status_message_and_iterations(self):
        rng = np.random.default_rng(7)
        dates = pd.bdate_range("2020-01-01", periods=160)
        returns = pd.DataFrame(rng.normal(0, [.01, .02, .03], size=(160, 3)),
                               columns=["a", "b", "c"], index=dates)
        optimizer = ERCOptimizer(lookback=160, min_observations=20, maxiter=1)
        with self.assertRaises(PortfolioOptimizationError) as caught:
            optimizer.weights(context(returns))
        error = caught.exception
        self.assertEqual(error.status, "max_iterations")
        self.assertEqual(error.iterations, 1)
        self.assertIn("did not converge to equal risk contributions", str(error))
        self.assertGreater(error.dispersion, optimizer.rc_tolerance)
        self.assertEqual(optimizer.last_result["status"], "max_iterations")

    def test_nonfinite_returns_are_rejected(self):
        dates = pd.bdate_range("2020-01-01", periods=3)
        returns = pd.DataFrame({"a": [.01, np.nan, .01], "b": [.01, .02, .03]}, index=dates)
        with self.assertRaises(PortfolioOptimizationError):
            ERCOptimizer(lookback=3, min_observations=2).weights(context(returns))

    def test_invalid_configuration_and_infeasible_cap(self):
        for kwargs in (dict(lookback=1, min_observations=2),
                       dict(shrinkage=1.1), dict(max_weight=0), dict(rc_tolerance=0),
                       dict(maxiter=0)):
            with self.assertRaises(ValueError):
                ERCOptimizer(**kwargs)
        rng = np.random.default_rng(3)
        dates = pd.bdate_range("2020-01-01", periods=40)
        returns = pd.DataFrame(rng.normal(size=(40, 3)), columns=list("abc"), index=dates)
        with self.assertRaisesRegex(ValueError, "infeasible"):
            ERCOptimizer(lookback=40, min_observations=20, max_weight=.3).weights(context(returns))

    def test_shadow_portfolio_integration_and_failure_state(self):
        dates, p, b, m = inputs(12)
        runner = ShadowPortfolioBacktester(
            p, b, m, [FractionStrategy("a"), FractionStrategy("b")],
            ERCOptimizer(lookback=5, min_observations=2),
            initial_cash=100, shadow_initial_cash=100)
        result = runner.run(dates[5], dates[-1], shadow_start=dates[1])
        self.assertEqual(runner.phase, Phase.COMPLETE)
        self.assertEqual(set(result.allocations.optimizer), {"EqualRiskContributionOptimizer"})
        for _, group in result.allocations.groupby("signal_date"):
            self.assertAlmostEqual(group.weight.sum(), 1)
            self.assertTrue(group.weight.ge(0).all())

        failing = ShadowPortfolioBacktester(
            p, b, m, [FractionStrategy("a"), FractionStrategy("b")],
            ERCOptimizer(lookback=5, min_observations=2, maxiter=2),
            initial_cash=100, shadow_initial_cash=100)
        # Force the internal solve to report non-convergence so the coordinator
        # must surface the optimization failure and stop before any main fill.
        forced = (np.array([.9, .1]), 1, 1.0, "max_iterations", "forced non-convergence")
        with mock.patch.object(ERCOptimizer, "_solve", return_value=forced):
            with self.assertRaises(PortfolioOptimizationError) as caught:
                failing.run(dates[5], dates[-1], shadow_start=dates[1])
        self.assertEqual(caught.exception.status, "max_iterations")
        self.assertEqual(failing.phase, Phase.FAILED)
        self.assertFalse(failing._main.oms.fills)


if __name__ == "__main__":
    unittest.main()
