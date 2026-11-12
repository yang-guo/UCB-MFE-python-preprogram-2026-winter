"""Synchronized shadow accounts and a strategy-compatible portfolio allocator.

All decisions precede the shared execution barrier. Shadow returns are generated
walk-forward, never reconstructed using a later fitted model.
"""

from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
import numbers
from typing import Mapping

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .contracts import Phase, Portfolio, Strategy, plain_amount
from .engine import Backtester, BacktestResult
from .risk import RiskManager


@dataclass(frozen=True)
class AllocationContext:
    signal_date: pd.Timestamp
    execution_date: pd.Timestamp
    portfolio: Portfolio
    shadow_returns: pd.DataFrame


class PortfolioOptimizer(ABC):
    """Receive completed shadow returns only; do not load external future data."""

    @abstractmethod
    def weights(self, context: AllocationContext) -> Mapping[str, float]:
        """Nonnegative weights indexed by shadow strategy ID, summing to one."""

    def settings(self):
        return {}


class PortfolioOptimizationError(ValueError):
    """An optimizer could not produce a valid, converged allocation.

    The exception carries the solver status, message, iteration count,
    objective, candidate weights, risk contributions, and the maximum relative
    dispersion from equal risk contribution.  Callers can therefore report a
    failed optimization instead of silently using a bad allocation.
    """

    def __init__(self, message, *, optimizer="", status=None, solver_message=None,
                 iterations=None, objective=None, weights=None,
                 risk_contributions=None, dispersion=None):
        details = [message]
        if status is not None:
            details.append(f"status={status}")
        if solver_message:
            details.append(f"solver_message={solver_message!r}")
        if iterations is not None:
            details.append(f"iterations={iterations}")
        if objective is not None:
            details.append(f"objective={objective!r}")
        if dispersion is not None:
            details.append(f"max_relative_rc_dispersion={dispersion!r}")
        super().__init__("; ".join(details))
        self.optimizer = optimizer
        self.status = status
        self.solver_message = solver_message
        self.iterations = iterations
        self.objective = objective
        self.weights = self._as_list(weights)
        self.risk_contributions = self._as_list(risk_contributions)
        self.dispersion = dispersion

    @staticmethod
    def _as_list(value):
        if value is None:
            return None
        array = np.asarray(value)
        if array.ndim == 0:
            return [array.item()]
        return array.tolist()


class EqualWeightOptimizer(PortfolioOptimizer):
    def weights(self, context):
        names = context.shadow_returns.columns
        return dict.fromkeys(names, 1 / len(names))


class MinimumVarianceOptimizer(PortfolioOptimizer):
    """Long-only minimum sample-variance weights over trailing shadow returns.

    Equal weights during warm-up or if all series have zero variance. Shrinkage
    toward diagonal covariance limits sensitivity to estimated correlations.
    This estimates risk from standalone accounts, not live contribution PnL.
    """

    def __init__(self, lookback=63, min_observations=20, shrinkage=0.1, max_weight=1.0):
        if (isinstance(lookback, bool) or not isinstance(lookback, numbers.Integral)
                or isinstance(min_observations, bool)
                or not isinstance(min_observations, numbers.Integral)):
            raise ValueError("Require 2 <= min_observations <= lookback")
        lookback, min_observations = int(lookback), int(min_observations)
        if not 2 <= min_observations <= lookback:
            raise ValueError("Require 2 <= min_observations <= lookback")
        try:
            shrinkage, max_weight = plain_amount(shrinkage), plain_amount(max_weight)
        except ValueError:
            raise ValueError("Invalid shrinkage or maximum weight") from None
        if not 0 <= shrinkage <= 1 or not 0 < max_weight <= 1:
            raise ValueError("Invalid shrinkage or maximum weight")
        self.lookback, self.min_observations = lookback, min_observations
        self.shrinkage, self.max_weight = shrinkage, max_weight

    def settings(self):
        return dict(lookback=self.lookback, min_observations=self.min_observations,
                    shrinkage=self.shrinkage, max_weight=self.max_weight)

    def weights(self, context):
        returns = context.shadow_returns.tail(self.lookback)
        names, count = list(returns.columns), len(returns.columns)
        if count * self.max_weight < 1 - 1e-10:
            raise ValueError("Maximum weight is infeasible for this strategy count")
        equal = np.full(count, 1 / count)
        if len(returns) < self.min_observations or count == 1:
            return dict(zip(names, equal))
        covariance = returns.cov().to_numpy()
        covariance = (1 - self.shrinkage) * covariance + self.shrinkage * np.diag(np.diag(covariance))
        scale = np.max(np.abs(covariance))
        if scale <= 1e-24:
            return dict(zip(names, equal))
        covariance /= scale  # Conditioning only; does not change the optimum.
        solved = minimize(lambda w: float(w @ covariance @ w), equal,
                          jac=lambda w: 2 * covariance @ w,
                          method="SLSQP", bounds=[(0, self.max_weight)] * count,
                          constraints={"type": "eq", "fun": lambda w: w.sum() - 1,
                                       "jac": lambda w: np.ones(count)},
                          options={"ftol": 1e-12, "maxiter": 200})
        if not solved.success:
            raise ValueError(f"Portfolio optimization failed: {solved.message}")
        return dict(zip(names, solved.x))


class EqualRiskContributionOptimizer(PortfolioOptimizer):
    """Long-only equal-risk-contribution weights over trailing shadow returns.

    A strategy's risk contribution is ``w_i * (Sigma w)_i / sigma_p``.  The
    optimizer seeks equal contributions subject to nonnegative weights, full
    investment, and an optional per-strategy cap.  It is therefore long-only
    and never uses leverage.

    Warm-up uses equal weights.  A strategy with zero sample variance receives
    zero weight; the remaining active strategies are solved on the reduced
    covariance.  If the Newton solver does not converge, or returns a
    point whose risk contributions are not equal within ``rc_tolerance``, the
    optimizer raises ``PortfolioOptimizationError`` with the iteration status,
    message, iteration count, objective, weights, and risk-contribution
    dispersion.  It never silently returns an unvalidated allocation.
    """

    def __init__(self, lookback=63, min_observations=20, shrinkage=0.1,
                 max_weight=1.0, rc_tolerance=1e-8, maxiter=1000, ftol=1e-14):
        if (isinstance(lookback, bool) or not isinstance(lookback, numbers.Integral)
                or isinstance(min_observations, bool)
                or not isinstance(min_observations, numbers.Integral)):
            raise ValueError("Require 2 <= min_observations <= lookback")
        lookback, min_observations = int(lookback), int(min_observations)
        if not 2 <= min_observations <= lookback:
            raise ValueError("Require 2 <= min_observations <= lookback")
        try:
            shrinkage = plain_amount(shrinkage)
            max_weight = plain_amount(max_weight)
            rc_tolerance = plain_amount(rc_tolerance)
            ftol = plain_amount(ftol)
        except ValueError:
            raise ValueError("Invalid ERC numeric setting") from None
        if not 0 <= shrinkage <= 1 or not 0 < max_weight <= 1:
            raise ValueError("Invalid shrinkage or maximum weight")
        if isinstance(maxiter, bool) or not isinstance(maxiter, numbers.Integral):
            raise ValueError("Invalid ERC tolerance or iteration limit")
        maxiter = int(maxiter)
        if not 0 < rc_tolerance <= 1 or not 0 < ftol <= 1 or not 1 <= maxiter <= 1_000_000:
            raise ValueError("Invalid ERC tolerance or iteration limit")
        self.lookback, self.min_observations = lookback, min_observations
        self.shrinkage, self.max_weight = shrinkage, max_weight
        self.rc_tolerance, self.maxiter, self.ftol = rc_tolerance, maxiter, ftol
        self.last_result = None

    def settings(self):
        return dict(lookback=self.lookback, min_observations=self.min_observations,
                    shrinkage=self.shrinkage, max_weight=self.max_weight,
                    rc_tolerance=self.rc_tolerance, maxiter=self.maxiter, ftol=self.ftol)

    @np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore")
    def weights(self, context):
        self.last_result = None
        returns = context.shadow_returns.tail(self.lookback)
        names = list(returns.columns)
        count = len(names)
        if count == 0:
            self.last_result = dict(status="invalid_input",
                                    solver_message="no strategy return series",
                                    iterations=0, objective=None, weights=None,
                                    risk_contributions=None, active=None, dispersion=None)
            raise PortfolioOptimizationError(
                "ERC requires at least one strategy return series",
                optimizer=type(self).__name__, status="invalid_input",
                solver_message="no strategy return series", iterations=0)
        if len(set(names)) != count:
            self.last_result = dict(status="invalid_input",
                                    solver_message="duplicate strategy identifiers",
                                    iterations=0, objective=None, weights=None,
                                    risk_contributions=None, active=None, dispersion=None)
            raise PortfolioOptimizationError(
                "ERC requires unique strategy identifiers",
                optimizer=type(self).__name__, status="invalid_input",
                solver_message="duplicate strategy identifiers", iterations=0)
        if np.iscomplexobj(returns.to_numpy()):
            self.last_result = dict(status="invalid_returns",
                                    solver_message="complex returns",
                                    iterations=0, objective=None, weights=None,
                                    risk_contributions=None, active=None, dispersion=None)
            raise PortfolioOptimizationError(
                "ERC returns must be real-valued", optimizer=type(self).__name__,
                status="invalid_returns", solver_message="complex returns", iterations=0)
        if any(pd.api.types.is_datetime64_any_dtype(dtype)
               or pd.api.types.is_timedelta64_dtype(dtype) for dtype in returns.dtypes):
            self.last_result = dict(status="invalid_returns",
                                    solver_message="datetime returns",
                                    iterations=0, objective=None, weights=None,
                                    risk_contributions=None, active=None, dispersion=None)
            raise PortfolioOptimizationError(
                "ERC returns must be numeric, not dates or durations",
                optimizer=type(self).__name__, status="invalid_returns",
                solver_message="datetime returns", iterations=0)
        try:
            values = returns.to_numpy(dtype=float)
        except (TypeError, ValueError):
            self.last_result = dict(status="invalid_returns",
                                    solver_message="non-numeric returns",
                                    iterations=0, objective=None, weights=None,
                                    risk_contributions=None, active=None, dispersion=None)
            raise PortfolioOptimizationError(
                "ERC returns must be numeric", optimizer=type(self).__name__,
                status="invalid_returns", solver_message="non-numeric returns",
                iterations=0) from None
        if not np.isfinite(values).all():
            self.last_result = dict(status="invalid_returns",
                                    solver_message="nonfinite returns",
                                    iterations=0, objective=None, weights=None,
                                    risk_contributions=None, active=None, dispersion=None)
            raise PortfolioOptimizationError(
                "ERC returns contain nonfinite values", optimizer=type(self).__name__,
                status="invalid_returns", solver_message="nonfinite returns", iterations=0)
        equal = np.full(count, 1 / count)
        if count * self.max_weight < 1 - 1e-12:
            self.last_result = dict(status="infeasible_cap",
                                    solver_message="maximum weight is infeasible",
                                    iterations=0, objective=None, weights=None,
                                    risk_contributions=None, active=None, dispersion=None)
            raise PortfolioOptimizationError(
                "Maximum weight is infeasible for this strategy count",
                optimizer=type(self).__name__, status="infeasible_cap",
                solver_message="maximum weight is infeasible", iterations=0)
        if count == 1:
            observed = len(returns) > 0
            self.last_result = dict(status="single_strategy", solver_message="one strategy",
                                    iterations=0, objective=None, weights=[1.0],
                                    risk_contributions=[1.0] if observed else None,
                                    active=[True], dispersion=0.0 if observed else None)
            return {names[0]: 1.0}
        if len(returns) < self.min_observations:
            self.last_result = dict(status="warmup", solver_message="insufficient observations",
                                    iterations=0, objective=None, weights=equal.tolist(),
                                    risk_contributions=None, active=None,
                                    dispersion=None)
            return dict(zip(names, equal))

        with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
            covariance = returns.cov().to_numpy()
            covariance = ((1 - self.shrinkage) * covariance
                          + self.shrinkage * np.diag(np.diag(covariance)))
        if not np.isfinite(covariance).all():
            self.last_result = dict(status="invalid_covariance",
                                    solver_message="covariance is not finite",
                                    iterations=0, objective=None, weights=None,
                                    risk_contributions=None, active=None, dispersion=None)
            raise PortfolioOptimizationError(
                "ERC covariance is not finite", optimizer=type(self).__name__,
                status="invalid_covariance", solver_message="covariance is not finite",
                iterations=0)
        diagonal = np.diag(covariance)
        if not np.isfinite(diagonal).all() or np.any(diagonal < 0):
            self.last_result = dict(status="invalid_covariance",
                                    solver_message="invalid variances",
                                    iterations=0, objective=None, weights=None,
                                    risk_contributions=None, active=None, dispersion=None)
            raise PortfolioOptimizationError(
                "ERC covariance has invalid variances", optimizer=type(self).__name__,
                status="invalid_covariance", solver_message="invalid variances",
                iterations=0)
        active = diagonal > 0
        if not active.any():
            self.last_result = dict(status="zero_variance",
                                    solver_message="all variances are zero",
                                    iterations=0, objective=None, weights=equal.tolist(),
                                    risk_contributions=None, active=active.tolist(),
                                    dispersion=None)
            return dict(zip(names, equal))
        active_names = [name for name, keep in zip(names, active) if keep]
        if len(active_names) * self.max_weight < 1 - 1e-12:
            self.last_result = dict(status="infeasible_cap",
                                    solver_message="maximum weight is infeasible for active strategies",
                                    iterations=0, objective=None, weights=None,
                                    risk_contributions=None, active=active.tolist(),
                                    dispersion=None)
            raise PortfolioOptimizationError(
                "Maximum weight is infeasible for the active strategy count",
                optimizer=type(self).__name__, status="infeasible_cap",
                solver_message="maximum weight is infeasible for active strategies",
                iterations=0)
        if len(active_names) == 1:
            weights = np.zeros(count)
            weights[active] = 1.0
            variance = float(weights @ covariance @ weights)
            sigma = np.sqrt(variance)
            contributions = weights * (covariance @ weights) / sigma
            self.last_result = dict(status="single_active",
                                    solver_message="one positive-variance strategy",
                                    iterations=0, objective=None, weights=weights.tolist(),
                                    risk_contributions=contributions.tolist(),
                                    active=active.tolist(), dispersion=0.0)
            return {name: float(weight) for name, weight in zip(names, weights)}

        sub_weights, iterations, objective, status, message = self._solve(
            covariance[np.ix_(active, active)], len(active_names))
        try:
            sub_weights = np.asarray(sub_weights, dtype=float).reshape(-1)
        except (TypeError, ValueError):
            self.last_result = dict(status="invalid_weights",
                                    solver_message="solver returned non-numeric weights",
                                    iterations=iterations, objective=objective,
                                    weights=None, risk_contributions=None,
                                    active=active.tolist(), dispersion=None)
            raise PortfolioOptimizationError(
                "ERC returned non-numeric weights",
                optimizer=type(self).__name__, status="invalid_weights",
                solver_message="solver returned non-numeric weights",
                iterations=iterations, objective=objective) from None
        if (sub_weights.size != len(active_names)
                or not np.isfinite(sub_weights).all()
                or np.any(sub_weights < -1e-12)
                or not np.isclose(sub_weights.sum(), 1.0, atol=1e-8, rtol=0)):
            self.last_result = dict(status="invalid_weights",
                                    solver_message="solver returned an invalid weight vector",
                                    iterations=iterations, objective=objective,
                                    weights=sub_weights.tolist(),
                                    risk_contributions=None, active=active.tolist(),
                                    dispersion=None)
            raise PortfolioOptimizationError(
                "ERC returned an invalid weight vector",
                optimizer=type(self).__name__, weights=sub_weights,
                iterations=iterations, objective=objective,
                status="invalid_weights",
                solver_message="solver returned an invalid weight vector")
        sub_weights = np.clip(sub_weights, 0.0, None)
        sub_weights /= sub_weights.sum()
        weights = np.zeros(count)
        weights[active] = sub_weights
        variance = float(weights @ covariance @ weights)
        if not np.isfinite(variance) or variance <= 0:
            self.last_result = dict(status="invalid_variance",
                                    solver_message="nonpositive portfolio variance",
                                    iterations=iterations, objective=objective,
                                    weights=weights.tolist(), risk_contributions=None,
                                    active=active.tolist(), dispersion=None)
            raise PortfolioOptimizationError(
                "ERC produced a nonpositive portfolio variance",
                optimizer=type(self).__name__, weights=weights,
                iterations=iterations, objective=objective,
                status="invalid_variance", solver_message="nonpositive portfolio variance")
        sigma = np.sqrt(variance)
        contributions = weights * (covariance @ weights) / sigma
        target = sigma / len(active_names)
        dispersion = float(np.max(np.abs(contributions[active] - target)) / target)
        if not np.isfinite(dispersion) or dispersion > 1e12:
            self.last_result = dict(status="singular_covariance",
                                    solver_message="portfolio variance is numerically zero",
                                    iterations=iterations, objective=objective,
                                    weights=weights.tolist(),
                                    risk_contributions=contributions.tolist(),
                                    active=active.tolist(), dispersion=dispersion)
            raise PortfolioOptimizationError(
                "ERC portfolio variance is numerically zero",
                optimizer=type(self).__name__, weights=weights,
                risk_contributions=contributions, dispersion=dispersion,
                status="singular_covariance",
                solver_message="portfolio variance is numerically zero",
                iterations=iterations, objective=objective)
        converged = dispersion <= self.rc_tolerance
        if not converged and status == "converged":
            status = "dispersion_check"
        diagnostics = dict(status="converged" if converged else status,
                           solver_message=message, iterations=iterations,
                           objective=objective)
        self.last_result = {**diagnostics, "weights": weights.tolist(),
                            "risk_contributions": contributions.tolist(),
                            "active": active.tolist(), "dispersion": dispersion}
        if not converged:
            raise PortfolioOptimizationError(
                "ERC did not converge to equal risk contributions",
                optimizer=type(self).__name__, weights=weights,
                risk_contributions=contributions, dispersion=dispersion, **diagnostics)
        if np.any(weights > self.max_weight + 1e-15):
            index = int(np.argmax(weights))
            self.last_result = {**diagnostics, "status": "cap_binding",
                                "solver_message": "maximum-weight cap binds",
                                "weights": weights.tolist(),
                                "risk_contributions": contributions.tolist(),
                                "active": active.tolist(), "dispersion": dispersion}
            raise PortfolioOptimizationError(
                f"ERC weight {weights[index]:.6f} exceeds max_weight={self.max_weight}",
                optimizer=type(self).__name__, weights=weights,
                risk_contributions=contributions, dispersion=dispersion,
                status="cap_binding", solver_message="maximum-weight cap binds",
                iterations=iterations, objective=objective)
        return {name: float(weight) for name, weight in zip(names, weights)}

    @staticmethod
    def _objective(y, covariance, count):
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            value = 0.5 * float(y @ covariance @ y) - float(np.mean(np.log(y)))
        return value if np.isfinite(value) else float("inf")

    @np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore")
    def _solve(self, covariance, count):
        # Newton on the strictly convex Spinu objective
        # 0.5 y' Sigma y - mean(log y); the normalized y is ERC.  The Hessian
        # is positive definite for y > 0, and a backtracking line search makes
        # the iteration globally convergent for any positive-definite covariance,
        # including negative correlations that break a naive fixed point.
        diagonal = np.diag(covariance)
        volatility = np.sqrt(diagonal)
        y = (1 / volatility) / np.sum(1 / volatility)
        variance = float(y @ covariance @ y)
        if np.isfinite(variance) and variance > 0:
            y = y / np.sqrt(variance)
        def normalized(vector):
            total = float(vector.sum())
            return vector / total if np.isfinite(total) and total > 0 else vector

        def dispersion_of(vector):
            weights = normalized(vector)
            variance = float(weights @ covariance @ weights)
            if not np.isfinite(variance) or variance <= 0:
                return float("inf")
            sigma = np.sqrt(variance)
            contributions = weights * (covariance @ weights) / sigma
            target = sigma / count
            return float(np.max(np.abs(contributions - target)) / target)

        value = self._objective(y, covariance, count)
        if not np.isfinite(value):
            return normalized(y), 0, value, "invalid_objective", "initial objective is not finite"
        best_y, best_value = y.copy(), value
        best_dispersion = dispersion_of(y)
        stalled = 0
        for iteration in range(1, self.maxiter + 1):
            if best_dispersion <= self.rc_tolerance:
                return (normalized(best_y), iteration - 1, best_value,
                        "converged", "risk contributions converged")
            gradient = covariance @ y - 1 / (count * y)
            if not np.isfinite(gradient).all():
                return normalized(best_y), iteration, best_value, "invalid_gradient", "gradient is not finite"
            hessian = covariance + np.diag(1 / (count * y * y))
            try:
                step = np.linalg.solve(hessian, -gradient)
            except np.linalg.LinAlgError:
                step = np.linalg.lstsq(hessian, -gradient, rcond=None)[0]
            if not np.isfinite(step).all():
                return normalized(best_y), iteration, best_value, "invalid_step", "Newton step is not finite"
            slope = float(gradient @ step)
            if not np.isfinite(slope) or slope >= 0:
                step = -gradient
                slope = float(gradient @ step)
                if not np.isfinite(slope) or slope >= 0:
                    gradient_norm = float(np.max(np.abs(gradient)))
                    if gradient_norm <= 1e-12 * max(1.0, float(np.max(np.abs(y)))):
                        return (normalized(best_y), iteration, best_value, "dispersion_check",
                                "gradient is zero but risk contributions exceed rc_tolerance")
                    return normalized(best_y), iteration, best_value, "invalid_direction", "no descent direction"
            alpha = 1.0
            accepted = False
            candidate = y
            candidate_value = value
            current_dispersion = dispersion_of(y)
            for _ in range(60):
                candidate = y + alpha * step
                if np.all(candidate > 0):
                    candidate_value = self._objective(candidate, covariance, count)
                    candidate_dispersion = dispersion_of(candidate)
                    slack = 1e-14 * max(1.0, abs(value))
                    if (np.isfinite(candidate_value)
                            and (candidate_value <= value + 1e-4 * alpha * slope + slack
                                 or candidate_dispersion < current_dispersion)):
                        accepted = True
                        break
                alpha *= 0.5
            if not accepted:
                return normalized(best_y), iteration, best_value, "line_search_failed", "line search could not decrease the objective"
            y = candidate
            value = candidate_value
            current_dispersion = dispersion_of(y)
            if current_dispersion < best_dispersion:
                best_y, best_value, best_dispersion = y.copy(), value, current_dispersion
            if best_dispersion <= self.rc_tolerance:
                return (normalized(best_y), iteration, best_value,
                        "converged", "risk contributions converged")
            step_norm = float(np.max(np.abs(alpha * step)))
            scale = max(float(np.max(np.abs(y))), np.finfo(float).tiny)
            if step_norm <= self.ftol * scale:
                stalled += 1
                if stalled >= 5 and best_dispersion > self.rc_tolerance:
                    return (normalized(best_y), iteration, best_value, "dispersion_check",
                            "step below ftol but risk contributions are not equal")
            else:
                stalled = 0
        return normalized(best_y), self.maxiter, best_value, "max_iterations", "iteration limit reached"


ERCOptimizer = EqualRiskContributionOptimizer


@dataclass(frozen=True)
class ShadowSpec:
    strategy: Strategy
    training_panel: pd.DataFrame | None = None
    feature_columns: tuple[str, ...] = ()
    risk_manager: RiskManager | None = None


class PortfolioStrategy(Strategy):
    """Adapter from synchronized component proposals to one OMS budget mapping.

    Component deployment fractions are budget / shadow NAV. Weight these by
    portfolio NAV, then scale all proposed buys to signal-date available cash.
    No existing lot is rebalanced or sold early. Contributions are proposal
    attribution, not realized component positions or independently earned PnL.
    """

    def __init__(self, strategy_id, optimizer):
        super().__init__(strategy_id)
        if not isinstance(optimizer, PortfolioOptimizer):
            raise TypeError("optimizer must implement PortfolioOptimizer")
        self.optimizer = deepcopy(optimizer)
        self._cutoff = None
        self._navs = None
        self._returns = None
        self._proposals = None
        self._versions = None
        self.allocations, self.contributions = [], []

    def observe(self, cutoff, navs, proposals, versions):
        """Publish one detached, append-only history at the shared signal close."""
        cutoff = pd.Timestamp(cutoff)
        if (not isinstance(navs.index, pd.DatetimeIndex) or navs.index.has_duplicates
                or not navs.index.is_monotonic_increasing or navs.empty
                or navs.index[-1] != cutoff or not np.isfinite(navs.to_numpy()).all()
                or navs.le(0).any().any() or navs.columns.has_duplicates):
            raise ValueError("Shadow NAVs must be synchronized, finite, and positive through cutoff")
        if self._cutoff is not None:
            if cutoff <= self._cutoff:
                raise ValueError("Cannot rewrite or republish shadow history")
            if not navs.loc[:self._cutoff].equals(self._navs):
                raise ValueError("Previously published shadow history was changed")
        if set(navs.columns) != set(proposals) or set(proposals) != set(versions):
            raise ValueError("Every shadow needs current proposals and model provenance")
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            returns = navs.pct_change(fill_method=None).iloc[1:]
        if not np.isfinite(returns.to_numpy()).all():
            raise ValueError("Derived shadow returns are not finite")
        self._cutoff = cutoff
        self._navs = navs.copy(deep=True)
        self._returns = returns.copy(deep=True)
        self._proposals = deepcopy(proposals)
        self._versions = dict(versions)

    def propose_buys(self, context):
        if self._cutoff != context.signal_date:
            raise ValueError("No synchronized shadow history at this decision cutoff")
        if (not np.isfinite([context.portfolio.nav, context.portfolio.available_cash]).all()
                or context.portfolio.nav <= 0 or context.portfolio.available_cash < 0):
            raise ValueError("Allocation requires finite positive NAV and nonnegative available cash")
        returns = self._returns.copy(deep=True)
        allocation_context = AllocationContext(context.signal_date, context.execution_date,
                                               deepcopy(context.portfolio), returns.copy(deep=True))
        raw_weights = dict(self.optimizer.weights(allocation_context))
        if any(type(key) is not str or not key for key in raw_weights):
            raise ValueError("Optimizer weights must cover all shadows, be nonnegative, and sum to one")
        try:
            weights = {key: plain_amount(value) for key, value in raw_weights.items()}
        except ValueError:
            raise ValueError("Optimizer weights must cover all shadows, be nonnegative, and sum to one") from None
        if (set(weights) != set(self._navs.columns)
                or any(w < 0 for w in weights.values())
                or not np.isclose(sum(weights.values()), 1, atol=1e-8, rtol=0)):
            raise ValueError("Optimizer weights must cover all shadows, be nonnegative, and sum to one")
        planned = []
        for strategy_id, weight in weights.items():
            for symbol, budget in self._proposals[strategy_id].items():
                if not np.isfinite(budget) or budget < 0:
                    raise ValueError("Invalid shadow purchase budget")
                with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                    fraction = budget / self._navs[strategy_id].iloc[-1]
                    amount = context.portfolio.nav * (weight * fraction)
                if not np.isfinite(amount):
                    raise ValueError("Portfolio allocation arithmetic is not finite")
                if amount > 0:
                    planned.append((strategy_id, symbol, amount))
        total = sum(amount for _, _, amount in planned)
        if not np.isfinite(total):
            raise ValueError("Total proposed allocation is not finite")
        scale = min(1.0, context.portfolio.available_cash / total) if total else 1.0
        orders = {}
        for strategy_id, symbol, amount in planned:
            amount *= scale
            if amount > 0:
                orders[symbol] = orders.get(symbol, 0.0) + amount
                self.contributions.append(dict(signal_date=context.signal_date,
                                               execution_date=context.execution_date,
                                               component_id=strategy_id, symbol=symbol,
                                               budget=amount, model_version=self._versions[strategy_id]))
        for strategy_id, weight in weights.items():
            self.allocations.append(dict(signal_date=context.signal_date,
                                         execution_date=context.execution_date, component_id=strategy_id,
                                         weight=weight, model_version=self._versions[strategy_id],
                                         return_cutoff=(returns.index[-1] if len(returns) else pd.NaT),
                                         return_observations=len(returns), cash_scale=scale,
                                         optimizer=type(self.optimizer).__name__,
                                         settings=deepcopy(self.optimizer.settings())))
        return orders


@dataclass
class PortfolioBacktestResult:
    portfolio: BacktestResult
    shadows: dict[str, BacktestResult]
    allocations: pd.DataFrame
    contributions: pd.DataFrame
    approved_contributions: pd.DataFrame = field(default_factory=pd.DataFrame)


class ShadowPortfolioBacktester:
    """A shared clock: prepare every account, then execute every account.

    Each shadow owns fresh strategy/model state and an independent OMS. The
    optimizer never receives shadow engines, fitting panels, or future prices.
    It sees only completed returns; this is a data contract, not a Python sandbox.
    """

    def __init__(self, prices=None, benchmark=None, membership=None, shadows=None, optimizer=None, *,
                 market_data=None, strategy_id="portfolio", fundamentals=None, initial_cash=1_000_000,
                 shadow_initial_cash=1_000_000, fee_bps=5, max_stale_sessions=5, risk_manager=None,
                 delisting="error"):
        if shadows is None:
            raise ValueError("Provide at least one shadow strategy")
        if market_data is not None:
            if any(frame is not None for frame in (prices, benchmark, membership)):
                raise ValueError("Pass either market_data or explicit frames, not both")
        elif prices is None or benchmark is None or membership is None:
            raise ValueError("Provide prices, benchmark, membership, or market_data")
        specs = [s if isinstance(s, ShadowSpec) else ShadowSpec(s) for s in shadows]
        names = [s.strategy.strategy_id for s in specs]
        if not names or len(set(names)) != len(names) or strategy_id in names:
            raise ValueError("Use nonempty unique shadow IDs distinct from the portfolio ID")
        common = dict(market_data=market_data, fundamentals=fundamentals, fee_bps=fee_bps,
                      max_stale_sessions=max_stale_sessions, delisting=delisting)
        self._shadows = {
            spec.strategy.strategy_id: Backtester(
                prices, benchmark, membership, deepcopy(spec.strategy),
                training_panel=spec.training_panel, feature_columns=spec.feature_columns,
                risk_manager=spec.risk_manager, initial_cash=shadow_initial_cash, **common)
            for spec in specs
        }
        adapter = PortfolioStrategy(strategy_id, optimizer if optimizer is not None else EqualWeightOptimizer())
        self._main = Backtester(prices, benchmark, membership, adapter, initial_cash=initial_cash,
                                risk_manager=risk_manager, **common)
        self.phase = Phase.READY

    def run(self, start, end, *, shadow_start=None, allow_partial=False):
        if self.phase != Phase.READY:
            raise ValueError("Create a new shadow coordinator for each run")
        shadow_start = start if shadow_start is None else shadow_start
        if pd.Timestamp(shadow_start) > pd.Timestamp(start):
            raise ValueError("Shadow history must begin no later than the main run")
        self.phase = Phase.DECIDING
        try:
            self._main.begin(start, end, allow_partial=allow_partial)
            for shadow in self._shadows.values():
                shadow.begin(shadow_start, end, allow_partial=allow_partial)
            while next(iter(self._shadows.values())).next_session is not None:
                day = next(iter(self._shadows.values())).next_session
                contexts = {}
                # No account may consume this day's quotes until every strategy
                # and the allocator have made their decisions at the prior close.
                for name, shadow in self._shadows.items():
                    if shadow.next_session != day:
                        raise ValueError("Shadow calendars diverged")
                    contexts[name] = shadow.prepare()
                if self._main.next_session == day:
                    cutoff = next(iter(contexts.values())).signal_date
                    if any(c.signal_date != cutoff for c in contexts.values()):
                        raise ValueError("Shadow signal cutoffs diverged")
                    navs = pd.concat({name: shadow.history_so_far.nav
                                      for name, shadow in self._shadows.items()}, axis=1)
                    self._main.strategy.observe(
                        cutoff, navs,
                        {name: shadow.pending_proposals for name, shadow in self._shadows.items()},
                        {name: shadow.model_version for name, shadow in self._shadows.items()})
                    self._main.prepare()
                for shadow in self._shadows.values():
                    shadow.advance()
                if self._main.next_session == day:
                    self._main.advance()
            portfolio = self._main.finish()
            shadows = {name: shadow.finish() for name, shadow in self._shadows.items()}
            # Attribution must succeed before the coordinator reports COMPLETE;
            # a late error must leave the run FAILED rather than half-published.
            adapter = self._main.strategy
            contributions = pd.DataFrame(adapter.contributions, columns=[
                "signal_date", "execution_date", "component_id", "symbol", "budget", "model_version"])
            approved = contributions.rename(columns={"budget": "proposed_budget"}).copy()
            approved["approved_budget"] = 0.0
            if not approved.empty:
                totals = contributions.groupby(["execution_date", "symbol"]).budget.sum().to_dict()
                buys = portfolio.orders.loc[
                    portfolio.orders.side.eq("BUY")
                    & portfolio.orders.status.eq("filled")]
                budgets = buys.set_index(["execution_date", "symbol"]).budget.to_dict()
                approved["approved_budget"] = [
                    (row.proposed_budget / totals[row.execution_date, row.symbol])
                    * budgets.get((row.execution_date, row.symbol), 0.0)
                    for row in approved.itertuples()]
            result = PortfolioBacktestResult(portfolio, shadows, pd.DataFrame(adapter.allocations),
                                             contributions, approved)
        except BaseException:
            self.phase = Phase.FAILED
            raise
        self.phase = Phase.COMPLETE
        return result
