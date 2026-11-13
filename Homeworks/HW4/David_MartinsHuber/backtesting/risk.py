"""Signal-close purchase controls. Existing 90-day lots are never liquidated."""

from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
import numbers
from typing import Mapping

import numpy as np
import pandas as pd

from .contracts import Portfolio, plain_amount


AUDIT_COLUMNS = ["signal_date", "execution_date", "strategy_id", "model_version",
                 "policy", "stage", "symbol", "budget_before", "budget_after", "reason", "metrics"]

# Drawdown ratios are dimensionless, so an absolute tolerance of a few ulps
# absorbs the binary rounding in 1 - nav/peak. Without it an exact 10% loss
# computes as 0.09999999999999998 and silently misses a 10% gate.
_RATIO_TOLERANCE = 8 * np.finfo(float).eps


@dataclass(frozen=True)
class RiskContext:
    signal_date: pd.Timestamp
    execution_date: pd.Timestamp
    portfolio: Portfolio
    history: pd.DataFrame
    fee_rate: float
    strategy_id: str = ""
    model_version: int = 0


@dataclass(frozen=True)
class RiskAdjustment:
    budgets: Mapping[str, float]
    reason: str
    metrics: Mapping
    symbol_reasons: Mapping[str, str] | None = None
    symbol_metrics: Mapping[str, Mapping] | None = None


@dataclass(frozen=True)
class RiskDecision:
    original_budgets: Mapping[str, float]
    approved_budgets: Mapping[str, float]
    audit: pd.DataFrame


class RiskPolicy(ABC):
    """A policy may only reduce existing proposals. No independent data reads."""

    priority = 10

    @abstractmethod
    def apply(self, context: RiskContext, budgets: Mapping[str, float]) -> RiskAdjustment:
        pass


class DrawdownGate(RiskPolicy):
    """Latch an entry reduction at threshold; optional recovery hysteresis.

    A None resume_drawdown keeps the gate latched for the rest of the run.
    History starts with the main account's initial capital, not shadow warm-up.
    """

    priority = 10

    def __init__(self, threshold=0.15, scale=0.0, resume_drawdown=None):
        try:
            threshold, scale = plain_amount(threshold), plain_amount(scale)
            if resume_drawdown is not None:
                resume_drawdown = plain_amount(resume_drawdown)
        except ValueError:
            raise ValueError("Invalid drawdown threshold or scaling factor") from None
        if not 0 < threshold <= 1 or not 0 <= scale <= 1:
            raise ValueError("Invalid drawdown threshold or scaling factor")
        if resume_drawdown is not None and not 0 <= resume_drawdown < threshold:
            raise ValueError("Recovery drawdown must be below the activation threshold")
        self.threshold, self.scale, self.resume_drawdown = threshold, scale, resume_drawdown
        self.active = False

    def apply(self, context, budgets):
        peak = context.history.nav.max()
        drawdown = max(0.0, 1 - context.portfolio.nav / peak)
        # Compare with a few ulps of tolerance so "reaches the threshold" and
        # "recovers to the resume level" include exact equality.
        if drawdown >= self.threshold - _RATIO_TOLERANCE:
            self.active = True
        elif (self.active and self.resume_drawdown is not None
              and drawdown <= self.resume_drawdown + _RATIO_TOLERANCE):
            self.active = False
        factor = self.scale if self.active else 1.0
        return RiskAdjustment({s: b * factor for s, b in budgets.items()},
                              "drawdown_restriction" if self.active else "drawdown_clear",
                              dict(high_water_mark=peak, drawdown=drawdown, active=self.active,
                                   scale=factor, threshold=self.threshold, resume_drawdown=self.resume_drawdown))


class VolatilityTarget(RiskPolicy):
    """Downscale new buys using trailing realized main-account volatility.

    This is not projected covariance risk targeting and cannot resize locked
    positions. Insufficient history follows an explicit allow/block policy.
    Zero realized volatility passes proposals without leverage.
    """

    priority = 20

    def __init__(self, annual_target=0.10, lookback=63, min_observations=20,
                 sessions_per_year=252, warmup="allow"):
        try:
            annual_target = plain_amount(annual_target)
        except ValueError:
            raise ValueError("Volatility target must be finite and positive") from None
        if annual_target <= 0:
            raise ValueError("Volatility target must be finite and positive")
        if (isinstance(lookback, bool) or not isinstance(lookback, numbers.Integral)
                or isinstance(min_observations, bool)
                or not isinstance(min_observations, numbers.Integral)
                or isinstance(sessions_per_year, bool)
                or not isinstance(sessions_per_year, numbers.Integral)):
            raise ValueError("Invalid volatility window or annualization frequency")
        # Convert before comparing so an int subclass cannot lie in __le__.
        lookback, min_observations = int(lookback), int(min_observations)
        sessions_per_year = int(sessions_per_year)
        if not 2 <= min_observations <= lookback or not 1 <= sessions_per_year <= 1_000_000:
            raise ValueError("Invalid volatility window or annualization frequency")
        if warmup not in ("allow", "block"):
            raise ValueError("warmup must be 'allow' or 'block'")
        self.annual_target = annual_target
        self.lookback, self.min_observations = lookback, min_observations
        self.sessions_per_year, self.warmup = sessions_per_year, warmup

    def apply(self, context, budgets):
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            returns = context.history.nav.pct_change(fill_method=None).iloc[1:].tail(self.lookback)
        if not np.isfinite(returns.to_numpy()).all():
            raise ValueError("Risk volatility inputs are not finite")
        volatility = np.nan
        if len(returns) < self.min_observations:
            factor = 1.0 if self.warmup == "allow" else 0.0
            reason = f"volatility_warmup_{self.warmup}"
        else:
            with np.errstate(over="ignore", invalid="ignore"):
                volatility = returns.std(ddof=1) * np.sqrt(self.sessions_per_year)
            if not np.isfinite(volatility):
                raise ValueError("Estimated risk volatility is not finite")
            factor = self.annual_target / volatility if volatility > self.annual_target else 1.0
            reason = "volatility_reduction" if factor < 1 else "volatility_clear"
        return RiskAdjustment({s: b * factor for s, b in budgets.items()}, reason,
                              dict(realized_volatility=volatility, annual_target=self.annual_target,
                                   observations=len(returns), scale=factor, warmup=self.warmup))


class SingleNameCap(RiskPolicy):
    """Limit new exposure relative to pre-trade signal-close NAV, before fees."""

    priority = 30

    def __init__(self, cap=0.25):
        try:
            cap = plain_amount(cap)
        except ValueError:
            raise ValueError("Single-name cap must be in (0, 1]") from None
        if not 0 < cap <= 1:
            raise ValueError("Single-name cap must be in (0, 1]")
        self.cap = cap

    def apply(self, context, budgets):
        approved, reasons, details = {}, {}, {}
        for symbol, value in context.portfolio.holdings.items():
            if value / context.portfolio.nav > self.cap:
                reasons[symbol] = "existing_exposure_breach_no_early_sale"
        for symbol, budget in budgets.items():
            value = context.portfolio.holdings.get(symbol, 0.0)
            room = max(0.0, self.cap * context.portfolio.nav - value)
            # Only multiply room by the fee factor if it is actually binding;
            # this avoids an unnecessary overflow at large capital values.
            approved[symbol] = room * (1 + context.fee_rate) if budget / (1 + context.fee_rate) > room else budget
            if approved[symbol] < budget:
                reasons.setdefault(symbol, "single_name_purchase_reduced")
        for symbol in set(budgets) | set(reasons):
            value = context.portfolio.holdings.get(symbol, 0.0)
            details[symbol] = dict(existing_value=value, current_weight=value / context.portfolio.nav,
                                   projected_signal_weight=(value + approved.get(symbol, 0.0)
                                                            / (1 + context.fee_rate)) / context.portfolio.nav)
        return RiskAdjustment(approved, "single_name_cap", dict(cap=self.cap, nav_basis="pre_trade_signal_close"), reasons, details)


class RiskManager:
    """Compose policies, enforce monotone reductions, then apply a cash limit.

    State and audit commit together only after every policy succeeds. Policy
    contexts are detached copies, and previously observed NAV cannot be rewritten.
    """

    def __init__(self, policies=()):
        policies = list(policies)
        if any(not isinstance(p, RiskPolicy) for p in policies):
            raise TypeError("All policies must implement RiskPolicy")
        self.policies = sorted(deepcopy(list(policies)), key=lambda p: p.priority)
        self._cutoff, self._history = None, None
        self._audit = []

    @property
    def audit(self):
        return pd.DataFrame(deepcopy(self._audit), columns=AUDIT_COLUMNS)

    @staticmethod
    def _validate_budgets(budgets, previous=None):
        values = {}
        for symbol, budget in dict(budgets).items():
            if type(symbol) is not str or not symbol:
                raise ValueError("Risk budgets require symbols and finite nonnegative amounts")
            try:
                values[symbol] = plain_amount(budget)
            except ValueError:
                raise ValueError("Risk budgets require symbols and finite nonnegative amounts") from None
        if previous is not None and any(s not in previous or b > previous[s] for s, b in values.items()):
            raise ValueError("A risk policy cannot add symbols or increase budgets")
        if not np.isfinite(sum(values.values())):
            raise ValueError("Total risk budget is not finite")
        return values

    def review(self, context: RiskContext, proposed: Mapping[str, float]) -> RiskDecision:
        signal, execution = pd.Timestamp(context.signal_date), pd.Timestamp(context.execution_date)
        if (pd.isna(signal) or pd.isna(execution)
                or signal.tz is not None or execution.tz is not None
                or signal != signal.normalize() or execution != execution.normalize() or signal >= execution):
            raise ValueError("Risk decisions require a signal preceding execution")
        try:
            history = context.history[["nav"]].copy(deep=True)
        except (TypeError, KeyError, IndexError):
            raise ValueError("Risk history must be positive, finite, and complete through the signal cutoff") from None
        if (not isinstance(history.index, pd.DatetimeIndex) or history.empty
                or history.index.hasnans or history.index.has_duplicates
                or not history.index.is_monotonic_increasing or history.index[-1] != signal):
            raise ValueError("Risk history must be positive, finite, and complete through the signal cutoff")
        try:
            history["nav"] = [plain_amount(value) for value in history.nav]
        except ValueError:
            raise ValueError("Risk history must be positive, finite, and complete through the signal cutoff") from None
        if history.nav.le(0).any():
            raise ValueError("Risk history must be positive, finite, and complete through the signal cutoff")
        portfolio = context.portfolio
        # Require the engine's plain dict. A Mapping subclass can report
        # truthful values() for reconciliation but lie to get() inside a policy.
        if type(portfolio.holdings) is not dict:
            raise ValueError("Risk snapshot must reconcile with the signal-close history")
        try:
            holdings = {symbol: plain_amount(value)
                        for symbol, value in portfolio.holdings.items()}
            nav = plain_amount(portfolio.nav)
            cash = plain_amount(portfolio.cash)
            available_cash = plain_amount(portfolio.available_cash)
            fee_rate = plain_amount(context.fee_rate)
        except ValueError:
            raise ValueError("Risk snapshot must reconcile with the signal-close history") from None
        if (nav <= 0 or cash < 0 or not 0 <= available_cash <= cash + 1e-8
                or not 0 <= fee_rate <= 1
                or any(type(k) is not str or not k for k in holdings)
                or not np.isclose(cash + sum(holdings.values()), nav, rtol=1e-12, atol=1e-8)
                or nav != history.nav.iloc[-1]):
            raise ValueError("Risk snapshot must reconcile with the signal-close history")
        if self._cutoff is not None:
            prior = history.loc[:self._cutoff]
            # Compare values, not dtypes: an int64 history and the equivalent
            # float64 history are the same observations.
            if (signal <= self._cutoff or not prior.index.equals(self._history.index)
                    or not np.array_equal(prior.nav.to_numpy(dtype=float),
                                          self._history.nav.to_numpy(dtype=float))):
                raise ValueError("Risk history cannot be replayed or rewritten")
        original = self._validate_budgets(proposed)
        current = original.copy()
        candidate = deepcopy(self.policies)
        records = []
        clean_context = RiskContext(signal, execution,
                                    Portfolio(cash, available_cash, nav, holdings),
                                    history, fee_rate,
                                    context.strategy_id, context.model_version)

        def record(name, stage, before, after, reason, metrics, reasons=None, details=None):
            reasons = dict(reasons or {})
            symbols = sorted(set(before) | set(after) | set(reasons) | set(details or {})) or [None]
            for symbol in symbols:
                row_metrics = deepcopy(dict(metrics))
                row_metrics.update(deepcopy(dict((details or {}).get(symbol, {}))))
                records.append(dict(signal_date=signal, execution_date=execution,
                                    strategy_id=context.strategy_id, model_version=context.model_version,
                                    policy=name, stage=stage, symbol=symbol,
                                    budget_before=before.get(symbol, 0.0), budget_after=after.get(symbol, 0.0),
                                    reason=reasons.get(symbol, reason), metrics=row_metrics))

        for stage, policy in enumerate(candidate):
            adjustment = policy.apply(deepcopy(clean_context), current.copy())
            if not isinstance(adjustment, RiskAdjustment):
                raise TypeError("RiskPolicy.apply must return RiskAdjustment")
            # Do not use `or {}`: a falsy dict subclass would skip the type
            # check and then be re-read by record(). Validate the exact objects
            # that will be consumed.
            reasons = {} if adjustment.symbol_reasons is None else adjustment.symbol_reasons
            details = {} if adjustment.symbol_metrics is None else adjustment.symbol_metrics
            if type(reasons) is not dict or type(details) is not dict:
                raise ValueError("Risk audit symbols must be exact nonempty strings")
            if (type(adjustment.reason) is not str or not adjustment.reason
                    or type(adjustment.metrics) is not dict
                    or any(type(key) is not str or not key.strip() or type(value) is not str
                           for key, value in reasons.items())
                    or any(type(key) is not str or not key.strip() or type(value) is not dict
                           for key, value in details.items())):
                raise ValueError("Risk audit symbols must be exact nonempty strings")
            # Snapshot before converting budgets: a hostile __float__ or keys()
            # can mutate the live objects after they were validated.
            reason = adjustment.reason
            # Take every shallow snapshot before any deepcopy runs: a hostile
            # __deepcopy__ in one field must not rewrite a sibling field.
            raw_reasons = dict(reasons)
            raw_details = {key: dict(value) for key, value in details.items()}
            raw_metrics = dict(adjustment.metrics)
            reasons = deepcopy(raw_reasons)
            details = deepcopy(raw_details)
            metrics = deepcopy(raw_metrics)
            approved = self._validate_budgets(adjustment.budgets, current)
            record(type(policy).__name__, stage, current, approved, reason,
                   metrics, reasons, details)
            current = approved
        total = sum(current.values())
        factor = min(1.0, available_cash / total) if total else 1.0
        if total > available_cash:
            approved = {s: min(b, (b / total) * available_cash) for s, b in current.items()}
        else:
            approved = current.copy()
        approved = {s: b for s, b in approved.items() if b > 0}
        # Remove representable rounding excess by the actual excess, not one
        # ulp at a time: an equal-weight book of n names can exceed the cash
        # cap by O(n) ulps, which made a fixed small iteration bound fail.
        for _ in range(4 * len(approved) + 64):
            excess = sum(approved.values()) - available_cash
            if excess <= 0:
                break
            largest = max(approved, key=approved.get)
            reduced = approved[largest] - excess
            if reduced <= 0:
                del approved[largest]
            else:
                approved[largest] = float(np.nextafter(reduced, 0.0))
        else:
            raise ValueError("Cash limit rounding did not converge")
        record("CashLimit", len(candidate), current, approved, "cash_reduction" if factor < 1 else "cash_clear",
               dict(available_cash=available_cash, scale=factor))
        self._validate_budgets(approved, original)
        # Prepare the detached audit before committing state, so a copy failure
        # cannot leave the manager half-committed.
        audit = deepcopy(records)
        self.policies, self._cutoff, self._history = candidate, signal, history.copy(deep=True)
        self._audit.extend(audit)
        return RiskDecision(original.copy(), approved.copy(), pd.DataFrame(records, columns=AUDIT_COLUMNS))
