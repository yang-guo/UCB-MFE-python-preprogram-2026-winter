"""Pairs_Tilt: long-only relative-value sleeves with a market fallback.

The strategy is deliberately long-only and 90-day-lot compatible.  A
cointegrated pair contributes only its cheap leg; the expensive leg is not
purchased.  This is a relative-value tilt, not a market-neutral short book.

Capital is organised as nine ten-calendar-day sleeves.  The first nine scheduled
decisions put one ninth of NAV into the cap-weighted eligible universe.  After
that, each maturing sleeve is reinvested at the next session into the strongest
eligible pair(s), with unused capital buying the cap-weighted market.

Pair rules:
* one pair per stock per sleeve;
* equal capital per pair, sized as ``sleeve cash / max_pairs``;
* greedy selection by FDR-adjusted ADF p-value then absolute spread z-score;
* at most ``max_pairs`` pairs per sleeve;
* if no pair is eligible, the whole sleeve buys the market.
"""

from __future__ import annotations

from bisect import bisect_right
from copy import deepcopy
import numbers

import numpy as np
import pandas as pd

from backtesting import DecisionContext, FactorResult, Strategy
from research.pairs import PairsResult, point_in_time_eligible


def _cross_sections(characteristics):
    mask = point_in_time_eligible(characteristics)
    eligible = characteristics.loc[mask]
    return {date: group.set_index("symbol")
            for date, group in eligible.groupby("date")}


def _latest_on_or_before(dates, signal):
    """Newest entry of a sorted date list at or before ``signal``, or ``None``."""
    index = bisect_right(dates, signal) - 1
    return None if index < 0 else dates[index]


def _scale_to_cash(budgets, cash, min_notional=1.0):
    if not np.isfinite(cash) or cash <= 0:
        return {}
    budgets = {symbol: float(value) for symbol, value in budgets.items()
               if np.isfinite(value) and value > 0}
    if not budgets:
        return {}
    total = sum(budgets.values())
    usable = (cash if total <= cash
              else max(0.0, cash - max(1e-6, abs(cash) * 1e-9)))
    if usable <= 0:
        return {}
    if total > usable:
        factor = usable / total
        scaled = {symbol: value * factor for symbol, value in budgets.items()}
        if min_notional <= 0 or all(value >= min_notional
                                     for value in scaled.values()):
            budgets = scaled
        else:
            # Scaling would push a slice below the floor.  Drop the smallest
            # original slices until every remaining scaled slice clears it.
            while budgets:
                factor = usable / sum(budgets.values())
                scaled = {symbol: value * factor
                          for symbol, value in budgets.items()}
                if all(value >= min_notional for value in scaled.values()):
                    budgets = scaled
                    break
                smallest = min(budgets, key=budgets.get)
                budgets.pop(smallest)
            else:
                return {}
    while sum(budgets.values()) > usable:
        largest = max(budgets, key=budgets.get)
        budgets[largest] = np.nextafter(budgets[largest], 0.0)
    return {symbol: value for symbol, value in budgets.items()
            if value >= min_notional}


class PairsTiltStrategy(Strategy):
    """Long-only pair-tilt strategy named ``Pairs_Tilt``."""

    required_financials = ()

    def __init__(self, pairs_result: PairsResult, factor_result: FactorResult, *,
                 max_pairs=3, sleeve_days=10, ramp_sleeves=9,
                 max_signal_age_days=45, min_order_notional=1.0,
                 strategy_id="Pairs_Tilt"):
        super().__init__(strategy_id)
        if not isinstance(pairs_result, PairsResult):
            raise TypeError("pairs_result must be a PairsResult")
        if not isinstance(factor_result, FactorResult):
            raise TypeError("factor_result must be a FactorResult")
        if (isinstance(max_pairs, bool) or not isinstance(max_pairs, numbers.Integral)
                or max_pairs < 1):
            raise ValueError("max_pairs must be a positive integer")
        if (isinstance(sleeve_days, bool)
                or not isinstance(sleeve_days, numbers.Integral)
                or sleeve_days < 1):
            raise ValueError("sleeve_days must be a positive integer")
        if (isinstance(ramp_sleeves, bool)
                or not isinstance(ramp_sleeves, numbers.Integral)
                or ramp_sleeves < 1):
            raise ValueError("ramp_sleeves must be a positive integer")
        if (isinstance(max_signal_age_days, bool)
                or not isinstance(max_signal_age_days, numbers.Integral)
                or max_signal_age_days < 0):
            raise ValueError("max_signal_age_days must be a nonnegative integer")
        if (isinstance(min_order_notional, bool)
                or not isinstance(min_order_notional, numbers.Real)
                or not np.isfinite(min_order_notional)
                or min_order_notional < 0):
            raise ValueError("min_order_notional must be finite and nonnegative")
        required_chars = {"date", "symbol", "ME", "eligible"}
        missing = required_chars - set(factor_result.characteristics.columns)
        if missing:
            raise ValueError("factor_result.characteristics is missing: "
                             + ", ".join(sorted(missing)))
        required_signals = {"date", "leg_a", "leg_b", "long_symbol",
                            "adf_p_adj", "zscore"}
        if not pairs_result.signals.empty:
            missing = required_signals - set(pairs_result.signals.columns)
            if missing:
                raise ValueError("pairs_result.signals is missing: "
                                 + ", ".join(sorted(missing)))
            if not pd.api.types.is_datetime64_any_dtype(pairs_result.signals.date):
                raise ValueError("pairs_result.signals.date must be datetime")
            if pairs_result.signals.date.dt.tz is not None:
                raise ValueError("pairs_result.signals.date must be timezone-naive")
            for column in ("zscore", "adf_p_adj"):
                if not pd.api.types.is_numeric_dtype(pairs_result.signals[column]):
                    raise ValueError("pairs_result.signals." + column
                                     + " must be numeric")
        if not pd.api.types.is_datetime64_any_dtype(factor_result.characteristics.date):
            raise ValueError("factor_result.characteristics.date must be datetime")
        if factor_result.characteristics.date.dt.tz is not None:
            raise ValueError("factor_result.characteristics.date must be timezone-naive")
        if not pd.api.types.is_numeric_dtype(factor_result.characteristics.ME):
            raise ValueError("factor_result.characteristics.ME must be numeric")
        self.pairs_result = pairs_result
        self.factor_result = factor_result
        self.characteristics = factor_result.characteristics
        self._cross_sections = _cross_sections(self.characteristics)
        # Read-only lookups built once.  Sessions only read them, so a decision
        # costs a binary search plus a cheap dynamic mask instead of a scan.
        self._cross_dates = sorted(self._cross_sections)
        self._me_by_date = {date: cross["ME"]
                            for date, cross in self._cross_sections.items()}
        signals = pairs_result.signals
        self._signals_by_date = {}
        if not signals.empty:
            self._signals_by_date = {
                pd.Timestamp(date): group
                for date, group in signals.groupby("date", sort=True)}
        self._signal_dates = sorted(self._signals_by_date)
        self.max_pairs = max_pairs
        self.sleeve_days = sleeve_days
        self.ramp_sleeves = ramp_sleeves
        self.max_signal_age_days = max_signal_age_days
        self.min_order_notional = float(min_order_notional)
        self._next_date = None
        self._ramp_remaining = ramp_sleeves

    def __deepcopy__(self, memo):
        clone = type(self).__new__(type(self))
        memo[id(self)] = clone
        shared = {"pairs_result", "factor_result", "characteristics",
                  "_cross_sections", "_cross_dates", "_me_by_date",
                  "_signals_by_date", "_signal_dates"}
        for key, value in self.__dict__.items():
            setattr(clone, key, value if key in shared else deepcopy(value, memo))
        return clone

    def propose_buys(self, context: DecisionContext):
        if self._next_date is None:
            self._next_date = pd.Timestamp(context.signal_date)
        signal = pd.Timestamp(context.signal_date)
        scheduled = signal >= self._next_date
        cash = context.portfolio.available_cash
        if cash <= 0:
            if scheduled:
                self._advance(signal)
            return {}

        if self._ramp_remaining > 0:
            if not scheduled:
                return {}
            budget = min(cash, max(context.portfolio.nav / self.ramp_sleeves,
                                   self.min_order_notional))
            budgets = self._market_budgets(context, budget)
            scaled = _scale_to_cash(budgets, cash, self.min_order_notional)
            # Do not burn a sleeve when no order can be formed; retry after
            # the missing cross-section or quote appears.
            if scaled:
                self._ramp_remaining -= 1
                self._advance(signal)
            return scaled

        # After the ramp, maturity proceeds are reinvested on the next session.
        if not scheduled and cash < max(1.0, 0.005 * context.portfolio.nav):
            return {}
        # One event deploys at most one sleeve; residual cash can be handled
        # on the next session instead of lumping every idle dollar at once.
        budget = min(cash, context.portfolio.nav / self.ramp_sleeves)
        budgets = self._pair_budgets(context, budget)
        scaled = _scale_to_cash(budgets, cash, self.min_order_notional)
        if scheduled and scaled:
            self._advance(signal)
        return scaled

    def _advance(self, signal):
        self._next_date = signal + pd.Timedelta(days=self.sleeve_days)

    def _market_weights(self, context):
        date = _latest_on_or_before(self._cross_dates,
                                    pd.Timestamp(context.signal_date))
        if date is None:
            return pd.Series(dtype=float)
        weights = self._me_by_date[date]
        last = context.prices.iloc[-1]
        allowed = {symbol for symbol in set(context.membership.symbol)
                   & set(context.prices.columns)
                   if np.isfinite(last.get(symbol, np.nan))
                   and last.get(symbol, 0.0) > 0}
        # The cached ME series is not filtered: membership and quotes change
        # every session, so only the static date lookup is reused.
        weights = weights.loc[weights.gt(0) & weights.index.isin(allowed)]
        if weights.empty:
            return pd.Series(dtype=float)
        return weights / weights.sum()

    def _market_budgets(self, context, budget):
        weights = self._market_weights(context)
        if weights.empty or budget <= 0:
            return {}
        if self.min_order_notional > 0:
            max_names = max(1, int(budget // self.min_order_notional))
            if len(weights) > max_names:
                weights = weights.nlargest(max_names)
                weights = weights / weights.sum()
        return {symbol: float(budget * weight)
                for symbol, weight in weights.items() if weight > 0}

    def _pair_budgets(self, context, cash):
        signal = pd.Timestamp(context.signal_date)
        # Future signals are never used, so the newest date at or before the
        # decision date is the only candidate; missing dates fall back.
        latest = _latest_on_or_before(self._signal_dates, signal)
        if latest is None:
            return self._market_budgets(context, cash)
        if (signal - latest).days > self.max_signal_age_days:
            return self._market_budgets(context, cash)

        last = context.prices.iloc[-1]
        allowed = {symbol for symbol in set(context.membership.symbol)
                   & set(context.prices.columns)
                   if np.isfinite(last.get(symbol, np.nan))
                   and last.get(symbol, 0.0) > 0}
        candidates = self._signals_by_date[latest]
        candidates = candidates.loc[candidates.long_symbol.isin(allowed)
                                    & candidates.leg_a.isin(allowed)
                                    & candidates.leg_b.isin(allowed)].copy()
        candidates = candidates.sort_values(
            ["adf_p_adj", "zscore"], ascending=[True, False], key=lambda col: (
                col.abs() if col.name == "zscore" else col), kind="stable")
        selected, used = [], set()
        for row in candidates.itertuples():
            if row.leg_a == row.leg_b or row.long_symbol not in (row.leg_a, row.leg_b):
                continue
            if (not np.isfinite(row.adf_p_adj)) or (not np.isfinite(row.zscore)):
                continue
            if row.leg_a in used or row.leg_b in used:
                continue
            selected.append(row)
            used.update((row.leg_a, row.leg_b, row.long_symbol))
            if len(selected) >= self.max_pairs:
                break

        if not selected:
            return self._market_budgets(context, cash)

        pair_capital = cash / self.max_pairs
        if pair_capital < self.min_order_notional:
            return self._market_budgets(context, cash)
        budgets = {}
        for row in selected:
            budgets[row.long_symbol] = budgets.get(row.long_symbol, 0.0) + pair_capital
        remaining = max(0.0, cash - pair_capital * len(selected))
        if remaining > 0:
            weights = self._market_weights(context)
            weights = weights.loc[~weights.index.isin(used)]
            if not weights.empty:
                weights = weights / weights.sum()
                for symbol, weight in weights.items():
                    budgets[symbol] = budgets.get(symbol, 0.0) + remaining * weight
        return budgets
