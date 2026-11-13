"""FF5 regression signal and two long-only trading strategies.

``FactorTiltStrategy`` is the tactical strategy. It estimates each stock's FF5
loadings from matured 90-calendar-day returns, ranks the cross-section by the
predicted return, and invests only in the top quintile. The bottom quintile is
never purchased. Existing lots are never sold early: the engine exits them at
their 90-day maturity, so the portfolio evolves as overlapping monthly
cohorts.

``CapWeightedMarketStrategy`` is the baseline. It holds the eligible universe at
market-cap weights and rolls each maturing cohort back into the market.

Both strategies use the same point-in-time eligible cross-section produced by
``FamaFrench5``. The factor result is treated as read-only reference data; the
engine's refit deep copy shares it rather than copying the full panel.
"""

from copy import deepcopy
import numbers

import numpy as np
import pandas as pd

from backtesting import DecisionContext, FactorResult, Strategy


def _month_ends(index):
    frame = pd.DataFrame({"date": pd.DatetimeIndex(index)})
    return pd.DatetimeIndex(
        frame.groupby(frame.date.dt.to_period("M")).date.max().to_numpy())


def _cross_sections(characteristics):
    """Index eligible rows once; O(1) lookup by formation date."""
    eligible = characteristics.loc[characteristics.eligible]
    return {date: group.set_index("symbol")
            for date, group in eligible.groupby("date")}


def _scale_to_cash(budgets, cash, min_notional=1.0):
    """Scale positive budgets below available cash, leaving a tiny buffer.

    The OMS checks each execution against total cash without a tolerance, while
    order submission allows a 1e-8 rounding tolerance per order. The buffer
    prevents the final order in a large basket from tripping that exact check.
    Orders below ``min_notional`` are dropped so a few ulps of residual cash do
    not create a full-basket dribble order every session.
    """
    budgets = {symbol: float(value) for symbol, value in budgets.items() if value > 0}
    usable = max(0.0, cash - max(1e-6, abs(cash) * 1e-9))
    if not budgets or usable <= 0:
        return {}
    total = sum(budgets.values())
    if total > usable:
        factor = usable / total
        budgets = {symbol: value * factor for symbol, value in budgets.items()}
    while sum(budgets.values()) > usable:
        largest = max(budgets, key=budgets.get)
        budgets[largest] = np.nextafter(budgets[largest], 0.0)
    return {symbol: value for symbol, value in budgets.items()
            if value >= min_notional}


class CapWeightedMarketStrategy(Strategy):
    """Long-only cap-weighted eligible universe, rolled monthly."""

    def __init__(self, factor_result: FactorResult, strategy_id="cap_weighted_market",
                 min_order_notional=1.0):
        super().__init__(strategy_id)
        if not isinstance(factor_result, FactorResult):
            raise TypeError("factor_result must be a FactorResult")
        characteristics = factor_result.characteristics
        required = {"date", "symbol", "ME", "eligible"}
        missing = required - set(characteristics.columns)
        if missing:
            raise ValueError(
                "factor_result.characteristics is missing: " + ", ".join(sorted(missing)))
        if (isinstance(min_order_notional, bool)
                or not isinstance(min_order_notional, numbers.Real)
                or not np.isfinite(min_order_notional)
                or min_order_notional < 0):
            raise ValueError("min_order_notional must be finite and nonnegative")
        self.factor_result = factor_result
        self.characteristics = characteristics
        self.min_order_notional = float(min_order_notional)
        self._cross_sections = _cross_sections(self.characteristics)
        self._eligible_dates = pd.DatetimeIndex(sorted(self._cross_sections))

    def __deepcopy__(self, memo):
        clone = type(self).__new__(type(self))
        memo[id(self)] = clone
        shared = {"factor_result", "characteristics", "_cross_sections", "_eligible_dates"}
        for key, value in self.__dict__.items():
            setattr(clone, key, value if key in shared else deepcopy(value, memo))
        return clone

    def propose_buys(self, context: DecisionContext):
        if context.portfolio.available_cash <= 0:
            return {}
        month_end = context.signal_date.month != context.execution_date.month
        date = context.signal_date if month_end else self._latest_date(context)
        cross = self._eligible(context, date)
        if cross.empty:
            return {}
        weights = cross.ME / cross.ME.sum()
        target = weights * context.portfolio.nav
        current = pd.Series(context.portfolio.holdings, dtype=float).reindex(
            weights.index, fill_value=0.0)
        gaps = (target - current).clip(lower=0.0)
        return _scale_to_cash(gaps.to_dict(), context.portfolio.available_cash,
                              min_notional=self.min_order_notional)

    def _latest_date(self, context):
        if self._eligible_dates.empty:
            return None
        position = self._eligible_dates.searchsorted(
            pd.Timestamp(context.signal_date), side="right") - 1
        return self._eligible_dates[position] if position >= 0 else None

    def _eligible(self, context, date):
        if date is None:
            return pd.DataFrame()
        cross = self._cross_sections.get(pd.Timestamp(date))
        if cross is None:
            return pd.DataFrame()
        allowed = set(context.membership.symbol) & set(context.prices.columns)
        return cross.loc[cross.index.isin(allowed) & cross.ME.gt(0)]


class FactorTiltStrategy(Strategy):
    """FF5 regression tilt: overweight the top quintile, exclude the bottom.

    The middle bucket remains at cap weight; the bottom bucket is never
    purchased. Existing lots are held to maturity.

    Parameters
    ----------
    factor_result:
        Precomputed ``FamaFrench5`` output. Only rows on or before the signal
        date are read.
    window, min_obs:
        Rolling regression window in monthly observations and the minimum
        number of matured 90-day returns required for a stock.
    top_q, bottom_q:
        Cross-sectional quintile fractions. The default is the top and bottom
        20%.
    lookback_months:
        Number of recent 90-day factor returns used for the expected factor
        return in the prediction.
    ramp_months:
        Start-up schedule. The first ``ramp_months`` monthly decisions invest
        ``NAV / ramp_months`` in the cap-weighted market. After the ramp, all
        available cash is allocated to the top quintile. The engine's 90-day
        lots then create the staggered sleeves automatically.
    """

    self_fitting = True

    def __init__(self, factor_result: FactorResult, *, window=60, min_obs=24,
                 top_q=0.2, bottom_q=0.2, lookback_months=12, ramp_months=3,
                 min_order_notional=1.0, strategy_id="ff5_tilt"):
        super().__init__(strategy_id)
        if not isinstance(factor_result, FactorResult):
            raise TypeError("factor_result must be a FactorResult")
        characteristics = factor_result.characteristics
        required = {"date", "symbol", "ME", "eligible"}
        missing = required - set(characteristics.columns)
        if missing:
            raise ValueError(
                "factor_result.characteristics is missing: " + ", ".join(sorted(missing)))
        if not isinstance(window, int) or window < min_obs or min_obs < 6:
            raise ValueError("window and min_obs must be increasing integers >= 6")
        if not 0 < top_q < 1 or not 0 < bottom_q < 1 or top_q + bottom_q >= 1:
            raise ValueError("top_q and bottom_q must be positive and leave a middle bucket")
        if not isinstance(lookback_months, int) or lookback_months < 1:
            raise ValueError("lookback_months must be a positive integer")
        if not isinstance(ramp_months, int) or ramp_months < 0:
            raise ValueError("ramp_months must be a nonnegative integer")
        if (isinstance(min_order_notional, bool)
                or not isinstance(min_order_notional, numbers.Real)
                or not np.isfinite(min_order_notional)
                or min_order_notional < 0):
            raise ValueError("min_order_notional must be finite and nonnegative")
        self.factor_result = factor_result
        self.factor_returns = factor_result.factor_returns.sort_index()
        self.characteristics = characteristics
        self.min_order_notional = float(min_order_notional)
        self._cross_sections = _cross_sections(self.characteristics)
        self._eligible_dates = pd.DatetimeIndex(sorted(self._cross_sections))
        self.window, self.min_obs = window, min_obs
        self.top_q, self.bottom_q = top_q, bottom_q
        self.lookback_months = lookback_months
        self.ramp_months = ramp_months
        self._ramp_remaining = ramp_months
        self.scores = pd.Series(dtype=float)
        self._fit_date = None
        self._observations = pd.Series(dtype=int)

    def __deepcopy__(self, memo):
        """Share the large read-only factor panels; copy only mutable state."""
        clone = type(self).__new__(type(self))
        memo[id(self)] = clone
        shared = {"factor_result", "factor_returns", "characteristics",
                  "_cross_sections", "_eligible_dates"}
        for key, value in self.__dict__.items():
            setattr(clone, key, value if key in shared else deepcopy(value, memo))
        return clone

    def refit_settings(self):
        return dict(window=self.window, min_obs=self.min_obs, top_q=self.top_q,
                    bottom_q=self.bottom_q, lookback_months=self.lookback_months,
                    factors=list(self.factor_returns.columns),
                    target="90-calendar-day return")

    def needs_refit(self, context):
        return context.signal_date.month != context.execution_date.month

    def refit(self, data, context):
        """Fit the FF5 regression on matured 90-day observations only."""
        factor_columns = ["MKT", "SMB", "HML", "RMW", "CMA"]
        raw_factors = self.factor_returns
        if (not isinstance(raw_factors, pd.DataFrame) or raw_factors.empty
                or not set(factor_columns).issubset(raw_factors.columns)):
            raise ValueError(
                "factor_returns must be a nonempty DataFrame with MKT, SMB, HML, RMW, CMA")
        if (not isinstance(raw_factors.index, pd.DatetimeIndex)
                or raw_factors.index.has_duplicates
                or not raw_factors.index.is_monotonic_increasing):
            raise ValueError("factor_returns needs a sorted, unique DatetimeIndex")
        signal = pd.Timestamp(context.signal_date)
        prices = context.prices
        if prices.empty or prices.shape[0] < 2:
            self.scores = pd.Series(dtype=float)
            self._observations = pd.Series(dtype=int)
            self._fit_date = signal
            return
        month_ends = _month_ends(prices.index)
        index = prices.index

        # Matured 90-calendar-day stock returns. Entry is the session after the
        # month-end signal; the exit is the first quoted close on/after 90 days.
        forward = {}
        for date in month_ends:
            entry_pos = index.get_loc(date) + 1
            if entry_pos >= len(index):
                continue
            exit_pos = index.searchsorted(index[entry_pos] + pd.Timedelta(days=90),
                                          side="left")
            if exit_pos >= len(index) or index[exit_pos] >= signal:
                continue
            forward[date] = (prices.iloc[exit_pos].div(prices.iloc[entry_pos]).sub(1.0))
        if not forward:
            self.scores = pd.Series(dtype=float)
            self._observations = pd.Series(dtype=int)
            self._fit_date = signal
            return
        returns = pd.DataFrame(forward).T.sort_index()

        # Three-month compounded factor return aligned to the same formation.
        calendar_ends = _month_ends(prices.index)
        factors = raw_factors[factor_columns].reindex(index=calendar_ends)
        if factors.dropna(how="all").empty:
            raise ValueError("factor_returns does not cover the signal calendar")
        gross = (1.0 + factors).rolling(3).apply(np.prod, raw=True)
        factor_90 = gross.shift(-2).sub(1.0).reindex(returns.index)
        # factor_90[t] needs factor returns through t+2. At the signal close
        # only t+2 < signal is known, so stop at the month-end three months
        # before the signal. Do not include the signal row itself.
        known_cutoff = (signal - pd.DateOffset(months=3)) + pd.offsets.MonthEnd(0)
        expected = factor_90.loc[:known_cutoff].tail(self.lookback_months).mean()
        expected = expected.reindex(factor_columns).fillna(0.0)

        universe = self._cross_sections.get(signal)
        if universe is None:
            raise ValueError(f"No factor cross-section at {signal.date()}")
        returns = returns.loc[:, returns.columns.isin(universe.index)]

        factor_matrix = factor_90.to_numpy(dtype=float)
        complete_factor = np.isfinite(factor_matrix).all(axis=1)
        expected_array = expected.to_numpy(dtype=float)
        window_start = signal - pd.DateOffset(months=self.window)
        scores, observations = {}, {}
        for column, symbol in enumerate(returns.columns):
            y = returns.iloc[:, column].to_numpy(dtype=float)
            valid = np.flatnonzero(np.isfinite(y) & complete_factor)
            # Keep the regression inside a true calendar window, not merely
            # the last N non-missing rows.
            valid = valid[factor_90.index[valid] >= window_start]
            if len(valid) < self.min_obs:
                continue
            positions = valid[-self.window:]
            x = factor_matrix[positions]
            values = y[positions]
            design = np.column_stack([np.ones(len(x)), x])
            if np.linalg.matrix_rank(design) < design.shape[1]:
                continue
            coefficients, *_ = np.linalg.lstsq(design, values, rcond=None)
            scores[symbol] = float(coefficients[0] + coefficients[1:] @ expected_array)
            observations[symbol] = len(values)

        self.scores = pd.Series(scores, dtype=float)
        self._observations = pd.Series(observations, dtype=int)
        self._fit_date = signal

    def propose_buys(self, context):
        month_end = context.signal_date.month != context.execution_date.month
        if self._ramp_remaining > 0:
            # During the ramp only the month-end sleeve purchase is allowed;
            # do not reinvest the intentionally uninvested cash mid-month.
            if not month_end:
                return {}
            cross = self._eligible(context, context.signal_date)
            if cross.empty or context.portfolio.available_cash <= 0:
                return {}
            budget = min(context.portfolio.available_cash,
                         context.portfolio.nav / max(1, self.ramp_months))
            weights = cross.ME / cross.ME.sum()
            self._ramp_remaining -= 1
            return _scale_to_cash(
                {symbol: float(budget * weight) for symbol, weight in weights.items()},
                context.portfolio.available_cash,
                min_notional=self.min_order_notional)
        if context.portfolio.available_cash <= 0:
            return {}
        # Month-end uses the fresh signal. Mid-month reinvests maturity
        # proceeds at the next session using the most recent fitted signal.
        date = context.signal_date if month_end else self._fit_date
        if date is None:
            return {}
        return self._invest(context, date)

    def _invest(self, context, date):
        cross = self._eligible(context, date)
        if cross.empty:
            return {}
        scores = self.scores.reindex(cross.index).dropna()
        if scores.empty:
            return {}
        cross = cross.loc[scores.index]
        weights = cross.ME / cross.ME.sum()

        # Count-based quintiles avoid quantile ties inflating a bucket. If the
        # cross-section is too small or has no score dispersion, fall back to
        # cap weights rather than sitting in cash.
        count = len(scores)
        n_bottom = max(1, int(np.floor(self.bottom_q * count)))
        n_top = max(1, int(np.floor(self.top_q * count)))
        if count < 2 or n_bottom + n_top >= count or scores.nunique() < 2:
            target_weights = weights
        else:
            ordered = scores.sort_values(kind="stable")
            bottom = ordered.index[:n_bottom]
            top = ordered.index[-n_top:]
            target_weights = weights.copy()
            bottom_pool = target_weights.loc[bottom].sum()
            target_weights.loc[bottom] = 0.0
            top_base = target_weights.loc[top].sum()
            if top_base <= 0:
                target_weights = weights
            else:
                target_weights.loc[top] += (bottom_pool
                                            * target_weights.loc[top] / top_base)
                target_weights = target_weights / target_weights.sum()
        if not np.isfinite(target_weights).all():
            return {}

        target = target_weights * context.portfolio.nav
        current = pd.Series(context.portfolio.holdings, dtype=float).reindex(
            target_weights.index, fill_value=0.0)
        gaps = (target - current).clip(lower=0.0)
        return _scale_to_cash(gaps.to_dict(), context.portfolio.available_cash,
                              min_notional=self.min_order_notional)

    def _eligible(self, context, date):
        if date is None:
            return pd.DataFrame()
        cross = self._cross_sections.get(pd.Timestamp(date))
        if cross is None:
            return pd.DataFrame()
        allowed = set(context.membership.symbol) & set(context.prices.columns)
        return cross.loc[cross.index.isin(allowed) & cross.ME.gt(0)]
