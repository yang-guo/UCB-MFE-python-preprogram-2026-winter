"""Volume-bar HMM market-timing strategy.

The model is the volume-only research design:

* benchmark price + total volume,
* trailing 90-session median volume and a ``5 x median`` bucket,
* 3-state Gaussian HMM on bar log returns,
* walk-forward fitting: the model at bar ``t`` is fitted only on bars before
  ``t``, and the current bar is only scored.

The strategy is a market-timing overlay. When the current state is in
``risk_on_states`` it buys an equal-weight basket of the current eligible
universe; otherwise it stays in cash. Existing lots are never sold early: the
engine's OMS holds every purchase for 90 calendar days and exits at maturity.
"""

from __future__ import annotations

from copy import deepcopy
import math
import numbers

import numpy as np
import pandas as pd

from backtesting import DecisionContext, Strategy
from research.bars import volume_bars
from research.hmm import fit_hmm, infer_state


def _as_series(value, *, name: str, column: str, positive: bool = True) -> pd.Series:
    """Normalize a Series or a date/value DataFrame to a sorted Series."""
    if isinstance(value, pd.DataFrame):
        if "date" not in value.columns or column not in value.columns:
            raise ValueError(f"{name} must contain 'date' and '{column}'")
        frame = value[["date", column]].copy()
        frame["date"] = pd.to_datetime(frame["date"])
        series = frame.set_index("date")[column]
    elif isinstance(value, pd.Series):
        series = value.copy()
        series.index = pd.to_datetime(series.index)
    else:
        raise TypeError(f"{name} must be a pandas Series or DataFrame")
    series = series.astype(float)
    if series.index.tz is not None or series.index.hasnans:
        raise ValueError(f"{name} must use timezone-naive, nonmissing dates")
    series = series[~series.index.duplicated(keep="last")].sort_index()
    if series.empty or not np.isfinite(series.to_numpy()).all():
        raise ValueError(f"{name} must be a nonempty finite series")
    if positive and (series <= 0).any():
        raise ValueError(f"{name} must be strictly positive")
    if not positive and (series < 0).any():
        raise ValueError(f"{name} must be nonnegative")
    return series.rename(name)


def _is_int(value) -> bool:
    """True for a real integer, excluding bool."""
    return isinstance(value, int) and not isinstance(value, bool)


def _market_cap_frame(value) -> pd.DataFrame | None:
    """Normalize a point-in-time membership/market-cap frame."""
    if value is None:
        return None
    if not isinstance(value, pd.DataFrame):
        raise TypeError("market_cap must be a pandas DataFrame or None")
    required = {"symbol", "quarter_start", "selection_date", "market_cap_proxy"}
    missing = required - set(value.columns)
    if missing:
        raise ValueError(
            "market_cap is missing columns: " + ", ".join(sorted(missing))
        )
    frame = value[list(required)].copy()
    frame["quarter_start"] = pd.to_datetime(frame["quarter_start"])
    frame["selection_date"] = pd.to_datetime(frame["selection_date"])
    frame["market_cap_proxy"] = pd.to_numeric(frame["market_cap_proxy"], errors="coerce")
    frame = frame.dropna(subset=["symbol", "quarter_start", "selection_date", "market_cap_proxy"])
    frame = frame.loc[np.isfinite(frame["market_cap_proxy"]) & frame["market_cap_proxy"].gt(0)]
    if frame.empty:
        raise ValueError("market_cap must contain positive finite market-cap rows")
    return frame.sort_values(["quarter_start", "selection_date", "symbol"]).reset_index(drop=True)


def _scale_to_cash(budgets, cash, min_notional=1.0):
    """Scale positive budgets below available cash, leaving a tiny buffer.

    The buffer can make an exact ``n x min_notional`` basket fall just below
    the floor; that is the conservative side of the contract (no order is
    created below the floor), not an overspend.
    """
    if not np.isfinite(cash) or cash <= 0:
        return {}
    budgets = {
        symbol: float(value)
        for symbol, value in budgets.items()
        if np.isfinite(value) and value > 0
    }
    usable = max(0.0, cash - max(1e-6, abs(cash) * 1e-9))
    if not budgets or usable <= 0:
        return {}
    try:
        total = math.fsum(budgets.values())
    except OverflowError:
        return {}
    if not np.isfinite(total) or total <= 0:
        return {}
    if total > usable:
        factor = usable / total
        budgets = {symbol: value * factor for symbol, value in budgets.items()}
    # The usable buffer already leaves room for rounding; no per-ulp loop is
    # needed and equal inputs stay equal.
    return {symbol: value for symbol, value in budgets.items()
            if value > 0 and value >= min_notional}


class HMMVolumeStrategy(Strategy):
    """Long-only market timing with a walk-forward volume-bar HMM.

    Parameters
    ----------
    benchmark, volume:
        Full benchmark price and volume-proxy series (or DataFrames with
        ``date`` and ``adj_close``/``volume``). The strategy only reads dates
        on or before ``context.signal_date``.
    median_window, bucket_multiple:
        Volume-bar rule. ``90`` sessions and ``5`` reproduce the research.
    n_states:
        Number of Gaussian HMM states. States are relabelled by emission mean,
        so state 0 is the lowest-mean state and ``n_states - 1`` the highest.
    min_train_bars:
        Minimum number of completed bars before the first fit.
    refit_every_bars:
        Refit the HMM after this many new completed bars. Between refits the
        state is updated with the last fitted model.
    risk_on_states:
        States in which new purchases are allowed. The default avoids only the
        lowest-mean (crisis) state.
    market_cap:
        Optional point-in-time membership frame with ``symbol``,
        ``quarter_start``, ``selection_date``, and ``market_cap_proxy``. When
        supplied, the top ``top_n`` eligible names are ranked by the latest
        known market cap and the basket is weighted by those market caps. If
        omitted, the strategy falls back to equal weight over the eligible
        names (the development backtest supplies market caps).
    top_n:
        Maximum number of names in a purchase basket, selected by market cap.
        ``None`` means every eligible name.
    liquidity_window:
        Legacy fallback used only when ``market_cap`` is not supplied: rank by
        trailing dollar volume over this window. The development backtest
        supplies market caps, so stock selection there is market-cap based and
        volume is used only for the HMM bar aggregation.

    A ``Backtester`` consumes the supplied strategy object and may advance its
    state between refits.  Pass a fresh ``deepcopy`` for each independent run.
    """

    self_fitting = True

    def __init__(
        self,
        benchmark,
        volume,
        *,
        market_cap=None,
        strategy_id: str = "hmm_volume",
        median_window: int = 90,
        bucket_multiple: float = 5.0,
        n_states: int = 3,
        min_train_bars: int = 250,
        refit_every_bars: int = 13,
        n_init: int = 3,
        seed: int = 7,
        n_iter: int = 200,
        risk_on_states=None,
        top_n: int | None = 50,
        liquidity_window: int = 63,
        min_order_notional: float = 1.0,
    ):
        super().__init__(strategy_id)
        if not _is_int(median_window) or median_window < 2:
            raise ValueError("median_window must be an integer of at least 2")
        if (isinstance(bucket_multiple, bool)
                or not isinstance(bucket_multiple, numbers.Real)
                or not np.isfinite(bucket_multiple)
                or bucket_multiple <= 0):
            raise ValueError("bucket_multiple must be finite and positive")
        if not _is_int(n_states) or n_states < 2:
            raise ValueError("n_states must be an integer of at least 2")
        if not _is_int(min_train_bars) or min_train_bars < max(20, 5 * n_states):
            raise ValueError(
                "min_train_bars must be at least max(20, 5 * n_states)"
            )
        if not _is_int(refit_every_bars) or refit_every_bars < 1:
            raise ValueError("refit_every_bars must be a positive integer")
        if not _is_int(n_init) or n_init < 1:
            raise ValueError("n_init must be a positive integer")
        if not _is_int(n_iter) or n_iter < 1:
            raise ValueError("n_iter must be a positive integer")
        if not _is_int(seed):
            raise ValueError("seed must be an integer")
        if risk_on_states is None:
            states = tuple(range(1, n_states))
        else:
            states = tuple(risk_on_states)
        if (not states
                or any(not _is_int(s) or s < 0 or s >= n_states for s in states)
                or len(set(states)) != len(states)):
            raise ValueError("risk_on_states must be unique valid state indices")
        if top_n is not None and (not _is_int(top_n) or top_n < 1):
            raise ValueError("top_n must be a positive integer or None")
        if not _is_int(liquidity_window) or liquidity_window < 1:
            raise ValueError("liquidity_window must be a positive integer")
        if (isinstance(min_order_notional, bool)
                or not isinstance(min_order_notional, numbers.Real)
                or not np.isfinite(min_order_notional)
                or min_order_notional < 0):
            raise ValueError("min_order_notional must be finite and nonnegative")

        self.benchmark = _as_series(benchmark, name="benchmark", column="adj_close")
        self.volume = _as_series(volume, name="volume", column="volume", positive=False)
        common = self.benchmark.index.intersection(self.volume.index)
        if common.empty:
            raise ValueError("benchmark and volume have no common dates")
        self.benchmark = self.benchmark.reindex(common)
        self.volume = self.volume.reindex(common)

        self.median_window = median_window
        self.bucket_multiple = float(bucket_multiple)
        self.n_states = n_states
        self.min_train_bars = min_train_bars
        self.refit_every_bars = refit_every_bars
        self.n_init = n_init
        self.seed = seed
        self.n_iter = n_iter
        self.risk_on_states = states
        self.top_n = top_n
        self.liquidity_window = liquidity_window
        self.market_cap = _market_cap_frame(market_cap)
        self.min_order_notional = float(min_order_notional)

        # The bar builder is causal: a future bar cannot change an earlier
        # bar. The strategy filters by bar end date and never reads a future
        # bar. This avoids rebuilding the same bars on every session.
        self._bars = volume_bars(
            self.benchmark,
            self.volume,
            median_window=self.median_window,
            bucket_multiple=self.bucket_multiple,
        ).bars
        if self._bars.empty:
            raise ValueError("The volume rule produced no completed bars")

        self._fit = None
        self._state = None
        self._probabilities = None
        self._fit_bar_count = 0
        self._fit_bar_end = None
        self._state_bar_count = 0
        self._state_bar_end = None
        self._fit_date = None
        self._bar_returns = None

    def __deepcopy__(self, memo):
        """Share the large read-only inputs; copy the mutable model state."""
        clone = type(self).__new__(type(self))
        memo[id(self)] = clone
        shared = {"benchmark", "volume", "_bars", "market_cap"}
        for key, value in self.__dict__.items():
            setattr(clone, key, value if key in shared else deepcopy(value, memo))
        return clone

    def _bars_at(self, signal_date) -> pd.DataFrame:
        """Return completed bars whose end is on or before the signal date."""
        signal = pd.Timestamp(signal_date)
        return self._bars.loc[self._bars["end"] <= signal]

    def _check_signal(self, signal_date) -> pd.Timestamp:
        """Reject a missing signal or one after the supplied history."""
        if signal_date is None or pd.isna(signal_date):
            raise ValueError("Signal date must be a valid timestamp")
        try:
            signal = pd.Timestamp(signal_date)
        except Exception:
            raise ValueError("Signal date must be a valid timestamp") from None
        if pd.isna(signal):
            raise ValueError("Signal date must be a valid timestamp")
        if signal.tz is not None:
            raise ValueError("Signal date must be timezone-naive")
        if signal > self.benchmark.index[-1]:
            raise ValueError(
                "Strategy benchmark/volume data ended before the signal date"
            )
        return signal

    def needs_refit(self, context: DecisionContext) -> bool:
        signal = self._check_signal(context.signal_date)
        bars = self._bars_at(signal)
        if len(bars) < self.min_train_bars + 1:
            return False
        if self._fit is None:
            return True
        if bars["end"].iloc[-1] <= self._fit_bar_end:
            return False
        return len(bars) - self._fit_bar_count >= self.refit_every_bars

    def refit(self, data, context: DecisionContext) -> None:
        signal = self._check_signal(context.signal_date)
        if self._fit_date is not None and signal < self._fit_date:
            raise ValueError("Refit signal date cannot move backwards")
        if self._state_bar_end is not None and signal < self._state_bar_end:
            raise ValueError("Refit signal date cannot move backwards")
        bars = self._bars_at(signal)
        if len(bars) < self.min_train_bars + 1:
            raise ValueError("Not enough completed bars to fit the HMM")
        if self._fit is not None and len(bars) < self._state_bar_count:
            raise ValueError("Refit cannot discard previously scored bars")
        returns = bars["log_return"].to_numpy(dtype=float)
        # Fit on bars strictly before the current bar, then score the current
        # bar. This is the same one-step-ahead rule as the research.
        fit = fit_hmm(
            returns[:-1],
            n_states=self.n_states,
            n_init=self.n_init,
            seed=self.seed,
            n_iter=self.n_iter,
        )
        state, probabilities = infer_state(fit, returns)
        self._fit = fit
        self._state = int(state)
        self._probabilities = probabilities
        self._fit_bar_count = len(bars)
        self._fit_bar_end = bars["end"].iloc[-1]
        self._state_bar_count = len(bars)
        self._state_bar_end = bars["end"].iloc[-1]
        self._fit_date = signal
        self._bar_returns = returns

    @staticmethod
    def _validate_panel(frame, name: str, signal: pd.Timestamp) -> None:
        """Reject a planning panel that is not a causal dated frame."""
        if frame is None or len(frame) == 0:
            return
        index = frame.index
        if not isinstance(index, pd.DatetimeIndex):
            raise ValueError(f"context.{name} must use a DatetimeIndex")
        if index.hasnans:
            raise ValueError(f"context.{name} contains a missing date")
        if index.tz is not None:
            raise ValueError(f"context.{name} must be timezone-naive")
        if index.max() > signal:
            raise ValueError(f"context.{name} contains data after the signal date")

    def _update_state(self, bars: pd.DataFrame) -> None:
        """Score a newly completed bar with the last fitted model."""
        if self._fit is None or len(bars) <= self._state_bar_count:
            return
        returns = bars["log_return"].to_numpy(dtype=float)
        state, probabilities = infer_state(self._fit, returns)
        self._state = int(state)
        self._probabilities = probabilities
        self._state_bar_count = len(bars)
        self._state_bar_end = bars["end"].iloc[-1]
        self._bar_returns = returns

    def propose_buys(self, context: DecisionContext):
        signal = self._check_signal(context.signal_date)
        if self._fit_date is not None and signal < self._fit_date:
            raise ValueError("Signal date moved backwards")
        if self._state_bar_end is not None and signal < self._state_bar_end:
            raise ValueError("Signal date moved backwards")
        self._validate_panel(context.prices, "prices", signal)
        self._validate_panel(context.volume, "volume", signal)

        if self._fit is None:
            return {}
        bars = self._bars_at(signal)
        self._update_state(bars)
        if self._state is None or self._state not in self.risk_on_states:
            return {}
        cash = float(context.portfolio.available_cash)
        if not np.isfinite(cash) or cash <= 0:
            return {}

        allowed = set(context.membership.symbol) & set(context.prices.columns)
        if not allowed or context.prices.empty:
            return {}
        last_prices = context.prices.iloc[-1]
        allowed = sorted(
            symbol for symbol in allowed
            if symbol in last_prices.index
            and np.isfinite(last_prices[symbol])
            and last_prices[symbol] > 0
        )
        if not allowed:
            return {}

        weights = self._market_cap_weights(context, allowed)
        if weights is None:
            if self.market_cap is not None:
                # Caps were supplied but no usable row exists at this signal.
                # Never let stock volume rank names on the cap-weighted path.
                return {}
            # Legacy fallback for callers that do not supply market caps.
            if self.top_n is not None and len(allowed) > self.top_n:
                selected = self._rank_by_liquidity(context, allowed)
            else:
                selected = allowed
            if not selected:
                return {}
            weights = pd.Series(1.0 / len(selected), index=selected)
        if weights.empty:
            return {}

        return _scale_to_cash(
            {symbol: cash * float(weight) for symbol, weight in weights.items()},
            cash,
            min_notional=self.min_order_notional,
        )

    def _rank_by_liquidity(self, context: DecisionContext, allowed):
        """Legacy fallback: rank by trailing dollar volume when no caps exist."""
        if context.volume is not None and not context.volume.empty:
            window = min(self.liquidity_window, len(context.prices))
            prices = context.prices.tail(window)
            volumes = context.volume.reindex(index=prices.index).tail(window)
            columns = [symbol for symbol in allowed if symbol in volumes.columns]
            if columns:
                dollar = (prices[columns] * volumes[columns]).mean()
                dollar = dollar.replace([np.inf, -np.inf], np.nan).dropna()
                ranked = dollar.sort_values(ascending=False, kind="stable").index.tolist()
                if len(ranked) >= self.top_n:
                    return ranked[: self.top_n]
                remainder = [symbol for symbol in allowed if symbol not in set(ranked)]
                return (ranked + remainder)[: self.top_n]
        return allowed[: self.top_n]

    def _market_cap_weights(self, context: DecisionContext, allowed):
        """Cap-weight the top ``top_n`` eligible names from the latest snapshot."""
        if self.market_cap is None or self.market_cap.empty:
            return None
        signal = pd.Timestamp(context.signal_date)
        caps = self.market_cap.loc[self.market_cap["selection_date"].le(signal)]
        if caps.empty:
            return None
        quarter = caps["quarter_start"].max()
        caps = caps.loc[caps["quarter_start"].eq(quarter)]
        caps = caps.loc[caps["symbol"].isin(allowed)]
        if caps.empty:
            return None
        caps = caps.sort_values(["selection_date", "symbol"]).drop_duplicates(
            "symbol", keep="last"
        ).set_index("symbol")
        caps = caps.loc[np.isfinite(caps["market_cap_proxy"]) & caps["market_cap_proxy"].gt(0)]
        if caps.empty:
            return None
        caps = caps.sort_values("market_cap_proxy", ascending=False, kind="stable")
        if self.top_n is not None:
            caps = caps.head(self.top_n)
        total = float(caps["market_cap_proxy"].sum())
        if not np.isfinite(total) or total <= 0:
            return None
        return caps["market_cap_proxy"] / total

    def refit_settings(self):
        return dict(
            median_window=self.median_window,
            bucket_multiple=self.bucket_multiple,
            n_states=self.n_states,
            min_train_bars=self.min_train_bars,
            refit_every_bars=self.refit_every_bars,
            n_init=self.n_init,
            seed=self.seed,
            n_iter=self.n_iter,
            risk_on_states=list(self.risk_on_states),
            top_n=self.top_n,
            weighting="market_cap" if self.market_cap is not None else "liquidity_fallback",
            liquidity_window=self.liquidity_window,
            min_order_notional=self.min_order_notional,
            model="GaussianHMM on volume-bar log returns",
            market_cap_rows=0 if self.market_cap is None else len(self.market_cap),
        )

    @property
    def state(self):
        return self._state

    @property
    def state_date(self):
        return self._state_bar_end

    @property
    def probabilities(self):
        return None if self._probabilities is None else self._probabilities.copy()

    @property
    def bars(self):
        return self._bars.copy()
