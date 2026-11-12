"""Long-only 90-day backtester with cash, lots, and daily NAV.

The accounting rules follow Lectures/modeling/README.md and the Lecture 8 OMS
design (inclusive entry budgets, independent lots, sales that free cash
immediately). This file is an independent implementation written for HW4.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd


@dataclass
class Lot:
    lot_id: int
    symbol: str
    units: float
    signal_date: pd.Timestamp
    entry_date: pd.Timestamp
    entry_price: float
    entry_fee: float
    due_date: pd.Timestamp


@dataclass
class Fill:
    date: pd.Timestamp
    signal_date: pd.Timestamp
    lot_id: int
    symbol: str
    side: str
    price: float
    units: float
    notional: float
    fee: float
    cash_change: float
    due_date: pd.Timestamp
    realized_pnl: float


class OrderBook:
    """Cash, lots, fees, and fills. Prices are supplied by the simulator."""

    def __init__(self, initial_cash: float = 1_000_000.0, fee_bps: float = 5.0):
        if initial_cash <= 0 or fee_bps < 0:
            raise ValueError("Capital must be positive and fees nonnegative")
        self.initial_cash = float(initial_cash)
        self.cash = float(initial_cash)
        self.fee_rate = fee_bps / 10_000.0
        self.lots: dict[int, Lot] = {}
        self.fills: list[Fill] = []
        self.total_fees = 0.0
        self.realized_pnl = 0.0
        self._next_id = 0

    def size_for_budget(self, budget: float, price: float) -> float:
        """Units bought so that price * units * (1 + fee) equals the budget."""
        if budget < 0 or price <= 0:
            raise ValueError("Budget must be nonnegative and price positive")
        return budget / (price * (1.0 + self.fee_rate))

    def buy(self, units: float, price: float, *, symbol: str, date, signal_date, holding_days: int = 90) -> int:
        date = pd.Timestamp(date)
        signal_date = pd.Timestamp(signal_date)
        if units <= 0 or price <= 0:
            raise ValueError("Size and price must be positive")
        if signal_date >= date:
            raise ValueError("Signal date must be strictly before the fill date")
        notional = units * price
        fee = notional * self.fee_rate
        if notional + fee > self.cash + 1e-8:
            raise ValueError("Insufficient cash for the purchase and its fee")
        lot = Lot(
            lot_id=self._next_id,
            symbol=symbol,
            units=units,
            signal_date=signal_date,
            entry_date=date,
            entry_price=price,
            entry_fee=fee,
            due_date=date + pd.Timedelta(days=holding_days),
        )
        self.cash = max(0.0, self.cash - notional - fee)
        self.lots[lot.lot_id] = lot
        self._next_id += 1
        self._record(lot, "BUY", date, units, price, fee, -(notional + fee), 0.0)
        return lot.lot_id

    def sell(self, units: float, price: float, *, lot_id: int, date, charge_fee: bool = True) -> float:
        lot = self.lots[lot_id]
        date = pd.Timestamp(date)
        if units <= 0 or price <= 0:
            raise ValueError("Size and price must be positive")
        if date < lot.entry_date:
            raise ValueError("A sale cannot precede its purchase")
        if units > lot.units + 1e-12:
            raise ValueError("Cannot sell more units than the lot owns")
        units = min(units, lot.units)
        notional = units * price
        fee = notional * self.fee_rate if charge_fee else 0.0
        entry_fee = lot.entry_fee * units / lot.units
        pnl = units * (price - lot.entry_price) - entry_fee - fee
        self.cash += notional - fee
        self.realized_pnl += pnl
        if np.isclose(units, lot.units):
            del self.lots[lot_id]
        else:
            self.lots[lot_id] = replace(lot, units=lot.units - units, entry_fee=lot.entry_fee - entry_fee)
        self._record(lot, "SELL", date, units, price, fee, notional - fee, pnl)
        return pnl

    def _record(self, lot: Lot, side: str, date, units, price, fee, cash_change, pnl) -> None:
        self.total_fees += fee
        self.fills.append(
            Fill(date, lot.signal_date, lot.lot_id, lot.symbol, side, price, units,
                 units * price, fee, cash_change, lot.due_date, pnl)
        )

    def holdings_value(self, marks: dict[str, float]) -> float:
        total = 0.0
        for lot in self.lots.values():
            price = marks[lot.symbol]
            if not np.isfinite(price) or price <= 0:
                raise ValueError(f"Invalid mark for {lot.symbol}")
            total += lot.units * price
        return total

    def nav(self, marks: dict[str, float]) -> float:
        return self.cash + self.holdings_value(marks)

    def exposure_by_symbol(self, marks: dict[str, float]) -> pd.Series:
        values: dict[str, float] = {}
        for lot in self.lots.values():
            values[lot.symbol] = values.get(lot.symbol, 0.0) + lot.units * marks[lot.symbol]
        return pd.Series(values, dtype=float)


@dataclass
class BacktestResult:
    history: pd.DataFrame
    fills: pd.DataFrame
    open_lots: pd.DataFrame
    diagnostics: pd.DataFrame
    pnl: pd.Series


class MonthlyTopNAllocator:
    """Buy the highest scores on the first session of each month.

    Decisions use the previous session's close, scores, cash, and NAV. Execution
    uses the current session's close. A sale at today's close cannot increase
    today's buy. Overlapping 90-day lots are funded by spending at most
    `cohort_fraction` of the previous close's NAV, and only cash already on hand.
    """

    def __init__(self, scores: pd.DataFrame, top_n: int = 20, cohort_fraction: float = 1.0 / 3.0,
                 holding_days: int = 90):
        if top_n < 1 or not 0 < cohort_fraction <= 1 or holding_days < 1:
            raise ValueError("Invalid allocation settings")
        scores = scores[["date", "symbol", "score"]].copy()
        scores["date"] = pd.to_datetime(scores["date"])
        if scores.duplicated(["date", "symbol"]).any() or not np.isfinite(scores.score).all():
            raise ValueError("Scores must be finite with one row per date and symbol")
        self.scores = {date: group.set_index("symbol").score for date, group in scores.groupby("date")}
        self.top_n = top_n
        self.cohort_fraction = cohort_fraction
        self.holding_days = holding_days

    def orders(self, signal_date, execution_date, cash: float, nav: float) -> pd.Series:
        signal_date = pd.Timestamp(signal_date)
        execution_date = pd.Timestamp(execution_date)
        if execution_date.to_period("M") == signal_date.to_period("M"):
            return pd.Series(dtype=float)
        ranked = self.scores.get(signal_date)
        if ranked is None or ranked.empty:
            return pd.Series(dtype=float)
        chosen = ranked.rename_axis("symbol").rename("score").reset_index().sort_values(
            ["score", "symbol"], ascending=[False, True]
        ).head(self.top_n)
        budget = min(cash, nav * self.cohort_fraction)
        per_name = budget / len(chosen) if len(chosen) else 0.0
        return pd.Series(per_name, index=chosen.symbol, name="budget", dtype=float)


class Backtester:
    """Plan at yesterday's close, fill at today's close, mark every session."""

    def __init__(self, prices: pd.DataFrame, benchmark: pd.DataFrame, allocator: MonthlyTopNAllocator,
                 initial_cash: float = 1_000_000.0, fee_bps: float = 5.0, max_stale_sessions: int = 5):
        self.allocator = allocator
        self.book = OrderBook(initial_cash, fee_bps)
        self.max_stale_sessions = max_stale_sessions
        prices = prices[["date", "symbol", "adj_close"]].copy()
        benchmark = benchmark[["date", "adj_close"]].copy()
        prices["date"] = pd.to_datetime(prices.date)
        benchmark["date"] = pd.to_datetime(benchmark.date)
        if prices.duplicated(["date", "symbol"]).any() or benchmark.date.duplicated().any():
            raise ValueError("Quotes need unique dates and symbols")
        invalid = ~np.isfinite(prices.adj_close) | prices.adj_close.le(0)
        prices = prices.loc[~invalid].copy()
        self.benchmark = benchmark.set_index("date").adj_close.sort_index()
        self.calendar = pd.DatetimeIndex(self.benchmark.index)
        prices = prices.loc[prices.date.isin(self.calendar)]
        self.quotes = prices.pivot(index="date", columns="symbol", values="adj_close").reindex(self.calendar)

    def run(self, start, end) -> BacktestResult:
        start = pd.Timestamp(start)
        end = pd.Timestamp(end)
        days = self.calendar[(self.calendar >= start) & (self.calendar <= end)]
        if not len(days) or self.calendar.get_loc(days[0]) == 0:
            raise ValueError("Need test sessions and one preceding signal session")
        last_marks: dict[str, float] = {}
        last_seen: dict[str, int] = {}
        diagnostics: list[dict] = []
        previous_day = self.calendar[self.calendar.get_loc(days[0]) - 1]
        history = [{
            "date": previous_day, "cash": self.book.cash, "holdings_value": 0.0,
            "nav": self.book.cash, "open_lots": 0, "fees": 0.0, "gross_traded": 0.0, "stale_marks": 0,
        }]
        for day in days:
            day_index = self.calendar.get_loc(day)
            signal_date = self.calendar[day_index - 1]
            nav_for_orders = self.book.nav(last_marks) if last_marks else self.book.cash
            orders = self.allocator.orders(signal_date, day, self.book.cash, nav_for_orders)
            if not orders.empty and (orders.lt(0).any() or orders.sum() > self.book.cash + 1e-8):
                raise ValueError("Order budgets must be nonnegative and covered by cash known before the fill")
            observed = self.quotes.loc[day].dropna()
            for symbol, price in observed.items():
                last_marks[symbol] = float(price)
                last_seen[symbol] = day_index
            stale = 0
            fill_start = len(self.book.fills)
            for lot in list(self.book.lots.values()):
                gap = day_index - last_seen.get(lot.symbol, day_index)
                if gap > 0:
                    stale += 1
                    diagnostics.append({"date": day, "symbol": lot.symbol, "event": "stale_mark"})
                if gap > self.max_stale_sessions:
                    diagnostics.append({"date": day, "symbol": lot.symbol, "event": "unresolved_held_gap"})
                if day >= lot.due_date and lot.symbol in observed.index:
                    self.book.sell(lot.units, float(observed[lot.symbol]), lot_id=lot.lot_id, date=day)
            for symbol, budget in orders.items():
                if budget <= 1e-8:
                    continue
                if symbol not in observed.index:
                    diagnostics.append({"date": day, "symbol": symbol, "event": "rejected_missing_quote"})
                    continue
                price = float(observed[symbol])
                units = self.book.size_for_budget(float(budget), price)
                self.book.buy(units, price, symbol=symbol, date=day, signal_date=signal_date,
                              holding_days=self.allocator.holding_days)
            nav = self.book.nav(last_marks) if last_marks else self.book.cash
            daily_fills = self.book.fills[fill_start:]
            history.append({
                "date": day,
                "cash": self.book.cash,
                "holdings_value": self.book.holdings_value(last_marks) if last_marks else 0.0,
                "nav": nav,
                "open_lots": len(self.book.lots),
                "fees": sum(fill.fee for fill in daily_fills),
                "gross_traded": sum(fill.notional for fill in daily_fills),
                "stale_marks": stale,
            })
        history = pd.DataFrame(history).set_index("date")
        first_close = days[0]
        history["benchmark_nav"] = self.book.initial_cash
        history.loc[days, "benchmark_nav"] = (
            self.book.initial_cash * self.benchmark.loc[days] / self.benchmark.loc[first_close]
        )
        lots = pd.DataFrame([lot.__dict__ for lot in self.book.lots.values()])
        fills = pd.DataFrame([fill.__dict__ for fill in self.book.fills])
        holdings = self.book.holdings_value(last_marks) if last_marks else 0.0
        pnl = pd.Series({
            "cash": self.book.cash,
            "holdings_value": holdings,
            "nav": self.book.cash + holdings,
            "realized_pnl": self.book.realized_pnl,
            "total_pnl": self.book.cash + holdings - self.book.initial_cash,
            "total_fees": self.book.total_fees,
        })
        return BacktestResult(
            history=history,
            fills=fills,
            open_lots=lots,
            diagnostics=pd.DataFrame(diagnostics, columns=["date", "symbol", "event"]),
            pnl=pnl,
        )


def performance(nav: pd.Series, annual_rf: float = 0.02) -> pd.Series:
    """CAGR uses elapsed calendar days / 365.25; Sharpe uses 252 sessions and 2% rf."""
    nav = nav.sort_index()
    if len(nav) < 2 or nav.le(0).any():
        raise ValueError("Need at least two positive NAV observations")
    returns = nav.pct_change(fill_method=None).dropna()
    excess = returns - ((1.0 + annual_rf) ** (1.0 / 252.0) - 1.0)
    volatility = returns.std(ddof=1)
    years = (nav.index[-1] - nav.index[0]).total_seconds() / (365.25 * 24 * 3600)
    return pd.Series({
        "final_value": float(nav.iloc[-1]),
        "total_return": float(nav.iloc[-1] / nav.iloc[0] - 1.0),
        "CAGR": float((nav.iloc[-1] / nav.iloc[0]) ** (1.0 / years) - 1.0),
        "max_drawdown": float((nav / nav.cummax() - 1.0).min()),
        "annualized_volatility": float(volatility * np.sqrt(252)),
        "Sharpe": float(np.sqrt(252) * excess.mean() / volatility) if volatility > 1e-12 else np.nan,
    })
