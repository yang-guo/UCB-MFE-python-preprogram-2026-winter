"""Homework 4 backtester.

Here I tried to use a class layout that follows the Lecture 8 framework (Lectures/modeling/backtest.py):

- ``Strategy`` decides what to buy at a signal close.
- ``OrderManagementSystem`` owns cash, lots, fees, and fills.
- ``Backtester`` advances the market calendar one session at a time.
- ``BacktestAnalytics`` turns daily NAV into performance metrics.

The implementation is my own simplified version of that design, written for the
frozen HW4 rules:

- Scores are prepared before the backtest; each monthly cohort buys the top N
  with a fee-inclusive budget of ``min(NAV * cohort_fraction, cash)``.
- Orders planned at the signal close fill at the next session's adjusted close.
- Each purchase is a separate lot, sold at the first quoted close on or after
  entry + 90 calendar days.
- A missing quote never creates a fill. Held positions keep their last observed
  mark, and the number of such stale marks is counted each day in ``history``.
  Delayed exits, unresolved exits, missing entry quotes, and large one-day price
  jumps in held lots are recorded as diagnostic events instead of being repaired.

Prices are adjusted closes, so dividends are already included and never added.
"""

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Lot:
    """One purchase with its own cost basis and maturity date."""

    lot_id: int
    symbol: str
    units: float
    signal_date: pd.Timestamp
    entry_date: pd.Timestamp
    entry_price: float
    entry_fee: float
    due_date: pd.Timestamp


@dataclass(frozen=True)
class PortfolioSnapshot:
    """What a strategy is allowed to see when it plans orders."""

    cash: float
    nav: float


# Strategy: decides what to buy
class Strategy(ABC):
    """Interface every strategy implements."""

    def __init__(self, holding_days=90):
        self.holding_days = holding_days

    @abstractmethod
    def generate_orders(self, signal_date, execution_date, portfolio):
        """Return fee-inclusive buy budgets indexed by symbol."""

    def should_sell(self, lot, date):
        """A lot matures 90 calendar days after entry."""
        return date >= lot.due_date


class TopNScoreStrategy(Strategy):
    """Buy the highest model scores on each signal date, equally weighted.

    ``scores`` has columns date, symbol, score. Only rows for the signal date
    are read, so the strategy cannot see later scores.
    """

    def __init__(self, scores, top_n=20, cohort_fraction=1 / 3, holding_days=90):
        super().__init__(holding_days)
        self.top_n = top_n
        self.cohort_fraction = cohort_fraction
        scores = scores.assign(date=pd.to_datetime(scores["date"]))
        self.scores = {
            date: group.set_index("symbol")["score"]
            for date, group in scores.groupby("date")
        }
        self.budget_cuts = []  # Cohorts whose budget was limited by cash.

    def score_stocks(self, signal_date):
        return self.scores.get(signal_date, pd.Series(dtype=float))

    def allocate_budgets(self, scores, portfolio, signal_date):
        """Split min(NAV * fraction, cash) equally across the top N scores."""
        if scores.empty:
            return pd.Series(dtype=float)
        # Sort by score, breaking ties alphabetically so runs are reproducible.
        ranked = (
            scores.rename("score").rename_axis("symbol").reset_index()
            .sort_values(["score", "symbol"], ascending=[False, True])
            .head(self.top_n)
        )
        target = portfolio.nav * self.cohort_fraction
        budget = min(target, portfolio.cash)
        if budget < target - 1e-6:
            self.budget_cuts.append(
                {"signal_date": signal_date, "target": target, "budget": budget}
            )
        if budget <= 1e-6:  # No cash left: skip this cohort rather than buy zero units.
            return pd.Series(dtype=float)
        return pd.Series(budget / len(ranked), index=ranked["symbol"].to_numpy())

    def generate_orders(self, signal_date, execution_date, portfolio):
        scores = self.score_stocks(signal_date)
        return self.allocate_budgets(scores, portfolio, signal_date)



# Order management: cash, lots, fees, fills
class OrderManagementSystem:
    """Book purchases and sales and keep every lot separate."""

    def __init__(self, initial_cash=1_000_000, fee_bps=5):
        self.initial_cash = float(initial_cash)
        self.cash = float(initial_cash)
        self.fee_rate = fee_bps / 10_000
        self.lots = {}
        self.fills = []
        self._next_lot_id = 0

    def size_for_budget(self, budget, price):
        """Units such that units * price + fee equals the budget."""
        return budget / (price * (1 + self.fee_rate))

    def buy(self, units, price, *, symbol, date, signal_date, holding_days):
        notional = units * price
        fee = notional * self.fee_rate
        if notional + fee > self.cash + 1e-6:
            raise ValueError(f"Insufficient cash to buy {symbol} on {date.date()}")
        self.cash = max(0.0, self.cash - notional - fee)  # Remove float dust.
        lot = Lot(self._next_lot_id, symbol, units, signal_date, date, price, fee,
                  date + pd.Timedelta(days=holding_days))
        self.lots[lot.lot_id] = lot
        self._next_lot_id += 1
        self._record_fill(lot, "BUY", date, price, fee, -(notional + fee), 0.0)
        return lot

    def sell(self, lot_id, price, date):
        """Sell a whole lot; cash is available immediately."""
        lot = self.lots.pop(lot_id)
        notional = lot.units * price
        fee = notional * self.fee_rate
        pnl = notional - fee - lot.units * lot.entry_price - lot.entry_fee
        self.cash += notional - fee
        self._record_fill(lot, "SELL", date, price, fee, notional - fee, pnl)
        return pnl

    def _record_fill(self, lot, side, date, price, fee, cash_change, pnl):
        self.fills.append({
            "date": date, "signal_date": lot.signal_date, "lot_id": lot.lot_id,
            "symbol": lot.symbol, "side": side, "units": lot.units, "price": price,
            "notional": lot.units * price, "fee": fee, "cash_change": cash_change,
            "cash_after": self.cash, "due_date": lot.due_date, "realized_pnl": pnl,
        })

    def holdings_value(self, marks):
        return sum(lot.units * marks[lot.symbol] for lot in self.lots.values())

    def snapshot(self, marks):
        return PortfolioSnapshot(self.cash, self.cash + self.holdings_value(marks))



# Backtester: the daily loop
@dataclass(frozen=True)
class BacktestResult:
    history: pd.DataFrame      # One row per session: cash, holdings, NAV, benchmark.
    fills: pd.DataFrame        # Every BUY and SELL.
    open_lots: pd.DataFrame    # Lots still held at the end, with their last mark.
    diagnostics: pd.DataFrame  # Delayed/unresolved exits, missing quotes, jumps.
    budget_cuts: pd.DataFrame  # Cohorts whose budget was limited by cash.


class Backtester:
    """Plan at the previous close, trade at today's close, then record NAV."""

    def __init__(self, prices, benchmark, strategy, oms=None,
                 max_exit_delay=5, jump_bounds=(0.1, 10.0)):
        """
        prices:    long table with date, symbol, adj_close (observed quotes only).
        benchmark: date, adj_close for ^SP500TR; its dates are the market calendar.
        """
        self.strategy = strategy
        self.oms = oms if oms is not None else OrderManagementSystem()
        self.max_exit_delay = max_exit_delay
        self.jump_bounds = jump_bounds

        benchmark = benchmark.assign(date=pd.to_datetime(benchmark["date"]))
        self.benchmark = benchmark.set_index("date")["adj_close"].sort_index()
        prices = prices.assign(date=pd.to_datetime(prices["date"]))
        self.calendar = self.benchmark.index[self.benchmark.index >= prices["date"].min()]
        # Wide table: rows are market sessions, columns are symbols, NaN = no quote.
        self.quotes = (
            prices.pivot(index="date", columns="symbol", values="adj_close")
            .reindex(self.calendar)
        )

    def _log(self, date, event, symbol=None, lot_id=None, value=np.nan,
             previous_mark=np.nan):
        self.diagnostics.append(
            {"date": date, "event": event, "symbol": symbol, "lot_id": lot_id,
             "value": value, "previous_mark": previous_mark}
        )

    def run_backtest(self, start, end):
        days = self.calendar[(self.calendar >= pd.Timestamp(start))
                             & (self.calendar <= pd.Timestamp(end))]
        first = self.calendar.get_loc(days[0])
        if first == 0:
            raise ValueError("Need one session before start for the first signal")

        oms = self.oms
        self.diagnostics = []
        marks = {}      # Last observed price per symbol.
        last_seen = {}  # Calendar position of that price.
        low, high = self.jump_bounds

        portfolio = oms.snapshot(marks)
        history = [{"date": self.calendar[first - 1], "cash": oms.cash,
                    "holdings_value": 0.0, "nav": oms.cash, "open_lots": 0,
                    "stale_marks": 0}]

        for day in days:
            i = self.calendar.get_loc(day)
            signal_date = self.calendar[i - 1]

            # 1. Plan with yesterday's close only: today's prices are unknown.
            orders = self.strategy.generate_orders(signal_date, day, portfolio)
            if orders.sum() > portfolio.cash + 1e-6:
                raise ValueError(f"Orders on {day.date()} exceed signal-date cash")

            # 2. Read today's quotes. Flag every held lot whose price jumps by more
            #    than 10x (or falls below 0.1x) since its last observed mark.
            observed = self.quotes.loc[day].dropna()
            for lot in oms.lots.values():
                if lot.symbol in observed.index:
                    ratio = observed[lot.symbol] / marks[lot.symbol]
                    if not low <= ratio <= high:
                        self._log(day, "held_price_jump", lot.symbol, lot.lot_id,
                                  value=ratio, previous_mark=marks[lot.symbol])
            marks.update(observed.to_dict())
            last_seen.update(dict.fromkeys(observed.index, i))

            # 3. Sell matured lots at today's close, if there is a quote.
            for lot in list(oms.lots.values()):
                if not self.strategy.should_sell(lot, day):
                    continue
                delay = i - self.calendar.searchsorted(lot.due_date)
                if lot.symbol in observed.index:
                    oms.sell(lot.lot_id, observed[lot.symbol], day)
                    if delay > 0:
                        self._log(day, "delayed_exit", lot.symbol, lot.lot_id, delay)
                elif delay == self.max_exit_delay + 1:
                    self._log(day, "unresolved_exit", lot.symbol, lot.lot_id, delay)

            # 4. Buy with the budgets fixed at the signal close.
            for symbol, budget in orders.items():
                if symbol not in observed.index:
                    self._log(day, "missing_entry_quote", symbol, value=budget)
                    continue
                price = observed[symbol]
                oms.buy(oms.size_for_budget(budget, price), price, symbol=symbol,
                        date=day, signal_date=signal_date,
                        holding_days=self.strategy.holding_days)

            # 5. Value the portfolio. Missing quotes keep the last observed mark.
            portfolio = oms.snapshot(marks)
            history.append({
                "date": day, "cash": oms.cash,
                "holdings_value": portfolio.nav - oms.cash, "nav": portfolio.nav,
                "open_lots": len(oms.lots),
                "stale_marks": sum(lot.symbol not in observed.index
                                   for lot in oms.lots.values()),
            })

        history = pd.DataFrame(history).set_index("date")
        # Benchmark: the same capital invested at the first test close.
        bench = self.benchmark.loc[days]
        history["benchmark_nav"] = oms.initial_cash
        history.loc[days, "benchmark_nav"] = oms.initial_cash * bench / bench.iloc[0]

        end_position = self.calendar.get_loc(days[-1])
        open_lots = pd.DataFrame([asdict(lot) for lot in oms.lots.values()],
                                 columns=list(Lot.__dataclass_fields__))
        open_lots["mark"] = open_lots["symbol"].map(marks)
        open_lots["value"] = open_lots["units"] * open_lots["mark"]
        open_lots["sessions_since_quote"] = [
            end_position - last_seen[s] for s in open_lots["symbol"]
        ]

        return BacktestResult(
            history=history,
            fills=pd.DataFrame(oms.fills),
            open_lots=open_lots,
            diagnostics=pd.DataFrame(self.diagnostics,
                                     columns=["date", "event", "symbol", "lot_id",
                                              "value", "previous_mark"]),
            budget_cuts=pd.DataFrame(self.strategy.budget_cuts,
                                     columns=["signal_date", "target", "budget"]),
        )



# Analytics: performance and diagnostics
class BacktestAnalytics:
    """Report the required metrics for the strategy and the benchmark."""

    def __init__(self, result, max_exit_delay=5):
        self.result = result
        self.max_exit_delay = max_exit_delay

    @staticmethod
    def performance(nav, annual_rf=0.02):
        """Final value, CAGR, max drawdown, and Sharpe from daily NAV."""
        nav = nav.sort_index()
        returns = nav.pct_change().dropna()
        daily_rf = (1 + annual_rf) ** (1 / 252) - 1
        years = (nav.index[-1] - nav.index[0]).days / 365.25
        return pd.Series({
            "final_value": nav.iloc[-1],
            "total_return": nav.iloc[-1] / nav.iloc[0] - 1,
            "CAGR": (nav.iloc[-1] / nav.iloc[0]) ** (1 / years) - 1,
            "max_drawdown": (nav / nav.cummax() - 1).min(),
            "annualized_volatility": returns.std() * np.sqrt(252),
            "Sharpe": np.sqrt(252) * (returns - daily_rf).mean() / returns.std(),
        })

    def summary(self, annual_rf=0.02):
        h = self.result.history
        return pd.DataFrame({
            "strategy": self.performance(h["nav"], annual_rf),
            "S&P 500 total return": self.performance(h["benchmark_nav"], annual_rf),
        })

    def diagnostics(self):
        r = self.result
        h = r.history.iloc[1:]  # Skip the initial-capital row.
        events = r.diagnostics["event"].value_counts()
        final_nav = r.history["nav"].iloc[-1]
        stale = r.open_lots.loc[r.open_lots["sessions_since_quote"] > self.max_exit_delay]
        return pd.Series({
            "buys": int(r.fills["side"].eq("BUY").sum()),
            "sells": int(r.fills["side"].eq("SELL").sum()),
            "total_fees": r.fills["fee"].sum(),
            "average_cash_fraction": (h["cash"] / h["nav"]).mean(),
            "budget_cuts": len(r.budget_cuts),
            "delayed_exits": int(events.get("delayed_exit", 0)),
            "unresolved_exits": int(events.get("unresolved_exit", 0)),
            "missing_entry_quotes": int(events.get("missing_entry_quote", 0)),
            "held_price_jumps": int(events.get("held_price_jump", 0)),
            "open_lots_at_end": len(r.open_lots),
            "stale_value_at_end": stale["value"].sum(),
            "stale_share_of_final_nav": stale["value"].sum() / final_nav,
            "final_nav_if_stale_worth_zero": final_nav - stale["value"].sum(),
        })
