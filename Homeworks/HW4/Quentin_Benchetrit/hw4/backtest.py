"""Backtester for a long-only book of 90-day lots, and the weekly strategy.

Order of events inside one session:
  1. read the day's closes; a holding with no quote keeps its last mark
  2. sell every lot that has reached maturity and has a quote today
  3. fill the orders that were decided at the previous close
  4. record cash, holdings and NAV
  5. show the closing state to the strategy, which may leave orders for tomorrow

Orders are dollar budgets fixed at the decision close, so the fill price is
never known when the decision is made, and a budget can only come from cash
that already existed at that close. Positions are held in units of adjusted
price: dividends are inside the price series and are not added again.
"""

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

HOLDING_DAYS = 90


@dataclass
class Lot:
    lot_id: int
    symbol: str
    units: float
    signal_date: pd.Timestamp
    entry_date: pd.Timestamp
    entry_price: float
    cost: float                 # cash paid, fee included
    due_date: pd.Timestamp
    last_price: float
    last_quote: pd.Timestamp
    sessions_without_quote: int = 0
    orphaned: bool = False      # quotes stopped for good; see Backtester

    @property
    def value(self):
        return self.units * self.last_price


@dataclass
class Snapshot:
    """What the strategy is allowed to see after the close."""
    date: pd.Timestamp
    cash: float
    nav: float
    holdings: dict              # symbol -> market value at the last mark


@dataclass
class Result:
    history: pd.DataFrame
    fills: pd.DataFrame
    open_lots: pd.DataFrame
    events: pd.DataFrame


class Backtester:
    """Daily loop over the market calendar.

    max_gap: a holding that has gone this many sessions without a quote is
    treated as gone (delisted, acquired, ticker reused). The file does not say
    what its holders received, so by default the lot simply stays on the books
    at its last observed close: it is never sold and frees no cash. Setting
    orphan_recovery to a number books that fraction of the last value as cash
    instead; I only use it to bracket the result (0 = total loss, 1 = paid out
    at the last close).
    """

    def __init__(self, market, strategy, initial_cash=1_000_000.0, fee_bps=10.0,
                 max_gap=63, orphan_recovery=None):
        self.calendar = market.calendar
        self.prices = market.adj_close.to_numpy()
        self.column = {symbol: j for j, symbol in enumerate(market.adj_close.columns)}
        self.strategy = strategy
        self.initial_cash = float(initial_cash)
        self.fee = fee_bps / 10_000
        self.max_gap = max_gap
        self.orphan_recovery = orphan_recovery

    def _book(self, date, lot, side, price, units, fee, cash_change, pnl=0.0):
        self.fills.append({"date": date, "signal_date": lot.signal_date, "lot_id": lot.lot_id,
                           "symbol": lot.symbol, "side": side, "price": price, "units": units,
                           "notional": units * price, "fee": fee, "cash_change": cash_change,
                           "due_date": lot.due_date, "pnl": pnl})

    def _buy(self, date, signal_date, symbol, budget, price):
        units = budget / (price * (1 + self.fee))
        fee = units * price * self.fee
        if budget > self.cash + 1e-6:
            raise RuntimeError(f"{date.date()}: order for {symbol} exceeds available cash")
        self.cash -= budget
        lot = Lot(self.next_lot, symbol, units, signal_date, date, price, budget,
                  date + pd.Timedelta(days=HOLDING_DAYS), price, date)
        self.next_lot += 1
        self.lots[lot.lot_id] = lot
        self._book(date, lot, "BUY", price, units, fee, -budget)

    def _sell(self, date, lot, price, side="SELL", fee_rate=None):
        fee_rate = self.fee if fee_rate is None else fee_rate
        fee = lot.units * price * fee_rate
        proceeds = lot.units * price - fee
        self.cash += proceeds
        del self.lots[lot.lot_id]
        self._book(date, lot, side, price, lot.units, fee, proceeds, proceeds - lot.cost)

    def _decide(self, date, next_date, holdings):
        """Ask the strategy for orders and check them against the cash on hand now."""
        nav = self.cash + sum(holdings.values())
        orders = self.strategy.orders(date, next_date, Snapshot(date, self.cash, nav, holdings))
        if any(budget <= 0 for budget in orders.values()) or sum(orders.values()) > self.cash + 1e-6:
            raise RuntimeError(f"{date.date()}: orders must be positive and covered by cash")
        return orders

    def run(self, start, end):
        calendar = self.calendar
        first = calendar.searchsorted(pd.Timestamp(start))
        last = calendar.searchsorted(pd.Timestamp(end), side="right") - 1
        if first < 1 or last < first:
            raise ValueError("need at least one session before the start date")

        self.cash, self.lots, self.fills, self.next_lot, events = self.initial_cash, {}, [], 0, []
        history = [{"date": calendar[first - 1], "cash": self.cash, "holdings": 0.0,
                    "nav": self.cash, "lots": 0, "stale_value": 0.0, "orphan_value": 0.0}]
        signal_date = calendar[first - 1]
        orders = self._decide(signal_date, calendar[first], {})

        for pos in range(first, last + 1):
            date, closes = calendar[pos], self.prices[pos]

            # 1. marks
            for lot in list(self.lots.values()):
                if lot.orphaned:
                    continue
                price = closes[self.column[lot.symbol]]
                if np.isfinite(price):
                    lot.last_price, lot.last_quote, lot.sessions_without_quote = price, date, 0
                    continue
                lot.sessions_without_quote += 1
                if lot.sessions_without_quote > self.max_gap:
                    lot.orphaned = True
                    events.append({"date": date, "symbol": lot.symbol, "lot_id": lot.lot_id,
                                   "event": "quotes stopped", "value": lot.value})
                    if self.orphan_recovery is not None:
                        self._sell(date, lot, lot.last_price * self.orphan_recovery,
                                   side="ASSUMED", fee_rate=0.0)

            # 2. sales at maturity, or at the first quote after it
            for lot in list(self.lots.values()):
                if date >= lot.due_date and lot.last_quote == date and not lot.orphaned:
                    self._sell(date, lot, lot.last_price)

            # 3. purchases decided yesterday
            for symbol, budget in orders.items():
                price = closes[self.column[symbol]]
                if not np.isfinite(price):
                    events.append({"date": date, "symbol": symbol, "lot_id": -1,
                                   "event": "no quote on entry day, order dropped", "value": budget})
                    continue
                self._buy(date, signal_date, symbol, budget, price)

            # 4. valuation
            holdings = {}
            stale = orphan = 0.0
            for lot in self.lots.values():
                holdings[lot.symbol] = holdings.get(lot.symbol, 0.0) + lot.value
                if lot.orphaned:
                    orphan += lot.value
                elif lot.sessions_without_quote:
                    stale += lot.value
            invested = sum(holdings.values())
            history.append({"date": date, "cash": self.cash, "holdings": invested,
                            "nav": self.cash + invested, "lots": len(self.lots),
                            "stale_value": stale, "orphan_value": orphan})
            assert self.cash > -1e-6

            # 5. decisions for the next session
            if pos < last:
                signal_date = date
                orders = self._decide(date, calendar[pos + 1], holdings)

        fill_columns = ["date", "signal_date", "lot_id", "symbol", "side", "price", "units",
                        "notional", "fee", "cash_change", "due_date", "pnl"]
        return Result(
            history=pd.DataFrame(history).set_index("date"),
            fills=pd.DataFrame(self.fills, columns=fill_columns),
            open_lots=pd.DataFrame([asdict(lot) for lot in self.lots.values()],
                                   columns=list(Lot.__dataclass_fields__)),
            events=pd.DataFrame(events, columns=["date", "symbol", "lot_id", "event", "value"]))


class WeeklyTopN:
    """Each week, split the free cash equally over the best-scored names.

    The first `tranches` weeks each deploy NAV / tranches, so the book is
    spread over thirteen weekly cohorts instead of one quarterly bet. After
    that a tranche reinvests whatever cash the maturing lots have released.
    A lot is sized at about NAV / (tranches x n_names); if a holiday makes two
    cohorts mature in the same week, twice as many names are bought rather
    than doubling each position. A name is skipped when buying it would push
    its weight above name_cap or its sector above sector_cap, and the next
    name on the list takes its place.
    """

    def __init__(self, scores, schedule, sectors, n_names=10, tranches=13,
                 name_cap=0.03, sector_cap=0.30, min_budget=1_000.0):
        self.scores = {date: group.sort_values(["prediction", "symbol"], ascending=[False, True])
                       .symbol.tolist() for date, group in scores.groupby("date")}
        self.entry_after = dict(zip(schedule.date, schedule.entry_date))
        self.sectors = dict(sectors)
        self.n_names, self.tranches = n_names, tranches
        self.name_cap, self.sector_cap, self.min_budget = name_cap, sector_cap, min_budget
        self.placed = 0

    def orders(self, date, next_date, portfolio):
        if self.entry_after.get(date) != next_date or date not in self.scores:
            return {}
        budget = portfolio.cash
        if self.placed < self.tranches:
            budget = min(budget, portfolio.nav / self.tranches)
        if budget < self.min_budget:
            return {}
        self.placed += 1

        usual_lot = portfolio.nav / (self.tranches * self.n_names)
        count = max(self.n_names, round(budget / usual_lot))
        per_name = budget / count
        by_sector = {}
        for symbol, value in portfolio.holdings.items():
            sector = self.sectors[symbol]
            by_sector[sector] = by_sector.get(sector, 0.0) + value
        chosen = {}
        for symbol in self.scores[date]:
            sector = self.sectors[symbol]
            if portfolio.holdings.get(symbol, 0.0) + per_name > self.name_cap * portfolio.nav:
                continue
            if by_sector.get(sector, 0.0) + per_name > self.sector_cap * portfolio.nav:
                continue
            chosen[symbol] = per_name
            by_sector[sector] = by_sector.get(sector, 0.0) + per_name
            if len(chosen) == count:
                break
        return chosen


def benchmark_nav(benchmark, history, initial_cash=1_000_000.0):
    """Buy the index at the first close of the run and hold it."""
    dates = history.index
    nav = initial_cash * benchmark.reindex(dates) / benchmark.loc[dates[1]]
    nav.iloc[0] = initial_cash
    return nav.rename("benchmark")


def performance(nav, risk_free=0.02):
    """Final value, CAGR, maximum drawdown and daily Sharpe ratio of a NAV series."""
    returns = nav.pct_change().dropna()
    years = (nav.index[-1] - nav.index[0]).days / 365.25
    daily_rf = (1 + risk_free) ** (1 / 252) - 1
    return pd.Series({
        "final_value": nav.iloc[-1],
        "total_return": nav.iloc[-1] / nav.iloc[0] - 1,
        "cagr": (nav.iloc[-1] / nav.iloc[0]) ** (1 / years) - 1,
        "max_drawdown": (nav / nav.cummax() - 1).min(),
        "volatility": returns.std() * np.sqrt(252),
        "sharpe": (returns - daily_rf).mean() / returns.std() * np.sqrt(252),
    })


def market_exposure(nav, benchmark_series, risk_free=0.02):
    """Beta and annualised alpha from a regression of daily excess returns."""
    daily_rf = (1 + risk_free) ** (1 / 252) - 1
    y = nav.pct_change().dropna() - daily_rf
    x = benchmark_series.pct_change().reindex(y.index) - daily_rf
    beta = np.cov(y, x)[0, 1] / x.var()
    return pd.Series({"beta": beta, "alpha": (y.mean() - beta * x.mean()) * 252,
                      "correlation": np.corrcoef(y, x)[0, 1]})
