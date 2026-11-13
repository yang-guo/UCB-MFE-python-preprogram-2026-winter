"""My long-only backtester for HW4.

I wrote this myself. The lecture's `Lectures/modeling/backtest.py` gave me the
general idea (separate lots, fees inside the dollar budget, a stale-price
limit), but this code is my own and works on a wide price table.

How one market session works (day t):

1. Read today's closes. They are used for exits, entries and marks.
2. Sell every lot whose 90 calendar days are up (due date <= t), at today's
   close. If the stock has no quote today, the lot waits for the next quote.
3. Buy the orders that were planned at the previous close, at today's close.
   Each buy is capped at the cash in the account, so the same cash can never
   be spent twice.
4. Value the account: cash + units x last observed close for every open lot.
5. If t is a signal date, ask the strategy for new orders. It only sees the
   account after today's close, and they fill at the *next* session's close.

Prices are adjusted closes, so one "unit" is total-return exposure and
dividends are never added a second time.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Lot:
    """One purchase. Each lot has its own price, fee, and due date."""

    lot_id: int
    symbol: str
    units: float
    signal_date: pd.Timestamp
    entry_date: pd.Timestamp
    entry_price: float
    entry_fee: float
    due_date: pd.Timestamp


@dataclass
class Account:
    """What the strategy is allowed to see when it plans orders."""

    date: pd.Timestamp
    cash: float
    nav: float
    holdings: pd.Series  # market value by symbol


@dataclass
class BacktestResult:
    daily: pd.DataFrame      # one row per session: cash, holdings, NAV, ...
    trades: pd.DataFrame     # every buy and sell
    open_lots: pd.DataFrame  # lots still held at the end
    events: pd.DataFrame     # skipped orders, stale prices, late exits

    @property
    def nav(self):
        return self.daily["nav"]


class Backtester:
    """Run a strategy day by day over a wide table of adjusted closes.

    prices: DataFrame, index = market sessions, columns = symbols,
        values = adjusted close. NaN means "no quote that day". Prices are
        never forward-filled.
    cost_bps: cost charged on every buy and every sale, in basis points.
    stale_limit: a held stock can go this many sessions without a quote
        while it is still valued at its last close.
    dark_policy: what to do once a held stock has gone longer than
        stale_limit with no quote. The data has no delisting proceeds, so I
        can't know the true value. "last" keeps the last close (optimistic),
        "zero" values it at 0 (pessimistic). I report both.
    """

    def __init__(self, prices, initial_cash=1_000_000.0, cost_bps=10.0,
                 holding_days=90, stale_limit=5, dark_policy="last"):
        if not isinstance(prices.index, pd.DatetimeIndex) or not prices.index.is_monotonic_increasing:
            raise ValueError("prices needs a sorted DatetimeIndex of market sessions")
        if prices.index.has_duplicates or prices.columns.has_duplicates:
            raise ValueError("prices has duplicate dates or symbols")
        values = prices.to_numpy(dtype=float)
        if (values[np.isfinite(values)] <= 0).any() or np.isinf(values).any():
            raise ValueError("prices must be positive; leave missing quotes as NaN")
        if dark_policy not in ("last", "zero"):
            raise ValueError("dark_policy must be 'last' or 'zero'")
        self.prices = prices
        self.calendar = prices.index
        self.initial_cash = float(initial_cash)
        self.cost = cost_bps / 10_000
        self.holding_days = holding_days
        self.stale_limit = stale_limit
        self.dark_policy = dark_policy

    def run(self, strategy, start, end):
        """strategy(account) -> Series of dollar budgets (fees included) by symbol.

        The strategy is called after the close of every session in the run
        except the last. It can return an empty Series to do nothing.
        """
        days = self.calendar[(self.calendar >= pd.Timestamp(start)) & (self.calendar <= pd.Timestamp(end))]
        if len(days) == 0:
            raise ValueError("No market sessions between start and end")
        first = self.calendar.get_loc(days[0])
        if first == 0:
            raise ValueError("Need one session before start to make the first decision")

        cash = self.initial_cash
        lots = {}
        next_id = 0
        trades, events, daily = [], [], []
        last_price, last_seen = {}, {}  # last observed close and its session number
        col = {s: i for i, s in enumerate(self.prices.columns)}
        values = self.prices.to_numpy(dtype=float)

        # The first decision is made at the close before `start`, with all cash.
        signal_day = self.calendar[first - 1]
        orders = self._check_orders(strategy(Account(signal_day, cash, cash, pd.Series(dtype=float))), cash)
        orders_signal = signal_day

        for day in days:
            i = self.calendar.get_loc(day)
            row = values[i]
            trades_before = len(trades)

            def quote(symbol):
                price = row[col[symbol]] if symbol in col else np.nan
                return price if np.isfinite(price) else None

            # 1. update marks for every stock we hold or are about to buy
            for symbol in {lot.symbol for lot in lots.values()} | set(orders.index):
                price = quote(symbol)
                if price is not None:
                    last_price[symbol], last_seen[symbol] = price, i

            # 2. exits: sell lots that are due, if there is a quote today
            for lot in sorted(lots.values(), key=lambda x: x.lot_id):
                if day < lot.due_date:
                    continue
                price = quote(lot.symbol)
                if price is None:
                    events.append({"date": day, "symbol": lot.symbol, "lot_id": lot.lot_id,
                                   "event": "exit_waiting_for_quote"})
                    continue
                proceeds = lot.units * price
                fee = proceeds * self.cost
                cash += proceeds - fee
                pnl = proceeds - fee - (lot.units * lot.entry_price + lot.entry_fee)
                trades.append(self._trade(day, lot, "SELL", price, fee, proceeds - fee, pnl))
                if day > lot.due_date + pd.Timedelta(days=4):
                    events.append({"date": day, "symbol": lot.symbol, "lot_id": lot.lot_id,
                                   "event": "late_exit"})
                del lots[lot.lot_id]

            # 3. entries planned at the previous close fill at today's close
            for symbol, budget in orders.items():
                price = quote(symbol)
                if price is None:
                    events.append({"date": day, "symbol": symbol, "lot_id": None,
                                   "event": "buy_skipped_no_quote"})
                    continue
                budget = min(budget, cash)  # never spend cash we don't have
                if budget <= 1e-6:
                    events.append({"date": day, "symbol": symbol, "lot_id": None,
                                   "event": "buy_skipped_no_cash"})
                    continue
                units = budget / (price * (1 + self.cost))
                fee = units * price * self.cost
                cash = max(0.0, cash - units * price - fee)
                lot = Lot(next_id, symbol, units, orders_signal, day, price, fee,
                          day + pd.Timedelta(days=self.holding_days))
                lots[next_id] = lot
                next_id += 1
                trades.append(self._trade(day, lot, "BUY", price, fee, -(units * price + fee), 0.0))

            # 4. value the account at today's close
            holdings = {}
            stale = dark = 0
            dark_value = 0.0
            for lot in lots.values():
                gap = i - last_seen[lot.symbol]
                value = lot.units * last_price[lot.symbol]
                if gap > 0:
                    stale += 1
                if gap > self.stale_limit:
                    dark += 1
                    dark_value += value
                    if self.dark_policy == "zero":
                        value = 0.0
                holdings[lot.symbol] = holdings.get(lot.symbol, 0.0) + value
            holdings = pd.Series(holdings, dtype=float)
            nav = cash + holdings.sum()
            day_trades = trades[trades_before:]
            daily.append({"date": day, "cash": cash, "holdings_value": holdings.sum(), "nav": nav,
                          "open_lots": len(lots), "stale_lots": stale, "dark_lots": dark,
                          "dark_value_at_last_price": dark_value,
                          "fees": sum(t["fee"] for t in day_trades),
                          "traded": sum(t["notional"] for t in day_trades)})

            # 5. plan the next session's orders using only today's close
            if day != days[-1]:
                orders = self._check_orders(strategy(Account(day, cash, nav, holdings)), cash)
                orders_signal = day
            else:
                orders = pd.Series(dtype=float)

        open_lots = pd.DataFrame([vars(lot) for lot in lots.values()],
                                 columns=list(Lot.__dataclass_fields__))
        if len(open_lots):
            open_lots["last_price"] = open_lots.symbol.map(last_price)
            open_lots["last_quote_date"] = open_lots.symbol.map(lambda s: self.calendar[last_seen[s]])
            open_lots["market_value"] = open_lots.units * open_lots.last_price
        return BacktestResult(
            daily=pd.DataFrame(daily).set_index("date"),
            trades=pd.DataFrame(trades, columns=["date", "signal_date", "lot_id", "symbol", "side", "price",
                                                 "units", "notional", "fee", "cash_change", "due_date", "pnl"]),
            open_lots=open_lots,
            events=pd.DataFrame(events, columns=["date", "symbol", "lot_id", "event"]),
        )

    @staticmethod
    def _check_orders(orders, cash):
        if orders is None:
            return pd.Series(dtype=float)
        if not isinstance(orders, pd.Series) or not orders.index.is_unique:
            raise ValueError("orders must be a Series of budgets with one row per symbol")
        if not np.isfinite(orders).all() or (orders < 0).any():
            raise ValueError("order budgets must be finite and nonnegative")
        if orders.sum() > cash + 1e-6:
            raise ValueError("orders ask for more cash than the account had at the signal close")
        return orders[orders > 0]

    @staticmethod
    def _trade(day, lot, side, price, fee, cash_change, pnl):
        return {"date": day, "signal_date": lot.signal_date, "lot_id": lot.lot_id, "symbol": lot.symbol,
                "side": side, "price": price, "units": lot.units, "notional": lot.units * price,
                "fee": fee, "cash_change": cash_change, "due_date": lot.due_date, "pnl": pnl}


def performance(nav, initial_cash=1_000_000.0, annual_rf=0.02):
    """Final value, CAGR, max drawdown and Sharpe from a daily NAV series.

    The first daily return is measured from the starting cash, so fees paid on
    the first day count. CAGR uses calendar time (days / 365.25); Sharpe uses
    252 sessions a year and a 2% annual risk-free rate turned into a daily rate.
    """
    nav = nav.sort_index()
    values = pd.concat([pd.Series([initial_cash]), pd.Series(nav.to_numpy())], ignore_index=True)
    returns = values.pct_change().dropna()
    daily_rf = (1 + annual_rf) ** (1 / 252) - 1
    years = (nav.index[-1] - nav.index[0]).days / 365.25
    excess = returns - daily_rf
    return pd.Series({
        "start": nav.index[0].date(),
        "end": nav.index[-1].date(),
        "final_value": nav.iloc[-1],
        "total_return": nav.iloc[-1] / initial_cash - 1,
        "CAGR": (nav.iloc[-1] / initial_cash) ** (1 / years) - 1,
        "annual_volatility": returns.std(ddof=1) * np.sqrt(252),
        "max_drawdown": (values / values.cummax() - 1).min(),
        "Sharpe": excess.mean() / returns.std(ddof=1) * np.sqrt(252),
    })


def benchmark_nav(benchmark, start, end, initial_cash=1_000_000.0):
    """$1M bought at the first close on/after `start`, held through `end`."""
    b = benchmark.loc[pd.Timestamp(start):pd.Timestamp(end)]
    return initial_cash * b / b.iloc[0]
