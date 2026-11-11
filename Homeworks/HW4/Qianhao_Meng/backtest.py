"""Long-only, daily-close backtester for HW4.

A strategy sizes orders at the signal close, the account books fills and keeps
the ledger, and run_backtest advances one market session at a time.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


INITIAL_CAPITAL = 1_000_000.0
HOLDING_DAYS = 90
FEE_RATE = 0.001
MIN_ORDER = 0.01
RISK_FREE = 0.02
STALE_SESSIONS = 5
FILL_COLUMNS = ["date", "side", "symbol", "price", "units", "fee", "cash_change",
                "signal_date", "entry_date", "due", "realized_pnl"]
DAILY_COLUMNS = ["cash", "holdings", "nav", "lots", "stale_value", "long_stale_value"]


@dataclass
class Lot:
    """One purchase with its own cost, maturity, and latest observed mark."""
    symbol: str
    units: float
    cost: float
    signal_date: pd.Timestamp
    entry_date: pd.Timestamp
    due: pd.Timestamp
    last_price: float
    last_session: int


@dataclass
class BacktestResult:
    daily: pd.DataFrame
    fills: pd.DataFrame
    events: pd.DataFrame
    open_lots: list


class TopNStrategy:
    """Split one sleeve of the portfolio equally among the top-ranked stocks.

    Rankings map each signal date to symbols ordered from best to worst. They
    must be prepared from information known at that date's close.
    """

    def __init__(self, rankings, top_n=20, sleeve_fraction=0.25):
        self.rankings = rankings
        self.top_n = top_n
        self.sleeve_fraction = sleeve_fraction

    def orders(self, signal_date, cash, nav):  # decide what to buy and how much
        """Return fee-inclusive budgets sized from the signal-close cash and NAV."""
        symbols = self.rankings.get(signal_date, [])[:self.top_n]
        if not symbols or cash <= 0:
            return []

        budget = min(cash, self.sleeve_fraction * nav) / len(symbols)
        return [(symbol, budget) for symbol in symbols]


class Account:
    """Cash, open lots, and the fill ledger."""

    def __init__(self, cash=INITIAL_CAPITAL, fee_rate=FEE_RATE):
        self.cash = cash
        self.fee_rate = fee_rate
        self.lots = []
        self.fills = []

    def buy(self, symbol, budget, price, date, signal_date, session):
        """Spend a budget that already includes the entry fee."""
        budget = min(budget, self.cash)
        # Skip floating-point dust left when the sleeves are fully invested.
        if budget < MIN_ORDER:
            return

        notional = budget / (1 + self.fee_rate)
        units, fee = notional / price, budget - notional
        due = date + pd.Timedelta(days=HOLDING_DAYS)

        # record the state of making that deal
        self.cash -= budget
        self.lots.append(Lot(symbol, units, budget, signal_date, date, due, price, session))
        self.fills.append((date, "buy", symbol, price, units, fee, -budget, signal_date, date, due, 0.0))

    def sell(self, lot, price, date):
        """Close a whole lot, pay the exit fee, and book its realized PnL."""
        notional = lot.units * price
        fee = notional * self.fee_rate
        self.cash += notional - fee
        self.lots.remove(lot)
        self.fills.append((date, "sell", lot.symbol, price, lot.units, fee, notional - fee,
                           lot.signal_date, lot.entry_date, lot.due, notional - fee - lot.cost))

    def snapshot(self, session):
        """Value open lots at their latest observed prices."""
        values = [(lot.units * lot.last_price, session - lot.last_session) for lot in self.lots]
        holdings = sum(value for value, _ in values)
        stale = sum(value for value, age in values if age > 0)
        long_stale = sum(value for value, age in values if age > STALE_SESSIONS)
        return self.cash, holdings, self.cash + holdings, len(self.lots), stale, long_stale


def _sell_due_lots(account, prices, date, previous_date, session, events):
    for lot in list(account.lots):
        price = prices.get(lot.symbol, np.nan)
        if pd.isna(price):
            # A stale mark is never a fill, so a due lot waits for its next quote.
            if session - lot.last_session == STALE_SESSIONS + 1:
                events.append((date, lot.symbol, "long_quote_gap"))
            continue
        lot.last_price, lot.last_session = price, session
        if date >= lot.due:
            if previous_date >= lot.due:
                events.append((date, lot.symbol, "delayed_sale"))
            account.sell(lot, price, date)


def _fill_orders(account, orders, prices, date, signal_date, session, events):
    for symbol, budget in orders:
        price = prices.get(symbol, np.nan)
        if pd.isna(price):
            events.append((date, symbol, "canceled_buy"))
            continue
        account.buy(symbol, budget, price, date, signal_date, session)


def run_backtest(quotes, strategy, first_signal, account=None):
    """Fill yesterday's orders at today's close and record the closing value.

    Quotes hold adjusted closes with sessions as rows and symbols as columns.
    A missing entry quote cancels that order. A missing exit quote delays the
    sale until a quote appears. Orders keep the budgets set at the signal
    close, so same-day sale proceeds only fund later orders.
    """
    account = Account() if account is None else account
    daily, events = [], []
    signal_date = first_signal
    orders = strategy.orders(signal_date, account.cash, account.cash)
    
    for session, (date, prices) in enumerate(quotes.iterrows()):
        _sell_due_lots(account, prices, date, signal_date, session, events)
        _fill_orders(account, orders, prices, date, signal_date, session, events)
        row = account.snapshot(session)
        daily.append(row)
        signal_date = date
        orders = strategy.orders(signal_date, account.cash, row[2])
    daily = pd.DataFrame(daily, columns=DAILY_COLUMNS, index=quotes.index)
    fills = pd.DataFrame(account.fills, columns=FILL_COLUMNS)
    events = pd.DataFrame(events, columns=["date", "symbol", "event"])
    return BacktestResult(daily, fills, events, account.lots)


def performance(nav, initial=INITIAL_CAPITAL, risk_free=RISK_FREE):
    """Use elapsed calendar time for CAGR and 252 sessions for daily Sharpe.

    The initial capital is the starting point, so first-day fees count in
    returns and drawdown.
    """
    values = np.r_[initial, nav.to_numpy()]
    returns = values[1:] / values[:-1] - 1
    peak = np.maximum.accumulate(values)
    daily_hurdle = (1 + risk_free) ** (1 / 252) - 1
    elapsed_years = (nav.index[-1] - nav.index[0]).days / 365.25
    risk = returns.std(ddof=1)
    return pd.Series({
        "final_value": values[-1],
        "cagr": (values[-1] / initial) ** (1 / elapsed_years) - 1,
        "max_drawdown": (values / peak - 1).min(),
        "sharpe": (returns.mean() - daily_hurdle) / risk * np.sqrt(252) if risk > 0 else np.nan,
    })
