"""A small daily backtester: a portfolio ledger plus a session-by-session loop.

Written for HW4. The overall split into a strategy that plans orders, a
ledger that books fills, a daily loop, and analytics follows the structure
taught in Lecture 8 (Lectures/modeling/backtest.py); the code is my own.

Timeline for every market session d (signal session t = d - 1):

    close of t  : strategy sees prices/features through t and the portfolio
                  valued at t; it returns fee-inclusive dollar budgets whose
                  total is <= cash at t.
    close of d  : 1) lots whose 90-calendar-day maturity is <= d are sold at
                     d's close (or at the next session with a quote);
                  2) planned buys fill at d's close (rejected if no quote);
                  3) every open lot is marked at its latest quote -> NAV.

Positions are fractional units of the *adjusted* close, so dividends are
already inside returns and are never added again.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class Lot:
    lot_id: int
    n: int                    # column index of the stock
    symbol: str
    units: float
    signal_date: pd.Timestamp
    entry_date: pd.Timestamp
    entry_price: float
    entry_fee: float
    due_date: pd.Timestamp
    frozen_mark: float = float("nan")   # set if the stock's quotes become unreliable
    frozen_last_quote: object = None     # last trusted quote date of a frozen lot


@dataclass
class Snapshot:
    """What the strategy may see at the signal close."""
    date: pd.Timestamp
    cash: float
    nav: float
    holdings: dict            # n -> market value at the signal close
    open_lots: list = field(default_factory=list)


class Ledger:
    """Cash, open lots and a fill log. Every cash movement goes through here."""

    def __init__(self, initial_cash=1_000_000.0, fee_bps=10.0):
        self.initial_cash = float(initial_cash)
        self.cash = float(initial_cash)
        self.fee = fee_bps / 10_000
        self.lots = {}
        self.fills = []
        self._next_id = 0

    def buy(self, n, symbol, budget, price, date, signal_date, holding_days):
        """Spend a fee-inclusive budget: units * price * (1 + fee) == budget."""
        if not (np.isfinite(price) and price > 0 and budget > 0):
            raise ValueError("Buy needs a positive price and budget")
        if not signal_date < date:
            raise ValueError("Signal must precede the fill")
        if budget > self.cash + 1e-6:
            raise ValueError(f"Insufficient cash: budget {budget:.2f} > cash {self.cash:.2f}")
        units = budget / (price * (1 + self.fee))
        notional = units * price
        fee = notional * self.fee
        self.cash -= notional + fee
        if abs(self.cash) < 1e-6:
            self.cash = 0.0
        lot = Lot(self._next_id, n, symbol, units, signal_date, date, price, fee,
                  date + pd.Timedelta(days=holding_days))
        self.lots[lot.lot_id] = lot
        self._next_id += 1
        self._log(lot, "BUY", date, price, notional, fee, -(notional + fee), 0.0)
        return lot

    def sell(self, lot_id, price, date):
        lot = self.lots.pop(lot_id)
        if not (np.isfinite(price) and price > 0) or date < lot.entry_date:
            raise ValueError("Invalid sale")
        notional = lot.units * price
        fee = notional * self.fee
        self.cash += notional - fee
        pnl = notional - fee - lot.units * lot.entry_price - lot.entry_fee
        self._log(lot, "SELL", date, price, notional, fee, notional - fee, pnl)
        return pnl

    def _log(self, lot, side, date, price, notional, fee, cash_change, pnl):
        self.fills.append({"date": date, "side": side, "lot_id": lot.lot_id, "symbol": lot.symbol,
                           "signal_date": lot.signal_date, "entry_date": lot.entry_date,
                           "due_date": lot.due_date, "price": price, "units": lot.units,
                           "notional": notional, "fee": fee, "cash_change": cash_change,
                           "realized_pnl": pnl, "cash_after": self.cash})

    def holdings_value(self, marks):
        values = {}
        for lot in self.lots.values():
            price = lot.frozen_mark if np.isfinite(lot.frozen_mark) else marks[lot.n]
            values[lot.n] = values.get(lot.n, 0.0) + lot.units * price
        return values


@dataclass
class BacktestResult:
    daily: pd.DataFrame       # cash, holdings, nav, stale marks, ... per session
    fills: pd.DataFrame
    open_lots: pd.DataFrame   # lots still open at the end (valued in final NAV)
    events: pd.DataFrame      # rejected orders, late exits, stale marks
    ledger: Ledger
    frozen: pd.DataFrame      # lots frozen after a long quote gap


class Backtester:
    def __init__(self, data, strategy, fee_bps=10.0, initial_cash=1_000_000.0,
                 holding_days=90, stale_warn_sessions=5, max_quote_gap=21):
        self.data = data
        self.strategy = strategy
        self.ledger = Ledger(initial_cash, fee_bps)
        self.holding_days = holding_days
        self.stale_warn = stale_warn_sessions
        # A held stock that reappears after > max_quote_gap sessions without a
        # quote is not trusted: tickers get reused (e.g. TE after TECO Energy's
        # 2016 takeover) and there is no delisting-proceeds table. Such a lot is
        # frozen at its last pre-gap price and never sold at the new quotes.
        self.max_gap = max_quote_gap

    def run(self, start, end):
        data, ledger = self.data, self.ledger
        cal = data.calendar
        days = np.flatnonzero((cal >= pd.Timestamp(start)) & (cal <= pd.Timestamp(end)))
        if not len(days) or days[0] == 0:
            raise ValueError("Need sessions and a preceding signal session")
        N = len(data.symbols)
        marks = np.full(N, np.nan)          # last observed adjusted close
        last_seen = np.full(N, -1)
        # Marks at the first signal close (nothing held yet, kept for completeness).
        first_t = days[0] - 1
        seen = np.isfinite(data.adj_close[first_t])
        marks[seen], last_seen[seen] = data.adj_close[first_t, seen], first_t
        rows, events, frozen = [], [], []
        nav_prev = ledger.cash
        holdings_prev = {}

        for d in days:
            t = d - 1
            date, signal_date = cal[d], cal[t]
            # ---- decision at the close of t (no prices from d are visible) ----
            snapshot = Snapshot(signal_date, ledger.cash, nav_prev, dict(holdings_prev),
                                list(ledger.lots.values()))
            orders = self.strategy.plan(t, snapshot)
            planned = sum(orders.values())
            if planned > snapshot.cash + 1e-6:
                raise ValueError(f"{signal_date.date()}: orders {planned:.2f} exceed signal-date cash")

            # ---- the close of d ----
            quotes = data.adj_close[d]
            seen = np.isfinite(quotes)
            for lot in ledger.lots.values():
                if (seen[lot.n] and not np.isfinite(lot.frozen_mark)
                        and d - last_seen[lot.n] > self.max_gap):
                    lot.frozen_mark = float(marks[lot.n])
                    lot.frozen_last_quote = cal[last_seen[lot.n]]
                    frozen.append({"date": date, "lot_id": lot.lot_id, "symbol": lot.symbol,
                                   "frozen_mark": lot.frozen_mark, "last_quote": cal[last_seen[lot.n]]})
                    events.append({"date": date, "symbol": lot.symbol, "event": "frozen_after_quote_gap",
                                   "detail": f"no quote from {cal[last_seen[lot.n]].date()} to {date.date()}; "
                                             f"kept at {lot.frozen_mark:.2f}, not sold at {quotes[lot.n]:.2f}"})
            marks[seen], last_seen[seen] = quotes[seen], d
            sold = 0
            for lot in list(ledger.lots.values()):
                if lot.due_date <= date:
                    if np.isfinite(quotes[lot.n]) and not np.isfinite(lot.frozen_mark):
                        ledger.sell(lot.lot_id, float(quotes[lot.n]), date)
                        sold += 1
                        late = (date - lot.due_date).days
                        if late > 4:
                            events.append({"date": date, "symbol": lot.symbol, "event": "late_exit",
                                           "detail": f"due {lot.due_date.date()}, sold {late} days late"})
                    elif not np.isfinite(lot.frozen_mark) and (
                            (date - lot.due_date).days <= 1 or (d - last_seen[lot.n]) % 21 == 0):
                        events.append({"date": date, "symbol": lot.symbol, "event": "exit_waiting_for_quote",
                                       "detail": f"due {lot.due_date.date()}, last quote {cal[last_seen[lot.n]].date()}"})
            bought = 0
            for n, budget in orders.items():
                if budget <= 0:
                    continue
                if not np.isfinite(quotes[n]):
                    events.append({"date": date, "symbol": data.symbols[n], "event": "buy_rejected_no_quote",
                                   "detail": f"budget {budget:,.0f} stays in cash"})
                    continue
                ledger.buy(n, data.symbols[n], budget, float(quotes[n]), date, signal_date, self.holding_days)
                bought += 1

            # ---- mark to market ----
            held = {lot.n for lot in ledger.lots.values()}
            stale = [n for n in held if last_seen[n] < d or any(
                np.isfinite(l.frozen_mark) for l in ledger.lots.values() if l.n == n)]
            for n in stale:
                if d - last_seen[n] == self.stale_warn + 1:
                    events.append({"date": date, "symbol": data.symbols[n], "event": "stale_mark",
                                   "detail": f"no quote since {cal[last_seen[n]].date()}"})
            holdings_prev = ledger.holdings_value(marks)
            hv = sum(holdings_prev.values())
            nav_prev = ledger.cash + hv
            rows.append({"date": date, "cash": ledger.cash, "holdings": hv, "nav": nav_prev,
                         "open_lots": len(ledger.lots), "names": len(held), "buys": bought, "sells": sold,
                         "stale_names": len(stale)})

        daily = pd.DataFrame(rows).set_index("date")
        open_lots = pd.DataFrame([vars(l) | {
            "last_mark": l.frozen_mark if np.isfinite(l.frozen_mark) else marks[l.n],
            "last_quote": l.frozen_last_quote if l.frozen_last_quote is not None else cal[last_seen[l.n]]}
            for l in ledger.lots.values()])
        return BacktestResult(daily, pd.DataFrame(ledger.fills), open_lots,
                              pd.DataFrame(events, columns=["date", "symbol", "event", "detail"]), ledger,
                              pd.DataFrame(frozen, columns=["date", "lot_id", "symbol", "frozen_mark", "last_quote"]))


def performance(nav, annual_rf=0.02):
    """Final value, CAGR (actual days / 365.25), max drawdown, Sharpe on daily
    returns versus a 2% annual risk-free rate converted to a daily rate."""
    nav = nav.dropna()
    returns = nav.pct_change().dropna()
    years = (nav.index[-1] - nav.index[0]).days / 365.25
    rf_daily = (1 + annual_rf) ** (1 / 252) - 1
    excess = returns - rf_daily
    return pd.Series({
        "start": nav.index[0].date(), "end": nav.index[-1].date(),
        "final_value": nav.iloc[-1],
        "total_return": nav.iloc[-1] / nav.iloc[0] - 1,
        "CAGR": (nav.iloc[-1] / nav.iloc[0]) ** (1 / years) - 1,
        "max_drawdown": (nav / nav.cummax() - 1).min(),
        "annual_volatility": returns.std(ddof=1) * np.sqrt(252),
        "Sharpe": np.sqrt(252) * excess.mean() / excess.std(ddof=1),
    })


def benchmark_nav(data, start, end, initial=1_000_000.0):
    """$1M invested in ^SP500TR at the first test close, held to the end."""
    b = data.benchmark.loc[pd.Timestamp(start):pd.Timestamp(end)]
    return initial * b / b.iloc[0]
