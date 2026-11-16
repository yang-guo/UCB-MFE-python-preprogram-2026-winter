"""My own daily backtester for HW4: cash, lots, fills, and daily portfolio value.

Written for this homework. The overall shape (a strategy plans orders from the
previous close, an account books fills, the loop marks positions daily)
follows the ideas taught in Lecture 8 (Lectures/modeling/backtest.py), but the
code here is my own, including the stale/frozen-quote policy.

How one market session `day` works, in order:

1. PLAN    The strategy is asked for orders using the portfolio as of the
           previous session's close (`signal_date`). It never sees `day`'s
           prices. Orders are dollar budgets that already include the fee, and
           together they must fit inside the cash available at signal_date.
2. QUOTES  Read `day`'s adjusted closes. Held stocks without a quote keep their
           last observed price (a "stale" mark). After more than
           MAX_STALE_SESSIONS sessions without a quote, the stock is "frozen":
           still valued at its last price, but it cannot be sold. If quotes
           come back, it is unfrozen and treated normally again.
3. SELL    Every lot whose 90-calendar-day maturity has arrived is sold at
           `day`'s close. If maturity fell on a weekend or holiday, this is the
           next market session. If the stock has no quote, the sale waits for
           the next real quote and is flagged as delayed. No price is invented.
4. BUY     Each order is filled at `day`'s close. If the stock has no quote,
           the order is rejected and logged, and the cash stays in the account.
5. MARK    Record cash, holdings value, and NAV (net asset value: cash plus
           the market value of all open lots).

Prices are adjusted closes, so returns already include dividends. Dividends
are never added separately. Positions are fractional "adjusted-price units".
"""

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

import config
import data

CASH_TOLERANCE = 1e-6  # Dollars; absorbs floating-point dust only.


class InsufficientCashError(ValueError):
    """A purchase would cost more than the cash in the account."""


class OrderError(ValueError):
    """A strategy returned orders that break the rules (e.g. spend cash it does not have)."""


@dataclass(frozen=True)
class Lot:
    """One purchase. Each lot has its own price, fee, and maturity date."""

    lot_id: int
    symbol: str
    units: float
    signal_date: pd.Timestamp
    entry_date: pd.Timestamp
    entry_price: float
    entry_fee: float
    due_date: pd.Timestamp

    @property
    def cost(self):
        """Total cash paid for this lot, fee included."""
        return self.units * self.entry_price + self.entry_fee


@dataclass(frozen=True)
class PortfolioSnapshot:
    """A read-only view of the account that the strategy uses to plan orders."""

    date: pd.Timestamp
    cash: float
    nav: float
    holdings: pd.Series  # Market value by symbol (frozen stocks at their last price).


class Portfolio:
    """The account: cash, open lots, and a log of every fill."""

    def __init__(self, initial_cash=config.INITIAL_CAPITAL, fee_bps=config.FEE_BPS):
        if not initial_cash > 0 or not fee_bps >= 0:
            raise ValueError("Capital must be positive and fees nonnegative")
        self.initial_cash = float(initial_cash)
        self.cash = float(initial_cash)
        self.fee_rate = fee_bps / 10_000
        self.lots = {}
        self.fills = []
        self.realized_pnl = 0.0
        self._next_id = 0

    def buy(self, symbol, budget, price, date, signal_date, holding_days=config.HOLDING_DAYS):
        """Spend exactly `budget` (fee included) on `symbol` at `price`."""
        if not (np.isfinite(budget) and budget > 0 and np.isfinite(price) and price > 0):
            raise ValueError("Budget and price must be positive and finite")
        if not pd.Timestamp(signal_date) < pd.Timestamp(date):
            raise ValueError("The signal must come before the fill")
        units = budget / (price * (1 + self.fee_rate))  # Leave room for the fee.
        notional = units * price
        fee = notional * self.fee_rate
        if notional + fee > self.cash + CASH_TOLERANCE:
            raise InsufficientCashError(
                f"{symbol} on {pd.Timestamp(date).date()}: needs ${notional + fee:,.2f}, "
                f"only ${self.cash:,.2f} available")
        self.cash = max(0.0, self.cash - notional - fee)
        date = pd.Timestamp(date)
        lot = Lot(self._next_id, symbol, units, pd.Timestamp(signal_date), date, price, fee,
                  date + pd.Timedelta(days=holding_days))
        self.lots[lot.lot_id] = lot
        self._next_id += 1
        self._log(lot, "BUY", date, price, fee, -(notional + fee), 0.0, False)
        return lot

    def sell(self, lot_id, price, date, delayed=False):
        """Sell a whole lot at `price`; the proceeds go back into cash."""
        if not (np.isfinite(price) and price > 0):
            raise ValueError("Price must be positive and finite")
        lot = self.lots.pop(lot_id)
        notional = lot.units * price
        fee = notional * self.fee_rate
        proceeds = notional - fee
        pnl = proceeds - lot.cost
        self.cash += proceeds
        self.realized_pnl += pnl
        self._log(lot, "SELL", pd.Timestamp(date), price, fee, proceeds, pnl, delayed)
        return pnl

    def _log(self, lot, side, date, price, fee, cash_change, pnl, delayed):
        self.fills.append({"date": date, "side": side, "lot_id": lot.lot_id, "symbol": lot.symbol,
                           "units": lot.units, "price": price, "notional": lot.units * price,
                           "fee": fee, "cash_change": cash_change, "realized_pnl": pnl,
                           "signal_date": lot.signal_date, "due_date": lot.due_date,
                           "exit_delayed": delayed})

    def holdings(self, marks):
        """Market value by symbol, using each stock's last observed price."""
        values = {}
        for lot in self.lots.values():
            values[lot.symbol] = values.get(lot.symbol, 0.0) + lot.units * marks[lot.symbol]
        return pd.Series(values, dtype=float)

    def snapshot(self, date, marks):
        h = self.holdings(marks)
        return PortfolioSnapshot(pd.Timestamp(date), self.cash, self.cash + h.sum(), h)

    def reconcile(self, marks):
        """Check: starting cash + realized PnL + unrealized PnL == NAV."""
        unrealized = sum(lot.units * marks[lot.symbol] - lot.cost for lot in self.lots.values())
        nav = self.cash + self.holdings(marks).sum()
        expected = self.initial_cash + self.realized_pnl + unrealized
        if not np.isclose(nav, expected, rtol=0, atol=1e-6 * self.initial_cash):
            raise AssertionError(f"Accounting does not reconcile: NAV {nav} vs {expected}")
        return pd.Series({"cash": self.cash, "nav": nav, "realized_pnl": self.realized_pnl,
                          "unrealized_pnl": unrealized, "total_pnl": nav - self.initial_cash,
                          "total_fees": sum(f["fee"] for f in self.fills)})


@dataclass(frozen=True)
class BacktestResult:
    history: pd.DataFrame      # One row per session: cash, holdings, NAV, frozen value, ...
    fills: pd.DataFrame        # Every BUY and SELL.
    open_lots: pd.DataFrame    # Lots still held at the end (valued in the final NAV).
    diagnostics: pd.DataFrame  # stale_mark / frozen / unfrozen / exit_delayed / rejected_no_quote.
    summary: pd.Series         # Reconciled cash, NAV, PnL, fees.


class Backtester:
    """Advance one market session at a time and keep the books.

    prices:   long table with columns date, symbol, adj_close (gaps simply missing).
    calendar: market sessions (the benchmark's dates).
    strategy: any object with generate_orders(signal_date, execution_date, snapshot)
              returning a pd.Series of fee-inclusive dollar budgets indexed by symbol.
    """

    def __init__(self, prices, calendar, initial_cash=config.INITIAL_CAPITAL,
                 fee_bps=config.FEE_BPS, holding_days=config.HOLDING_DAYS,
                 max_stale_sessions=config.MAX_STALE_SESSIONS):
        self.calendar = pd.DatetimeIndex(calendar)
        if not (self.calendar.is_monotonic_increasing and self.calendar.is_unique):
            raise ValueError("Calendar must be sorted and unique")
        prices = prices[["date", "symbol", "adj_close"]].copy()
        prices["date"] = pd.to_datetime(prices.date)
        if prices.duplicated(["date", "symbol"]).any():
            raise ValueError("Duplicate (date, symbol) quotes")
        if not (np.isfinite(prices.adj_close) & prices.adj_close.gt(0)).all():
            raise ValueError("Quotes must be positive and finite; leave missing quotes out")
        if not prices.date.isin(self.calendar).all():
            raise ValueError("Every quote date must be a market session")
        # Rows = sessions, columns = symbols; NaN means "no quote that day".
        self.quotes = prices.pivot(index="date", columns="symbol", values="adj_close").reindex(self.calendar)
        self.initial_cash = initial_cash
        self.fee_bps = fee_bps
        self.holding_days = holding_days
        self.max_stale = max_stale_sessions
        self.portfolio = None  # Kept after a run (even a failed one) for inspection.

    @staticmethod
    def _validate_orders(orders, cash):
        if not isinstance(orders, pd.Series) or not orders.index.is_unique or orders.index.hasnans:
            raise OrderError("Orders must be a Series of budgets with unique symbols")
        if not np.isfinite(orders).all() or orders.lt(0).any():
            raise OrderError("Order budgets must be finite and nonnegative")
        if orders.sum() > cash + CASH_TOLERANCE:
            raise OrderError(f"Orders total ${orders.sum():,.2f} but only ${cash:,.2f} "
                             "cash was available at the signal close")

    def run(self, strategy, start, end, last_entry=None):
        """Trade from `start` to `end`. No new buys after `last_entry` (default: `end`)."""
        end = data.check_end(end)  # Same final-test lock as the data loader.
        start = pd.Timestamp(start)
        last_entry = end if last_entry is None else pd.Timestamp(last_entry)
        cal = self.calendar
        days = cal[(cal >= start) & (cal <= end)]
        if not len(days) or cal.get_loc(days[0]) == 0:
            raise ValueError("Need at least one session, plus one session before it for signals")

        pf = self.portfolio = Portfolio(self.initial_cash, self.fee_bps)
        last_price, last_seen = {}, {}  # Per held symbol: last quote and its session index.
        frozen, delayed_logged, diagnostics = set(), set(), []

        prev = cal[cal.get_loc(days[0]) - 1]
        snapshot = pf.snapshot(prev, last_price)
        history = [self._history_row(prev, pf, last_price, frozen, 0)]

        for day in days:
            i = cal.get_loc(day)
            signal_date = cal[i - 1]

            # 1. PLAN with yesterday's snapshot only.
            if day <= last_entry:
                orders = strategy.generate_orders(signal_date, day, snapshot)
                self._validate_orders(orders, snapshot.cash)
            else:
                orders = pd.Series(dtype=float)

            # 2. QUOTES: update held symbols, then classify any gaps.
            observed = self.quotes.iloc[i].dropna()
            held = sorted({lot.symbol for lot in pf.lots.values()})
            stale_today = 0
            for sym in held:
                if sym in observed.index:
                    last_price[sym], last_seen[sym] = observed[sym], i
                    if sym in frozen:
                        frozen.discard(sym)
                        diagnostics.append((day, sym, "unfrozen"))
                elif i - last_seen[sym] <= self.max_stale:
                    stale_today += 1
                    diagnostics.append((day, sym, "stale_mark"))
                elif sym not in frozen:
                    frozen.add(sym)
                    diagnostics.append((day, sym, "frozen"))

            # 3. SELL matured lots that have a real quote today.
            for lot in list(pf.lots.values()):
                if day < lot.due_date:
                    continue
                if lot.symbol in observed.index:
                    first_exit_session = cal[cal.searchsorted(lot.due_date)]
                    pf.sell(lot.lot_id, observed[lot.symbol], day, delayed=day > first_exit_session)
                elif lot.lot_id not in delayed_logged:
                    delayed_logged.add(lot.lot_id)
                    diagnostics.append((day, lot.symbol, "exit_delayed"))

            # 4. BUY at today's close.
            for sym, budget in orders.items():
                if budget <= CASH_TOLERANCE:
                    continue
                if sym not in observed.index:
                    diagnostics.append((day, sym, "rejected_no_quote"))
                    continue
                pf.buy(sym, budget, observed[sym], day, signal_date, self.holding_days)
                last_price[sym], last_seen[sym] = observed[sym], i

            # 5. MARK and record.
            snapshot = pf.snapshot(day, last_price)
            history.append(self._history_row(day, pf, last_price, frozen, stale_today))

        return BacktestResult(
            history=pd.DataFrame(history).set_index("date"),
            fills=pd.DataFrame(pf.fills),
            open_lots=pd.DataFrame([asdict(lot) for lot in pf.lots.values()],
                                   columns=list(Lot.__dataclass_fields__)),
            diagnostics=pd.DataFrame(diagnostics, columns=["date", "symbol", "event"]),
            summary=pf.reconcile(last_price),
        )

    @staticmethod
    def _history_row(day, pf, marks, frozen, stale_today):
        holdings = pf.holdings(marks)
        frozen_value = holdings[holdings.index.isin(frozen)].sum()
        nav = pf.cash + holdings.sum()
        return {"date": day, "cash": pf.cash, "holdings_value": holdings.sum(), "nav": nav,
                "frozen_value": frozen_value,
                "nav_frozen_at_zero": nav - frozen_value,  # Sensitivity: frozen lots worth $0.
                "open_lots": len(pf.lots), "stale_marks": stale_today}
