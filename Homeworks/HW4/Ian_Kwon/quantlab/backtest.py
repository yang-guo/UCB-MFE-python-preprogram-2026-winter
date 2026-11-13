"""Strategy-independent daily backtester.

Design follows the Lecture 8 framework (``Lectures/modeling/backtest.py``):
separate Strategy / OrderManagementSystem / Backtester / analytics roles,
fee-inclusive budgets ``q = B / (P (1 + c))``, one lot per purchase, sales at
the first quoted close on/after a 90-calendar-day maturity, fractional
adjusted-price units (total-return exposure; dividends are never added).

Written from scratch for HW4 with these differences:

* **Point-in-time view.** A strategy receives a *read-only numpy view* of
  prices through the signal close.  Writing to it raises; rows after the
  signal date are not in it.  (The lecture copies the history every day,
  which is too slow for 3,849 symbols x 2,600 sessions.)
* **Long quote gaps** (decided with the user): a held lot whose symbol has not
  quoted for more than ``max_stale_sessions`` is *frozen*: it stays in NAV at
  its last observed mark, its cash stays locked, and it resumes normally if
  quotes return.  No exit fill is invented.  ``on_long_gap='halt'`` restores
  the lecture behaviour (raise).
* ``on_long_gap='settle_at_maturity'`` (accounting sensitivity): a lot that
  is frozen when it matures is settled at its last observed mark, so the cash
  is recycled.  The fill is labelled ``settlement='stale_mark'``.  This
  approximates a cash buyout at the last traded price (most frozen lots in
  this database are acquisitions) and invents a fill at a stale price.
* **Ticker reuse guard.** If quotes return only after more than
  ``max_resume_gap_sessions`` (default 126, ~6 months), the lot becomes
  *terminally frozen*: it keeps its last mark to the end and is never sold,
  because the new quotes most likely belong to a different security reusing
  the ticker (e.g. SFD: cash buyout at ~34 USD on 2013-09-26, a new Smithfield
  listing under SFD in 2025).
* **Holding period enforced in the OMS**, not only in the strategy.
* Vectorized marks: aggregate units per symbol x last observed mark.

Timeline for each market session d (signal date s = previous session):
  1. strategy plans orders from the snapshot at the close of s and prices <= s
  2. today's quotes are read
  3. matured lots with a quote today are sold (cash is booked immediately)
  4. planned purchases with a quote today are bought at today's close
  5. holdings are marked (stale/frozen bookkeeping) and NAV is recorded
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

EPS = 1e-8


class HeldPriceGapError(ValueError):
    """Raised under on_long_gap='halt' when a holding stops quoting too long."""


@dataclass(frozen=True)
class Lot:
    lot_id: int
    symbol: str
    col: int
    units: float
    signal_date: pd.Timestamp
    entry_date: pd.Timestamp
    entry_price: float
    entry_fee: float
    due_date: pd.Timestamp


@dataclass(frozen=True)
class PortfolioSnapshot:
    """Detached account view at a close (what a strategy may plan with)."""

    date: pd.Timestamp
    cash: float
    nav: float
    holdings: pd.Series          # market value by symbol (aggregated across lots)
    fee_rate: float
    frozen_symbols: frozenset = frozenset()


class PointInTimeView:
    """Prices through ``asof`` only.  Read-only; later rows are absent."""

    def __init__(self, calendar: pd.DatetimeIndex, symbols: pd.Index, px: np.ndarray, asof_index: int):
        self.calendar = calendar[: asof_index + 1]
        self.symbols = symbols
        view = px[: asof_index + 1]
        view = view.view()
        view.flags.writeable = False
        self.prices = view
        self.asof = self.calendar[-1]

    def window(self, sessions: int) -> pd.DataFrame:
        """Last ``sessions`` rows as a DataFrame copy (convenient for small strategies)."""
        return pd.DataFrame(self.prices[-sessions:], index=self.calendar[-sessions:], columns=self.symbols)


@dataclass(frozen=True)
class DecisionContext:
    signal_date: pd.Timestamp
    execution_date: pd.Timestamp
    portfolio: PortfolioSnapshot
    view: PointInTimeView


class Strategy(ABC):
    """Every strategy implements ``generate_orders``; the engine does the rest."""

    holding_days: int = 90

    @abstractmethod
    def generate_orders(self, ctx: DecisionContext) -> pd.Series:
        """Fee-inclusive dollar budgets indexed by symbol, for the next close."""

    def should_sell(self, lot: Lot, date: pd.Timestamp) -> bool:
        return date >= lot.due_date


class OrderManagementSystem:
    """Cash, lots, fees and PnL.  Knows nothing about strategies or models."""

    FILL_COLUMNS = ["date", "signal_date", "lot_id", "symbol", "side", "price", "units", "notional",
                    "fee", "cash_change", "due_date", "realized_pnl", "settlement"]

    def __init__(self, initial_cash=1_000_000.0, fee_bps=5.0, enforce_holding_period=True):
        if not np.isfinite([initial_cash, fee_bps]).all() or initial_cash <= 0 or fee_bps < 0:
            raise ValueError("Capital must be positive and fees nonnegative")
        self.initial_cash = float(initial_cash)
        self.cash = float(initial_cash)
        self.fee_rate = fee_bps / 10_000
        self.enforce_holding_period = enforce_holding_period
        self.lots: dict[int, Lot] = {}
        self.fills: list[dict] = []
        self.closed: list[dict] = []
        self.total_fees = 0.0
        self.realized_pnl = 0.0
        self._next_id = 0

    def size_for_budget(self, budget: float, price: float) -> float:
        if not np.isfinite([budget, price]).all() or budget < 0 or price <= 0:
            raise ValueError("Budget must be nonnegative and price positive")
        return budget / (price * (1 + self.fee_rate))

    def buy(self, units, price, *, symbol, col, date, signal_date, holding_days) -> int:
        if not np.isfinite([units, price]).all() or units <= 0 or price <= 0:
            raise ValueError("Units and price must be finite and positive")
        date, signal_date = pd.Timestamp(date), pd.Timestamp(signal_date)
        if not signal_date < date:
            raise ValueError("A buy needs a signal date strictly before its fill date")
        notional = units * price
        fee = notional * self.fee_rate
        if notional + fee > self.cash + EPS:
            raise ValueError(f"Insufficient cash: need {notional + fee:.2f}, have {self.cash:.2f}")
        lot = Lot(self._next_id, symbol, col, units, signal_date, date, price, fee,
                  date + pd.Timedelta(days=holding_days))
        self._next_id += 1
        self.cash = max(0.0, self.cash - notional - fee)  # clear floating-point dust only
        self.lots[lot.lot_id] = lot
        self._record(lot, "BUY", date, units, price, fee, -(notional + fee), 0.0)
        return lot.lot_id

    def sell_lot(self, lot_id, price, *, date, settlement="market") -> float:
        """Close a whole lot at ``price`` (the engine never sells partial lots)."""
        lot = self.lots[lot_id]
        date = pd.Timestamp(date)
        if not np.isfinite(price) or price <= 0:
            raise ValueError("Sale price must be finite and positive")
        if date <= lot.entry_date:
            raise ValueError("A sale must follow its purchase")
        if self.enforce_holding_period and date < lot.due_date:
            raise ValueError(f"Lot {lot_id} sold before its {lot.due_date.date()} maturity")
        notional = lot.units * price
        fee = notional * self.fee_rate
        pnl = notional - fee - lot.units * lot.entry_price - lot.entry_fee
        self.cash += notional - fee
        self.realized_pnl += pnl
        del self.lots[lot_id]
        self._record(lot, "SELL", date, lot.units, price, fee, notional - fee, pnl, settlement)
        self.closed.append({**asdict(lot), "exit_date": date, "exit_price": price, "exit_fee": fee,
                            "settlement": settlement,
                            "net_return": pnl / (lot.units * lot.entry_price + lot.entry_fee)})
        return pnl

    def _record(self, lot, side, date, units, price, fee, cash_change, pnl, settlement="market"):
        self.total_fees += fee
        self.fills.append({"date": date, "signal_date": lot.signal_date, "lot_id": lot.lot_id,
                           "symbol": lot.symbol, "side": side, "price": price, "units": units,
                           "notional": units * price, "fee": fee, "cash_change": cash_change,
                           "due_date": lot.due_date, "realized_pnl": pnl, "settlement": settlement})


@dataclass(frozen=True)
class BacktestResult:
    history: pd.DataFrame       # one row per session (+ initial-capital row)
    fills: pd.DataFrame
    lots_closed: pd.DataFrame
    open_lots: pd.DataFrame     # at the end date, with final marks and frozen state
    events: pd.DataFrame        # rejected orders, stale marks, freezes, resumptions
    pnl: pd.Series
    settings: dict = field(default_factory=dict)


class Backtester:
    def __init__(self, calendar: pd.DatetimeIndex, symbols: pd.Index, px: np.ndarray, strategy: Strategy,
                 oms: OrderManagementSystem | None = None, max_stale_sessions: int = 5,
                 on_long_gap: str = "freeze", max_resume_gap_sessions: int = 126):
        if px.shape != (len(calendar), len(symbols)):
            raise ValueError("Price matrix must be sessions x symbols")
        finite = np.isfinite(px)
        if (px[finite] <= 0).any():
            raise ValueError("Prices must be positive; use NaN for missing quotes")
        if not calendar.is_monotonic_increasing or calendar.has_duplicates:
            raise ValueError("Calendar must be strictly increasing")
        if on_long_gap not in ("freeze", "halt", "settle_at_maturity"):
            raise ValueError("on_long_gap must be 'freeze', 'halt' or 'settle_at_maturity'")
        if not isinstance(strategy, Strategy):
            raise TypeError("strategy must implement quantlab.backtest.Strategy")
        self.calendar, self.symbols, self.px = calendar, symbols, px
        self.strategy = strategy
        self.oms = oms or OrderManagementSystem()
        if self.oms.fills or self.oms.lots or self.oms.cash != self.oms.initial_cash:
            raise ValueError("Use a fresh OMS for each run")
        self.max_stale = max_stale_sessions
        self.on_long_gap = on_long_gap
        self.max_resume_gap = max_resume_gap_sessions
        self.terminal: dict[int, float] = {}  # lot_id -> frozen mark (never revived)
        self._col = {s: i for i, s in enumerate(symbols)}
        self._ran = False

    # ---------------------------------------------------------------- helpers
    def _terminal_values(self) -> pd.Series:
        vals = {}
        for lot_id, mark in self.terminal.items():
            lot = self.oms.lots[lot_id]
            vals[lot.symbol] = vals.get(lot.symbol, 0.0) + lot.units * mark
        return pd.Series(vals, dtype=float)

    def _snapshot(self, date, units, marks, frozen) -> PortfolioSnapshot:
        held = units > 0
        values = units[held] * marks[held]
        holdings = pd.Series(values, index=self.symbols[held], dtype=float)
        if self.terminal:
            holdings = holdings.add(self._terminal_values(), fill_value=0.0)
        nav = self.oms.cash + float(holdings.sum())
        return PortfolioSnapshot(pd.Timestamp(date), self.oms.cash, nav, holdings, self.oms.fee_rate,
                                 frozenset(self.symbols[frozen & held]))

    @staticmethod
    def _validate_orders(orders, cash):
        if not isinstance(orders, pd.Series):
            raise TypeError("Strategies must return a pandas Series of budgets")
        if not orders.index.is_unique or orders.index.hasnans:
            raise ValueError("Order symbols must be unique and non-missing")
        if len(orders) and (not np.isfinite(orders.to_numpy(float)).all() or (orders < 0).any()):
            raise ValueError("Budgets must be finite and nonnegative")
        if orders.sum() > cash + 1e-6:
            raise ValueError(f"Budgets {orders.sum():.2f} exceed signal-close cash {cash:.2f}")

    # -------------------------------------------------------------------- run
    def run(self, start, end) -> BacktestResult:
        if self._ran:
            raise RuntimeError("Create a new Backtester (and OMS) for each run")
        self._ran = True
        cal = self.calendar
        days = np.flatnonzero((cal >= pd.Timestamp(start)) & (cal <= pd.Timestamp(end)))
        if not len(days) or days[0] == 0:
            raise ValueError("Need sessions in [start, end] and one preceding signal session")
        N = len(self.symbols)
        units = np.zeros(N)                 # aggregate units per symbol
        marks = np.full(N, np.nan)          # last observed price per symbol
        last_seen = np.full(N, -10**9)      # session index of last quote
        frozen = np.zeros(N, bool)
        events, history = [], []
        snap = self._snapshot(cal[days[0] - 1], units, np.nan_to_num(marks), frozen)
        history.append({"date": cal[days[0] - 1], "cash": snap.cash, "holdings_value": 0.0, "nav": snap.nav,
                        "open_lots": 0, "fees": 0.0, "bought": 0.0, "sold": 0.0, "stale_symbols": 0,
                        "frozen_symbols": 0, "frozen_value": 0.0})
        for i in days:
            day, signal = cal[i], cal[i - 1]
            view = PointInTimeView(cal, self.symbols, self.px, i - 1)
            orders = self.strategy.generate_orders(DecisionContext(signal, day, snap, view))
            self._validate_orders(orders, snap.cash)

            quote = self.px[i]
            seen = np.isfinite(quote)
            # ticker-reuse guard: quotes returning after a very long gap do not revive old lots
            for c in np.flatnonzero(seen & frozen & (i - last_seen > self.max_resume_gap)):
                for lot in [l for l in self.oms.lots.values() if l.col == c and l.lot_id not in self.terminal]:
                    self.terminal[lot.lot_id] = marks[c]
                    units[c] -= lot.units
                    events.append({"date": day, "symbol": self.symbols[c], "event": "terminal_freeze_after_long_gap",
                                   "value": lot.units * marks[c], "lot_id": lot.lot_id})
                frozen[c] = False
            units[np.abs(units) < 1e-12] = 0.0
            marks[seen] = quote[seen]
            last_seen[seen] = i
            fills_before = len(self.oms.fills)

            # exits first: sale proceeds are booked today (usable at the next decision)
            for lot in sorted(self.oms.lots.values(), key=lambda l: l.lot_id):
                if lot.lot_id in self.terminal:
                    continue
                if seen[lot.col] and self.strategy.should_sell(lot, day):
                    units[lot.col] -= lot.units
                    self.oms.sell_lot(lot.lot_id, quote[lot.col], date=day)
                elif (self.on_long_gap == "settle_at_maturity" and frozen[lot.col] and not seen[lot.col]
                      and self.strategy.should_sell(lot, day)):
                    units[lot.col] -= lot.units
                    self.oms.sell_lot(lot.lot_id, marks[lot.col], date=day, settlement="stale_mark")
                    events.append({"date": day, "symbol": lot.symbol, "event": "settled_at_last_mark",
                                   "value": lot.units * marks[lot.col], "lot_id": lot.lot_id})
            # entries at today's close with budgets fixed at the signal close
            for symbol, budget in orders.items():
                if budget <= EPS:
                    continue
                c = self._col.get(symbol)
                if c is None or not seen[c]:
                    events.append({"date": day, "symbol": symbol, "event": "rejected_no_quote", "value": budget})
                    continue
                q = self.oms.size_for_budget(budget, quote[c])
                self.oms.buy(q, quote[c], symbol=symbol, col=c, date=day, signal_date=signal,
                             holding_days=self.strategy.holding_days)
                units[c] += q
            units[np.abs(units) < 1e-12] = 0.0

            held = units > 0
            gap = i - last_seen
            stale = held & ~seen
            newly_frozen = held & (gap > self.max_stale) & ~frozen
            resumed = frozen & seen
            for c in np.flatnonzero(newly_frozen):
                if self.on_long_gap == "halt":
                    raise HeldPriceGapError(f"{self.symbols[c]} unquoted since {cal[last_seen[c]].date()}")
                events.append({"date": day, "symbol": self.symbols[c], "event": "frozen_at_last_mark",
                               "value": units[c] * marks[c]})
            for c in np.flatnonzero(resumed):
                events.append({"date": day, "symbol": self.symbols[c], "event": "quotes_resumed",
                               "value": units[c] * marks[c]})
            frozen = (frozen | newly_frozen) & ~seen & held
            for c in np.flatnonzero(stale & ~frozen):
                events.append({"date": day, "symbol": self.symbols[c], "event": "stale_mark", "value": units[c] * marks[c]})

            snap = self._snapshot(day, units, np.nan_to_num(marks), frozen)
            terminal_value = float(self._terminal_values().sum()) if self.terminal else 0.0
            today = self.oms.fills[fills_before:]
            history.append({"date": day, "cash": snap.cash, "holdings_value": float(snap.holdings.sum()),
                            "nav": snap.nav, "open_lots": len(self.oms.lots),
                            "fees": sum(f["fee"] for f in today),
                            "bought": sum(f["notional"] for f in today if f["side"] == "BUY"),
                            "sold": sum(f["notional"] for f in today if f["side"] == "SELL"),
                            "stale_symbols": int(stale.sum()), "frozen_symbols": int(frozen.sum()),
                            "frozen_value": float((units[frozen] * marks[frozen]).sum()) + terminal_value})

        hist = pd.DataFrame(history).set_index("date")
        open_lots = pd.DataFrame([asdict(l) for l in self.oms.lots.values()],
                                 columns=list(Lot.__dataclass_fields__))
        if len(open_lots):
            term = open_lots.lot_id.map(self.terminal)
            open_lots["terminal"] = term.notna()
            open_lots["final_mark"] = term.fillna(pd.Series(marks[open_lots.col.to_numpy()], index=open_lots.index))
            open_lots["last_quote_date"] = cal[last_seen[open_lots.col.to_numpy()]]
            open_lots["frozen"] = frozen[open_lots.col.to_numpy()] | open_lots.terminal
            open_lots["market_value"] = open_lots.units * open_lots.final_mark
        final_marks = np.nan_to_num(marks)
        unrealized = sum(l.units * (self.terminal.get(l.lot_id, final_marks[l.col]) - l.entry_price) - l.entry_fee
                         for l in self.oms.lots.values())
        pnl = pd.Series({"initial_cash": self.oms.initial_cash, "final_cash": self.oms.cash,
                         "holdings_value": hist.holdings_value.iloc[-1], "nav": hist.nav.iloc[-1],
                         "realized_pnl": self.oms.realized_pnl, "unrealized_pnl": unrealized,
                         "total_pnl": hist.nav.iloc[-1] - self.oms.initial_cash, "total_fees": self.oms.total_fees})
        return BacktestResult(hist, pd.DataFrame(self.oms.fills, columns=OrderManagementSystem.FILL_COLUMNS),
                              pd.DataFrame(self.oms.closed), open_lots,
                              pd.DataFrame(events, columns=["date", "symbol", "event", "value", "lot_id"]), pnl,
                              {"fee_rate": self.oms.fee_rate, "max_stale_sessions": self.max_stale,
                               "on_long_gap": self.on_long_gap, "holding_days": self.strategy.holding_days,
                               "max_resume_gap_sessions": self.max_resume_gap,
                               "start": str(cal[days[0]].date()), "end": str(cal[days[-1]].date())})


def frozen_writeoff_nav(result: BacktestResult) -> pd.Series:
    """Lower bound: lots still frozen at the end are worth zero from the day they froze.

    (Upper bound is the reported NAV, which keeps them at their last mark.)
    """
    nav = result.history.nav.copy()
    if not len(result.open_lots) or not result.open_lots.frozen.any():
        return nav
    ev = result.events.loc[result.events.event.eq("frozen_at_last_mark")]
    for _, lot in result.open_lots.loc[result.open_lots.frozen].iterrows():
        froze = ev.loc[ev.symbol.eq(lot.symbol) & ev.date.ge(lot.entry_date), "date"]
        since = froze.min() if len(froze) else nav.index[-1]
        nav.loc[nav.index >= since] -= lot.units * lot.final_mark
    return nav
