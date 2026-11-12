"""A small daily-close backtester for long-only, fixed-holding-period lots.

Design (inspired by the Lecture 8 framework in ``Lectures/modeling/backtest.py``:
separate strategy / order management / daily loop, one *lot* per purchase,
fee-inclusive budgets, adjusted-price units).  The implementation below is my
own and adds what the full HW4 universe needs:

* **Suspect quotes.**  Some tickers interleave two price series or have broken
  split adjustments.  A quote more than x3 (or /3) away from the last accepted
  mark is not traded on and not used for valuation.  A *fall* is accepted once
  it persists for 5 sessions (a real crash must be booked as a loss); a *rise*
  is never accepted, so a data error can never inflate NAV.
* **Missing quotes / delistings.**  A held stock without an accepted quote is
  marked at its last accepted price ("stale").  A lot that is due is sold at
  the first accepted quote on/after its maturity.  After 5 stale sessions a
  holding is flagged *unresolved*: we never invent a sale or payout, so its
  capital stays frozen (it is not recycled into cash).  We report two NAVs:
  ``nav`` (unresolved lots at their last mark: upper bound) and ``nav_low``
  (unresolved lots at zero: lower bound).

Timing and cash: orders are planned from the portfolio and information at the
signal close (the previous session) and fill at today's close.  The snapshot
also reports ``maturing``: the signal-close value of lots whose maturity is
on/before today, i.e. cash that today's sales are *expected* to return.  On the
fill day we execute sales first and purchases second; if the actual cash is
smaller than the planned budgets (prices fell, or a quote was missing), every
purchase is scaled down proportionally.  Cash therefore never goes negative
and is never spent twice.
"""

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .features import SUSPECT_JUMP


@dataclass(frozen=True)
class Lot:
    lot_id: int
    symbol: str
    units: float            # adjusted-price units (total-return exposure)
    signal_date: pd.Timestamp
    entry_date: pd.Timestamp
    entry_price: float
    entry_fee: float
    due_date: pd.Timestamp  # entry date + holding period (calendar days)


@dataclass(frozen=True)
class Snapshot:
    """What a strategy is allowed to see when it plans orders."""
    date: pd.Timestamp
    cash: float
    nav: float
    holdings: pd.Series     # market value by symbol at the signal close
    fee_rate: float
    maturing: float = 0.0   # signal-close value of lots due by the next session (gross of fees)

    @property
    def available(self):
        """Cash plus expected after-fee proceeds of lots that mature at the next close."""
        return self.cash + self.maturing * (1 - self.fee_rate)


class Account:
    """Cash, open lots and the fill ledger (the order management system)."""

    FILL_COLUMNS = ["date", "signal_date", "lot_id", "symbol", "side", "price", "units",
                    "notional", "fee", "cash_change", "due_date", "realized_pnl"]

    def __init__(self, initial_cash=1_000_000.0, fee_bps=10.0):
        if initial_cash <= 0 or fee_bps < 0:
            raise ValueError("Capital must be positive and fees nonnegative")
        self.initial_cash = float(initial_cash)
        self.cash = float(initial_cash)
        self.fee_rate = fee_bps / 10_000
        self.lots: dict[int, Lot] = {}
        self.fills: list[dict] = []
        self._next_id = 0

    def buy(self, symbol, budget, price, *, date, signal_date, holding_days):
        """Spend a fee-inclusive ``budget`` on ``symbol`` at ``price``."""
        if not (np.isfinite(budget) and np.isfinite(price)) or budget <= 0 or price <= 0:
            raise ValueError("Budget and price must be positive and finite")
        if not signal_date < date:
            raise ValueError("The decision must be made before the fill date")
        units = budget / (price * (1 + self.fee_rate))
        notional = units * price
        fee = notional * self.fee_rate
        if notional + fee > self.cash + 1e-6:
            raise ValueError(f"Insufficient cash on {date.date()}: never spend the same cash twice")
        self.cash = max(0.0, self.cash - notional - fee)
        lot = Lot(self._next_id, symbol, units, signal_date, date, price, fee,
                  date + pd.Timedelta(days=holding_days))
        self.lots[lot.lot_id] = lot
        self._next_id += 1
        self._record(lot, "BUY", date, price, notional, fee, -(notional + fee), 0.0)
        return lot

    def sell(self, lot_id, price, *, date):
        """Close a whole lot at ``price``; cash arrives immediately."""
        lot = self.lots.pop(lot_id)
        if date < lot.due_date:
            raise ValueError("Lots are only sold at or after maturity")
        notional = lot.units * price
        fee = notional * self.fee_rate
        pnl = notional - fee - lot.units * lot.entry_price - lot.entry_fee
        self.cash += notional - fee
        self._record(lot, "SELL", date, price, notional, fee, notional - fee, pnl)
        return pnl

    def _record(self, lot, side, date, price, notional, fee, cash_change, pnl):
        self.fills.append({"date": date, "signal_date": lot.signal_date, "lot_id": lot.lot_id,
                           "symbol": lot.symbol, "side": side, "price": price, "units": lot.units,
                           "notional": notional, "fee": fee, "cash_change": cash_change,
                           "due_date": lot.due_date, "realized_pnl": pnl})


@dataclass
class BacktestResult:
    history: pd.DataFrame       # one row per session (plus the initial-capital row)
    fills: pd.DataFrame
    open_lots: pd.DataFrame     # still held at the end, with their final marks
    diagnostics: pd.DataFrame   # stale marks, suspect quotes, rejected orders, ...
    initial_cash: float
    fee_rate: float


class Backtester:
    """Advance one market session at a time."""

    def __init__(self, adj, strategy, *, initial_cash=1_000_000.0, fee_bps=10.0,
                 holding_days=90, max_stale_sessions=5, confirm_sessions=5,
                 suspect_jump=SUSPECT_JUMP, unresolved_exit="freeze"):
        """``unresolved_exit``: "freeze" (base case: never invent a sale) or
        "last_mark" (SENSITIVITY ONLY: a matured lot with no accepted quote for more
        than ``max_stale_sessions`` is assumed to exit at its last accepted mark,
        e.g. a cash acquisition paying about the last traded price)."""
        if unresolved_exit not in ("freeze", "last_mark"):
            raise ValueError("unresolved_exit must be 'freeze' or 'last_mark'")
        self.unresolved_exit = unresolved_exit
        self.adj = adj                                   # wide adjusted closes, NaN = no quote
        self.calendar = pd.DatetimeIndex(adj.index)
        self.col = {s: i for i, s in enumerate(adj.columns)}
        self.values = adj.to_numpy()
        self.strategy = strategy
        self.account = Account(initial_cash, fee_bps)
        self.holding_days = holding_days
        self.max_stale = max_stale_sessions
        self.confirm = confirm_sessions
        self.jump = suspect_jump

    # ------------------------------------------------------------------ quotes
    def _previous_quote(self, symbol, t):
        """Last observed quote strictly before session t (for new purchases)."""
        col = self.values[:t, self.col[symbol]]
        seen = np.flatnonzero(np.isfinite(col))
        return col[seen[-1]] if len(seen) else np.nan

    def _accept(self, reference, quote):
        return (not np.isfinite(reference)) or (1 / self.jump <= quote / reference <= self.jump)

    # ------------------------------------------------------------------ main loop
    def run(self, start, end):
        cal = self.calendar
        days = cal[(cal >= pd.Timestamp(start)) & (cal <= pd.Timestamp(end))]
        t0 = cal.get_loc(days[0])
        if t0 == 0:
            raise ValueError("Need one session before the start to make the first decision")
        acct = self.account
        mark, last_ok, suspect = {}, {}, {}   # per held symbol
        deferred = set()                      # lot ids past maturity but without a sellable quote
        diag = []
        history = [{"date": cal[t0 - 1], "cash": acct.cash, "holdings_value": 0.0,
                    "nav": acct.cash, "nav_low": acct.cash, "open_lots": 0, "stale_symbols": 0,
                    "unresolved_value": 0.0, "bought": 0.0, "sold": 0.0, "fees": 0.0}]
        snapshot = Snapshot(cal[t0 - 1], acct.cash, acct.cash, pd.Series(dtype=float), acct.fee_rate)

        for day in days:
            t = cal.get_loc(day)
            signal_date = cal[t - 1]
            # 1. Plan with yesterday's information only (today's prices are not passed).
            orders = self.strategy.orders(signal_date, day, snapshot)
            if orders.sum() > snapshot.available + 1e-6 or (orders < 0).any():
                raise ValueError("Orders must be nonnegative and covered by cash + expected proceeds")
            row = self.values[t]
            n_fills = len(acct.fills)

            # 2. Update marks for held symbols (accept, ignore as suspect, or stale).
            held = {lot.symbol for lot in acct.lots.values()}
            accepted = {}
            for sym in held:
                q = row[self.col[sym]]
                if not np.isfinite(q):
                    continue
                if self._accept(mark[sym], q):
                    accepted[sym] = q
                    suspect[sym] = 0
                elif q < mark[sym]:
                    # A >x3 fall: accept once it has persisted (a genuine crash is a real loss).
                    suspect[sym] = suspect.get(sym, 0) + 1
                    if suspect[sym] >= self.confirm:
                        accepted[sym] = q
                        suspect[sym] = 0
                        diag.append({"date": day, "symbol": sym, "event": "suspect_drop_confirmed"})
                    else:
                        diag.append({"date": day, "symbol": sym, "event": "suspect_drop_ignored"})
                else:
                    # A >x3 one-step rise is never accepted: in this data it is a broken
                    # series (e.g. LCI 29 -> 493 in Dec 2013).  The holding keeps its last
                    # good mark and becomes "unresolved" if no plausible quote returns.
                    diag.append({"date": day, "symbol": sym, "event": "suspect_rise_ignored"})
            for sym, q in accepted.items():
                mark[sym], last_ok[sym] = q, t

            # 3. Sell matured lots at today's accepted close.
            for lot in list(acct.lots.values()):
                if day >= lot.due_date:
                    if lot.symbol in accepted:
                        acct.sell(lot.lot_id, accepted[lot.symbol], date=day)
                        if lot.lot_id in deferred:
                            deferred.discard(lot.lot_id)
                            diag.append({"date": day, "symbol": lot.symbol, "event": "deferred_exit_filled",
                                         "lot_id": lot.lot_id})
                    elif self.unresolved_exit == "last_mark" and t - last_ok[lot.symbol] > self.max_stale:
                        acct.sell(lot.lot_id, mark[lot.symbol], date=day)
                        deferred.discard(lot.lot_id)
                        diag.append({"date": day, "symbol": lot.symbol, "event": "assumed_exit_at_last_mark",
                                     "lot_id": lot.lot_id})
                    elif lot.lot_id not in deferred:      # log once; retry every session
                        deferred.add(lot.lot_id)
                        diag.append({"date": day, "symbol": lot.symbol, "event": "exit_deferred",
                                     "lot_id": lot.lot_id})

            # 4. Buy at today's close (only with a quote that passes the jump check).
            fillable = {}
            for sym, budget in orders.items():
                if budget <= 1e-6:
                    continue
                q = row[self.col[sym]]
                if not np.isfinite(q):
                    diag.append({"date": day, "symbol": sym, "event": "rejected_missing_quote"})
                    continue
                ref = mark.get(sym, self._previous_quote(sym, t))
                if not self._accept(ref, q):
                    diag.append({"date": day, "symbol": sym, "event": "rejected_suspect_quote"})
                    continue
                fillable[sym] = (budget, q)
            planned = sum(b for b, _ in fillable.values())
            scale = min(1.0, acct.cash / planned) if planned > 0 else 1.0
            if scale < 1.0:
                diag.append({"date": day, "symbol": None, "event": f"orders_scaled_to_cash x{scale:.4f}"})
            for sym, (budget, q) in fillable.items():
                acct.buy(sym, budget * scale * (1 - 1e-12), q, date=day, signal_date=signal_date,
                         holding_days=self.holding_days)
                if sym not in held:                   # new holding: today's fill price is its mark
                    mark[sym], last_ok[sym], suspect[sym] = q, t, 0

            # 5. Value the account at today's close.
            value = {}
            for lot in acct.lots.values():
                value[lot.symbol] = value.get(lot.symbol, 0.0) + lot.units * mark[lot.symbol]
            holdings = pd.Series(value, dtype=float)
            stale = [s for s in value if last_ok[s] < t]
            unresolved = [s for s in value if t - last_ok[s] > self.max_stale]
            for s in stale:
                diag.append({"date": day, "symbol": s, "event": "unresolved_mark" if s in unresolved else "stale_mark"})
            unresolved_value = float(holdings.reindex(unresolved).sum()) if unresolved else 0.0
            nav = acct.cash + holdings.sum()
            todays = acct.fills[n_fills:]
            history.append({"date": day, "cash": acct.cash, "holdings_value": holdings.sum(), "nav": nav,
                            "nav_low": nav - unresolved_value, "open_lots": len(acct.lots),
                            "stale_symbols": len(stale), "unresolved_value": unresolved_value,
                            "bought": sum(f["notional"] for f in todays if f["side"] == "BUY"),
                            "sold": sum(f["notional"] for f in todays if f["side"] == "SELL"),
                            "fees": sum(f["fee"] for f in todays)})
            # Expected proceeds for tomorrow: lots due by the next session that have a
            # fresh accepted quote today (stale/unresolved lots are not counted).
            next_day = cal[t + 1] if t + 1 < len(cal) else day
            maturing = sum(lot.units * mark[lot.symbol] for lot in acct.lots.values()
                           if lot.due_date <= next_day and last_ok[lot.symbol] == t)
            snapshot = Snapshot(day, acct.cash, nav, holdings, acct.fee_rate, maturing)
            for sym in list(mark):                    # forget symbols no longer held
                if sym not in value:
                    mark.pop(sym), last_ok.pop(sym), suspect.pop(sym, None)

        open_lots = pd.DataFrame([asdict(l) for l in acct.lots.values()], columns=list(Lot.__dataclass_fields__))
        if len(open_lots):
            open_lots["last_mark"] = open_lots.symbol.map(mark)
            open_lots["last_quote_date"] = open_lots.symbol.map(lambda s: cal[last_ok[s]])
            open_lots["market_value"] = open_lots.units * open_lots.last_mark
            open_lots["unresolved"] = open_lots.symbol.map(lambda s: (t - last_ok[s]) > self.max_stale)
        return BacktestResult(
            history=pd.DataFrame(history).set_index("date"),
            fills=pd.DataFrame(acct.fills, columns=Account.FILL_COLUMNS),
            open_lots=open_lots,
            diagnostics=pd.DataFrame(diag, columns=["date", "symbol", "event", "lot_id"]),
            initial_cash=acct.initial_cash, fee_rate=acct.fee_rate)
