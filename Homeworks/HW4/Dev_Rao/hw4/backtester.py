"""My own lot-based, long-only daily backtester (written from scratch for HW4).

Accounting model
----------------
* Positions are fractional units of the ADJUSTED close, so a unit's value
  already includes reinvested dividends; dividends are never added again.
* Each purchase is a separate `Lot` held 90 calendar days. It is sold at the
  close on its due date, or at the first later session with a quote.
* Fees: `fee_rate` of notional on every buy and sell. A buy budget is
  fee-inclusive: units = budget / (price * (1 + fee_rate)).
* Daily order of events on session d:
    1. (optional) cash interest accrues;
    2. lots whose stock has had no quote for > max_stale sessions are exited
       at the last observed close x `forced_exit_recovery` (flagged);
    3. due lots with a quote today are sold at today's close;
    4. buys planned at the PREVIOUS signal close fill at today's close, each
       capped at the cash actually available (reductions are logged);
       buys without a quote today are rejected (logged), never invented;
    5. everything is marked (stale marks <= max_stale sessions are flagged);
    6. if d is a signal date, the strategy plans tomorrow's buys using only
       information through d's close.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config

EPS = 1e-6


@dataclass(frozen=True)
class Lot:
    lot_id: int
    symbol: str
    sector: str
    signal_date: pd.Timestamp
    entry_date: pd.Timestamp
    due_date: pd.Timestamp       # entry_date + 90 calendar days
    entry_price: float
    units: float
    entry_fee: float

    @property
    def cost(self):
        return self.units * self.entry_price + self.entry_fee


@dataclass
class Order:
    symbol: str
    sector: str
    budget: float                # fee-inclusive dollars
    signal_date: pd.Timestamp
    rank: int


class Ledger:
    """Cash, open lots, fills, realised PnL and fees."""

    def __init__(self, initial_cash=config.INITIAL_CAPITAL, fee_bps=config.FEE_BPS):
        if initial_cash <= 0 or fee_bps < 0:
            raise ValueError("capital must be positive and fees nonnegative")
        self.initial_cash = float(initial_cash)
        self.cash = float(initial_cash)
        self.fee_rate = fee_bps / 10_000
        self.lots = {}               # lot_id -> Lot
        self.fills = []
        self.closed = []
        self.total_fees = 0.0
        self.realized_pnl = 0.0
        self.interest = 0.0
        self._next_id = 1

    # -- trading -------------------------------------------------------------
    def buy(self, symbol, sector, date, signal_date, price, budget, holding_days=config.HOLDING_DAYS):
        date, signal_date = pd.Timestamp(date), pd.Timestamp(signal_date)
        assert date > signal_date, "a fill may never use the signal-date price"
        if not (np.isfinite(price) and price > 0 and budget > 0):
            raise ValueError("invalid buy")
        units = budget / (price * (1 + self.fee_rate))
        notional = units * price
        fee = notional * self.fee_rate
        if notional + fee > self.cash + EPS:
            raise ValueError("insufficient cash: the same cash can never be spent twice")
        lot = Lot(self._next_id, symbol, sector, signal_date, date,
                  date + pd.Timedelta(days=holding_days), float(price), units, fee)
        self._next_id += 1
        self.lots[lot.lot_id] = lot
        self.cash = max(0.0, self.cash - notional - fee)
        self.total_fees += fee
        self._fill(lot, "BUY", date, price, notional, fee, -(notional + fee), 0.0, "entry")
        return lot

    def sell(self, lot_id, date, price, reason="maturity", recovery=1.0):
        lot = self.lots.pop(lot_id)
        px = price * recovery
        notional = lot.units * px
        fee = notional * self.fee_rate
        pnl = notional - fee - lot.cost
        self.cash += notional - fee
        self.total_fees += fee
        self.realized_pnl += pnl
        self._fill(lot, "SELL", date, px, notional, fee, notional - fee, pnl, reason)
        self.closed.append({"lot_id": lot.lot_id, "symbol": lot.symbol, "sector": lot.sector,
                            "signal_date": lot.signal_date, "entry_date": lot.entry_date,
                            "due_date": lot.due_date, "exit_date": pd.Timestamp(date),
                            "entry_price": lot.entry_price, "exit_price": px, "units": lot.units,
                            "cost": lot.cost, "proceeds": notional - fee, "fees": lot.entry_fee + fee,
                            "realized_pnl": pnl, "return": pnl / lot.cost, "exit_reason": reason})
        return pnl

    def accrue_interest(self, daily_rate):
        if daily_rate:
            amt = self.cash * daily_rate
            self.cash += amt
            self.interest += amt

    def _fill(self, lot, side, date, price, notional, fee, cash_change, pnl, reason):
        self.fills.append({"date": pd.Timestamp(date), "lot_id": lot.lot_id, "symbol": lot.symbol,
                           "side": side, "signal_date": lot.signal_date, "price": price,
                           "units": lot.units, "notional": notional, "fee": fee,
                           "cash_change": cash_change, "realized_pnl": pnl, "reason": reason})

    # -- valuation -----------------------------------------------------------
    def exposures(self, marks):
        """Market value per symbol and per sector at the given marks."""
        by_sym, by_sec = {}, {}
        for lot in self.lots.values():
            v = lot.units * marks[lot.symbol]
            by_sym[lot.symbol] = by_sym.get(lot.symbol, 0.0) + v
            by_sec[lot.sector] = by_sec.get(lot.sector, 0.0) + v
        return by_sym, by_sec

    def unrealized_pnl(self, marks):
        return sum(lot.units * marks[lot.symbol] - lot.cost for lot in self.lots.values())


@dataclass
class Snapshot:
    """What a strategy may see when planning after the close on `date`."""
    date: pd.Timestamp
    next_session: pd.Timestamp
    cash: float
    nav: float
    by_symbol: dict
    by_sector: dict
    due_next_value: float        # marked value of lots due by the next session
    fee_rate: float


class TopNCohortStrategy:
    """Buy the top-N scored eligible names each scheduled signal date.

    Cohort budget = min(projected cash, NAV * cohort_fraction), where projected
    cash = cash + signal-close value of lots that mature by the next session
    (those are sold before buys; the fill-time cash cap is the hard limit).
    Equal weight within the cohort. Caps (by market value, summed across lots):
    `name_cap` of NAV per name and `sector_cap` of NAV per sector; a capped name
    passes its budget to the next-ranked name.
    """

    def __init__(self, scores, schedule, top_n=25, cohort_fraction=1 / 13, name_cap=0.05,
                 sector_cap=0.30, universe="all", liq_top=1000, search_depth=3, min_order=100.0):
        s = scores.copy()
        if universe == "liquid":
            s = s.loc[s.liq_rank <= liq_top]
        elif universe != "all":
            raise ValueError(universe)
        self.by_date = {d: g.sort_values("score", ascending=False) for d, g in s.groupby("date")}
        self.schedule = pd.DatetimeIndex(schedule)
        self.top_n, self.cohort_fraction = top_n, cohort_fraction
        self.name_cap, self.sector_cap = name_cap, sector_cap
        self.search_depth, self.min_order = search_depth, min_order

    def plan(self, snap: Snapshot):
        cands = self.by_date.get(snap.date)
        if cands is None or cands.empty:
            return []
        budget = min(snap.cash + snap.due_next_value, snap.nav * self.cohort_fraction)
        per_name = budget / self.top_n
        sym_room = {}
        sec_room = {}
        orders, remaining = [], budget
        for rank, row in enumerate(cands.head(self.top_n * self.search_depth).itertuples(), 1):
            if remaining < self.min_order or len(orders) >= self.top_n * self.search_depth:
                break
            s, sec = row.symbol, row.sector
            sym_room.setdefault(s, self.name_cap * snap.nav - snap.by_symbol.get(s, 0.0))
            sec_room.setdefault(sec, self.sector_cap * snap.nav - snap.by_sector.get(sec, 0.0))
            # rooms are in notional dollars; budgets are fee-inclusive
            want = min(per_name, remaining)
            alloc = min(want, sym_room[s] * (1 + snap.fee_rate), sec_room[sec] * (1 + snap.fee_rate))
            if alloc < self.min_order:
                continue
            orders.append(Order(s, sec, alloc, snap.date, rank))
            notional = alloc / (1 + snap.fee_rate)
            sym_room[s] -= notional
            sec_room[sec] -= notional
            remaining -= alloc
        return orders


@dataclass
class BacktestResult:
    history: pd.DataFrame
    fills: pd.DataFrame
    lots: pd.DataFrame           # closed lots with realised PnL + open lots at the end
    diagnostics: pd.DataFrame
    ledger: Ledger = field(repr=False)


class Backtester:
    def __init__(self, prices, strategy, sectors, fee_bps=config.FEE_BPS,
                 initial_cash=config.INITIAL_CAPITAL, max_stale=config.MAX_STALE_SESSIONS,
                 forced_exit_recovery=1.0, cash_rate_annual=0.0):
        """prices: DataFrame (market sessions x symbols) of adjusted closes, NaN = no quote."""
        assert prices.index.is_monotonic_increasing and prices.index.is_unique
        self.prices = prices
        self.px = prices.to_numpy(dtype="float64")
        self.col = {s: i for i, s in enumerate(prices.columns)}
        self.dates = prices.index
        self.strategy = strategy
        self.sectors = sectors
        self.fee_bps, self.initial_cash = fee_bps, initial_cash
        self.max_stale = max_stale
        self.recovery = forced_exit_recovery
        self.daily_rate = (1 + cash_rate_annual) ** (1 / 252) - 1 if cash_rate_annual else 0.0

    def run(self, start, end):
        """`start` = first signal date (initial-capital row); `end` = last mark date."""
        dates = self.dates
        i0, i1 = dates.get_loc(pd.Timestamp(start)), dates.get_loc(pd.Timestamp(end))
        schedule = set(self.strategy.schedule)
        led = Ledger(self.initial_cash, self.fee_bps)
        last_px, last_i = {}, {}        # symbol -> last observed close / session index
        pending, hist, diag = [], [], []
        for i in range(i0, i1 + 1):
            d = dates[i]
            row = self.px[i]
            led.accrue_interest(self.daily_rate)
            held = {lot.symbol for lot in led.lots.values()}
            for s in held:
                p = row[self.col[s]]
                if np.isfinite(p):
                    last_px[s], last_i[s] = p, i
            fees0, bought, sold, forced = led.total_fees, 0.0, 0.0, 0
            # 2-3. exits: stale beyond max_stale sessions, then due lots with a quote
            for lot in sorted(led.lots.values(), key=lambda l: l.lot_id):
                gap = i - last_i[lot.symbol]
                if gap > self.max_stale:
                    led.sell(lot.lot_id, d, last_px[lot.symbol], reason="no_quote_forced_exit",
                             recovery=self.recovery)
                    sold += lot.units * last_px[lot.symbol] * self.recovery
                    forced += 1
                    diag.append({"date": d, "event": "forced_exit", "symbol": lot.symbol, "lot_id": lot.lot_id,
                                 "detail": f"no quote for {gap} sessions; exit at last close "
                                           f"{last_px[lot.symbol]:.4f} x recovery {self.recovery}",
                                 "value": lot.units * last_px[lot.symbol]})
                elif lot.due_date <= d and gap == 0:
                    led.sell(lot.lot_id, d, row[self.col[lot.symbol]], reason="maturity")
                    sold += lot.units * row[self.col[lot.symbol]]
            # 4. fills of orders planned at the previous signal close
            reduced = rejected = 0
            for o in pending:
                p = row[self.col[o.symbol]] if o.symbol in self.col else np.nan
                if not np.isfinite(p):
                    rejected += 1
                    diag.append({"date": d, "event": "rejected_no_quote", "symbol": o.symbol,
                                 "lot_id": None, "detail": "no quote on entry session", "value": o.budget})
                    continue
                budget = o.budget
                if budget > led.cash + EPS:
                    reduced += 1
                    diag.append({"date": d, "event": "reduced_to_cash", "symbol": o.symbol, "lot_id": None,
                                 "detail": f"budget {budget:.2f} -> cash {led.cash:.2f}", "value": budget - led.cash})
                    budget = led.cash
                if budget < 1.0:
                    continue
                lot = led.buy(o.symbol, o.sector, d, o.signal_date, p, budget)
                last_px[o.symbol], last_i[o.symbol] = p, i
                bought += lot.units * p
            pending = []
            # 5. marks
            marks, stale = {}, 0
            for lot in led.lots.values():
                s = lot.symbol
                marks[s] = last_px[s]
                if last_i[s] < i:
                    stale += 1
            by_sym, by_sec = led.exposures(marks)
            hv = sum(by_sym.values())
            nav = led.cash + hv
            hist.append({"date": d, "cash": led.cash, "holdings_value": hv, "nav": nav,
                         "open_lots": len(led.lots), "n_names": len(by_sym),
                         "fees": led.total_fees - fees0, "gross_bought": bought, "gross_sold": sold,
                         "stale_marks": stale, "forced_exits": forced, "rejected": rejected,
                         "reduced": reduced, "realized_pnl_cum": led.realized_pnl,
                         "unrealized_pnl": led.unrealized_pnl(marks), "interest_cum": led.interest})
            # 6. plan tomorrow's buys with information through today's close
            if d in schedule and i < i1:
                nxt = dates[i + 1]
                due_val = sum(lot.units * marks[lot.symbol] for lot in led.lots.values() if lot.due_date <= nxt)
                snap = Snapshot(d, nxt, led.cash, nav, by_sym, by_sec, due_val, led.fee_rate)
                pending = self.strategy.plan(snap)
        history = pd.DataFrame(hist).set_index("date")
        fills = pd.DataFrame(led.fills)
        final_marks = {lot.symbol: last_px[lot.symbol] for lot in led.lots.values()}
        open_lots = pd.DataFrame([{"lot_id": l.lot_id, "symbol": l.symbol, "sector": l.sector,
                                   "signal_date": l.signal_date, "entry_date": l.entry_date,
                                   "due_date": l.due_date, "entry_price": l.entry_price, "units": l.units,
                                   "cost": l.cost, "mark": final_marks[l.symbol],
                                   "unrealized_pnl": l.units * final_marks[l.symbol] - l.cost,
                                   "exit_reason": "open_at_end"} for l in led.lots.values()])
        lots = pd.concat([pd.DataFrame(led.closed), open_lots], ignore_index=True)
        diagnostics = pd.DataFrame(diag, columns=["date", "event", "symbol", "lot_id", "detail", "value"])
        result = BacktestResult(history, fills, lots, diagnostics, led)
        reconcile(result)
        return result


def reconcile(result, tol=1e-4):
    """Accounting identities checked after every run (raise on failure).

    * cash rebuilt from initial capital + cumulative fill cash flows (+ interest)
      equals ledger cash on every day;
    * NAV - initial == realised + unrealised PnL (+ interest) every day;
    * fees in fills == ledger total == history daily sum;
    * every fill happened strictly after its signal date.
    """
    h, f, led = result.history, result.fills, result.ledger
    flows = f.groupby("date").cash_change.sum().reindex(h.index, fill_value=0.0).cumsum() if len(f) else 0.0
    rebuilt = led.initial_cash + flows + h.interest_cum
    assert np.allclose(rebuilt, h.cash, atol=tol * 100), "cash does not reconcile with fills"
    pnl = h.realized_pnl_cum + h.unrealized_pnl + h.interest_cum
    assert np.allclose(h.nav - led.initial_cash, pnl, atol=tol * 100), "NAV does not reconcile with PnL"
    if len(f):
        assert np.isclose(f.fee.sum(), led.total_fees) and np.isclose(h.fees.sum(), led.total_fees)
        assert (f.date > f.signal_date).all(), "fill on or before its signal date"
        sells = f.loc[f.side == "SELL"].merge(f.loc[f.side == "BUY", ["lot_id", "date"]], on="lot_id",
                                              suffixes=("", "_entry"))
        assert (sells.date > sells.date_entry).all(), "lot sold on or before its entry"
    assert (h.cash >= -EPS).all(), "negative cash (leverage)"
    return True
