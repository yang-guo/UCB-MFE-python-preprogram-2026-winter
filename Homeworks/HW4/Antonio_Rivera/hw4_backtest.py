"""HW4 backtester: my own implementation (no code copied from Lectures/modeling/backtest.py).

Roles (built one module at a time):
    OrderManager       executes buys/sells at a given size and price, owns cash and
                       open lots, charges fees, records every fill and each lot's PnL,
                       and keeps the daily books (cash, NAV, realized/unrealized PnL)
    Strategy           decides what/when to buy and when to sell; sees data only
                       through a MarketView cut at the signal close
                       (TopNScoreStrategy = monthly top-N cohorts)
    BacktestEngine     advances sessions, feeds data <= today, coordinates
                       (mark -> sell -> buy -> record -> plan, every session)
    PerformanceReport  returns, risk, trading costs, cash usage vs a buy-and-hold
                       ^SP500TR benchmark started at the same first close; chart

Accounting conventions (stated assumptions, see README):
    - Units are fractional ADJUSTED-price units, so dividends are already inside
      the price path and are never added separately.
    - Fee = fee_bps of notional on every buy and every sell. Default 10 bps per
      side; set fee_bps per OrderManager to test other trading-cost assumptions.
    - A buy is specified by a dollar BUDGET that already includes its fee:
          units = budget / (price x (1 + fee_rate)),  cash falls by exactly budget.
    - Lots are bought and sold whole; each keeps its own entry price and entry fee.
    - Cash earns no interest. Cash may never go negative.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

DEFAULT_FEE_BPS = 10
HOLDING_DAYS = 90
CASH_TOL = 1e-6  # dollars of floating-point dust tolerated


@dataclass(frozen=True)
class Lot:
    """One purchase. Immutable: a lot is opened once and closed once."""
    lot_id: int
    symbol: str
    units: float
    entry_date: pd.Timestamp
    entry_price: float
    entry_fee: float           # fee paid at entry ($)
    due_date: pd.Timestamp     # entry_date + 90 calendar days (sell on/after)

    @property
    def cost_basis(self):
        """Total cash that left the account for this lot."""
        return self.units * self.entry_price + self.entry_fee


class OrderManager:
    """Cash, open lots, fills, and per-lot PnL. Knows nothing about strategy or dates."""

    def __init__(self, initial_cash=1_000_000.0, fee_bps=DEFAULT_FEE_BPS):
        if not (np.isfinite(initial_cash) and initial_cash > 0):
            raise ValueError("initial_cash must be a positive number")
        self.initial_cash = float(initial_cash)
        self.cash = float(initial_cash)
        self.open_lots = {}        # lot_id -> Lot
        self.fills = []            # one dict per BUY or SELL
        self.closed = []           # one dict per fully closed lot (trade PnL)
        self.fees_paid = 0.0
        self.daily = []            # one dict per record() call: the daily books
        self._next_id = 1
        self._fills_recorded = 0   # fills already attributed to an earlier record()
        self.fee_bps = fee_bps     # validated by the setter below

    # --- fee setting -------------------------------------------------------------

    @property
    def fee_bps(self):
        """Fee per side in basis points of notional (10 bps = 0.10%)."""
        return self._fee_bps

    @fee_bps.setter
    def fee_bps(self, value):
        if not (np.isfinite(value) and value >= 0):
            raise ValueError(f"fee_bps must be a non-negative number, got {value}")
        if self.fills:
            # Changing fees mid-run would mix two cost assumptions in one result.
            raise RuntimeError("fee_bps can only be changed before the first fill; "
                               "run a separate backtest per fee level")
        self._fee_bps = float(value)

    @property
    def fee_rate(self):
        """fee_bps as a decimal: 10 bps -> 0.001."""
        return self._fee_bps / 10_000

    # --- execution ---------------------------------------------------------------

    def buy(self, symbol, budget, price, date, holding_days=HOLDING_DAYS):
        """Spend exactly `budget` dollars (fee included) on `symbol` at `price`."""
        _check_price(price)
        if not (np.isfinite(budget) and budget > 0):
            raise ValueError(f"budget must be positive, got {budget}")
        if budget > self.cash + CASH_TOL:
            raise ValueError(f"budget {budget:,.2f} exceeds cash {self.cash:,.2f}")
        units = budget / (price * (1 + self.fee_rate))
        fee = units * price * self.fee_rate
        date = pd.Timestamp(date)
        lot = Lot(self._next_id, symbol, units, date, float(price), fee,
                  date + pd.Timedelta(days=holding_days))
        self._next_id += 1
        self.open_lots[lot.lot_id] = lot
        self._move_cash(-budget)
        self.fees_paid += fee
        self._log("BUY", lot, date, price, fee, -budget)
        return lot

    def sell(self, lot_id, price, date):
        """Close a whole lot at `price`; return the closed-trade record."""
        _check_price(price)
        if lot_id not in self.open_lots:
            raise KeyError(f"lot {lot_id} is not open")
        lot = self.open_lots[lot_id]
        date = pd.Timestamp(date)
        if date <= lot.entry_date:
            raise ValueError(f"lot {lot_id}: cannot sell on/before its entry date")
        gross = lot.units * price
        fee = gross * self.fee_rate
        proceeds = gross - fee
        del self.open_lots[lot_id]
        self._move_cash(proceeds)
        self.fees_paid += fee
        self._log("SELL", lot, date, price, fee, proceeds)
        pnl = proceeds - lot.cost_basis
        trade = {"lot_id": lot.lot_id, "symbol": lot.symbol, "units": lot.units,
                 "entry_date": lot.entry_date, "exit_date": date,
                 "due_date": lot.due_date, "days_held": (date - lot.entry_date).days,
                 "entry_price": lot.entry_price, "exit_price": float(price),
                 "fees": lot.entry_fee + fee, "pnl": pnl, "return": pnl / lot.cost_basis}
        self.closed.append(trade)
        return trade

    # --- valuation -----------------------------------------------------------------

    def holdings_value(self, marks):
        """Σ units × mark. `marks` maps symbol -> price; every held symbol must be present."""
        missing = {l.symbol for l in self.open_lots.values()} - set(marks.keys())
        if missing:
            raise KeyError(f"no mark for held symbols: {sorted(missing)}")
        return sum(l.units * marks[l.symbol] for l in self.open_lots.values())

    def nav(self, marks):
        return self.cash + self.holdings_value(marks)

    def unrealized_pnl(self, marks):
        """Mark-to-market PnL of open lots, net of their entry fees (exit fee not yet paid)."""
        return sum(l.units * marks[l.symbol] - l.cost_basis for l in self.open_lots.values())

    def realized_pnl(self):
        return sum(t["pnl"] for t in self.closed)

    def exposure(self, marks):
        """Marked $ value of open lots per symbol (several lots in one symbol add up)."""
        out = {}
        for l in self.open_lots.values():
            out[l.symbol] = out.get(l.symbol, 0.0) + l.units * marks[l.symbol]
        return out

    # --- daily books ---------------------------------------------------------------

    def record(self, date, marks, stale_symbols=()):
        """Close the books for `date` at the given marks; append and return the row.

        Books identity (checked): nav = initial_cash + realized_pnl + unrealized_pnl.
        `stale_symbols` = held symbols whose mark is a last-known (not today's) quote;
        the caller decides marks, the books just record which ones were stale.
        """
        date = pd.Timestamp(date)
        if self.daily and date <= self.daily[-1]["date"]:
            raise ValueError(f"record dates must increase: {date.date()} after "
                             f"{self.daily[-1]['date'].date()}")
        today = self.fills[self._fills_recorded:]
        self._fills_recorded = len(self.fills)
        holdings = self.holdings_value(marks)
        realized, unrealized = self.realized_pnl(), self.unrealized_pnl(marks)
        nav = self.cash + holdings
        assert abs(nav - (self.initial_cash + realized + unrealized)) < 1e-4 * max(1.0, nav), \
            "books do not balance"
        row = {"date": date, "cash": self.cash, "holdings": holdings, "nav": nav,
               "realized_pnl": realized, "unrealized_pnl": unrealized,
               "open_lots": len(self.open_lots),
               "bought": sum(f["notional"] for f in today if f["side"] == "BUY"),
               "sold": sum(f["notional"] for f in today if f["side"] == "SELL"),
               "fees": sum(f["fee"] for f in today), "fees_cum": self.fees_paid,
               "stale_symbols": len(set(stale_symbols))}
        self.daily.append(row)
        return row

    def daily_frame(self):
        return pd.DataFrame(self.daily).set_index("date")

    # --- tables ------------------------------------------------------------------

    def fills_frame(self):
        return pd.DataFrame(self.fills)

    def closed_frame(self):
        return pd.DataFrame(self.closed)

    def open_lots_frame(self):
        return pd.DataFrame([vars(l) for l in self.open_lots.values()])

    # --- internals -----------------------------------------------------------------

    def _move_cash(self, amount):
        self.cash += amount
        if -CASH_TOL < self.cash < 0:
            self.cash = 0.0                           # floating-point dust
        assert self.cash >= 0, f"cash went negative: {self.cash}"

    def _log(self, side, lot, date, price, fee, cash_change):
        self.fills.append({"date": date, "side": side, "lot_id": lot.lot_id,
                           "symbol": lot.symbol, "units": lot.units, "price": float(price),
                           "notional": lot.units * price, "fee": fee,
                           "cash_change": cash_change, "cash_after": self.cash})


# =============================================================================
# Strategy layer: what the engine shows a strategy, and what a strategy decides
# =============================================================================

class LookAheadError(RuntimeError):
    """A strategy asked for data after its signal date."""


class MarketView:
    """Read-only window onto a (date, symbol) panel, cut at `cutoff` (the signal close).

    The engine builds one per session. Every accessor returns a COPY of rows with
    date <= cutoff, so a strategy can neither see nor modify the future.
    """

    def __init__(self, panel, cutoff, positions=None):
        self._panel = panel            # full panel; never handed out directly
        self.cutoff = pd.Timestamp(cutoff)
        self._pos = positions          # optional {date: row positions}, built once by the engine

    @property
    def columns(self):
        return list(self._panel.columns)

    def _rows_on(self, date):
        if self._pos is not None:     # fast path: O(rows that day) instead of a full scan
            return self._panel.iloc[self._pos.get(date, [])]
        return self._panel.loc[self._panel.date.eq(date)]

    def today(self, columns=None):
        """Rows dated exactly at the cutoff (one per symbol quoted that day)."""
        rows = self._rows_on(self.cutoff)
        return (rows if columns is None else rows[["date", "symbol"] + list(columns)]).copy()

    def history(self, start=None, columns=None):
        """Rows with start <= date <= cutoff."""
        start = pd.Timestamp(start) if start is not None else self._panel.date.min()
        if start > self.cutoff:
            raise LookAheadError(f"history start {start.date()} is after cutoff {self.cutoff.date()}")
        rows = self._panel.loc[self._panel.date.between(start, self.cutoff)]
        return (rows if columns is None else rows[["date", "symbol"] + list(columns)]).copy()

    def at(self, date, columns=None):
        """Rows at one past-or-present date; any future date raises LookAheadError."""
        date = pd.Timestamp(date)
        if date > self.cutoff:
            raise LookAheadError(f"asked for {date.date()} but cutoff is {self.cutoff.date()}")
        rows = self._rows_on(date)
        return (rows if columns is None else rows[["date", "symbol"] + list(columns)]).copy()


@dataclass(frozen=True)
class PortfolioState:
    """What a strategy may know about the account at the signal close."""
    cash: float
    nav: float
    exposure: dict             # symbol -> marked $ value of open lots
    open_lots: tuple           # tuple of Lot (immutable)


class Strategy:
    """Base class. Subclasses decide what/when to buy; the default sell rule is maturity.

    The engine calls, at each signal close d:
        plan_orders(d, view, state) -> {symbol: budget $}   filled at close d+1
    and at each session close d (before buys):
        lots_to_sell(d, open_lots, quoted) -> [lot_id, ...]  filled at close d
    """

    holding_days = HOLDING_DAYS

    def plan_orders(self, date, view, state):
        raise NotImplementedError

    def lots_to_sell(self, date, open_lots, quoted):
        """Sell every lot whose due date has arrived AND that has a real quote today.

        due <= date covers weekend/holiday maturities (sell next session).
        A due lot without a quote is held, marked stale, and sold on its next quote.
        """
        date = pd.Timestamp(date)
        return [l.lot_id for l in open_lots if l.due_date <= date and l.symbol in quoted]


TRADED_WINDOW = 21


def add_traded_days(panel, calendar, window=TRADED_WINDOW):
    """traded_days = number of the last `window` market sessions (through t) with volume > 0.

    Causal (uses volume up to t only); a session with no quote counts as not traded.
    """
    cal = pd.DatetimeIndex(calendar)
    out = panel.copy()
    counts = pd.Series(np.nan, index=out.index)
    for _, g in out.groupby("symbol", sort=False):
        traded = (g.set_index("date").volume.reindex(cal) > 0).astype(float)
        rolled = traded.rolling(window, min_periods=1).sum()
        counts.loc[g.index] = rolled.reindex(g.date).to_numpy()
    out["traded_days"] = counts
    return out


class TopNScoreStrategy(Strategy):
    """Periodic cohorts: buy the top-N eligible, liquid names by a score column.

    When : the first session of each calendar period is a signal date (orders fill
           at the next session's close). frequency="M" (monthly, ~3 overlapping
           cohorts) or "W" (weekly, ~13 cohorts; proceeds of a maturing lot are
           reinvested within a week -> removes the monthly cash drag).
    What : (skip_suspect) rows flagged mcap_suspect are never bought.
    What : the top_n names by `score_col` among rows with eligible == True, a
           non-missing score, and volume > 0 on >= min_volume_days of the last
           volume_window sessions. Ties break alphabetically.
    Size : cohort budget = min(cash, cohort_fraction x NAV), split equally.
           Each name is capped so (existing exposure + new) <= max_weight x NAV;
           budget removed by the cap stays in cash (not redistributed).
    Sell : inherited maturity rule.
    """

    def __init__(self, score_col="score", top_n=20, cohort_fraction=1 / 3, max_weight=0.05,
                 volume_window=21, min_volume_days=18, min_order=100.0, skip_suspect=True,
                 frequency="M", weighting="equal"):
        if not 0 < cohort_fraction <= 1 or not 0 < max_weight <= 1 or top_n < 1:
            raise ValueError("bad strategy parameters")
        if frequency not in ("M", "W"):
            raise ValueError("frequency must be 'M' (monthly) or 'W' (weekly)")
        if weighting not in ("equal", "rank"):
            raise ValueError("weighting must be 'equal' or 'rank'")
        self.frequency, self.weighting = frequency, weighting
        self.score_col, self.top_n = score_col, top_n
        self.cohort_fraction, self.max_weight = cohort_fraction, max_weight
        self.volume_window, self.min_volume_days = volume_window, min_volume_days
        self.min_order = min_order
        self.skip_suspect = skip_suspect     # skip rows flagged mcap_suspect (bad price/share basis)
        self._last_signal_month = None

    def is_signal_date(self, date):
        """True on the first session seen in each new calendar month (or week)."""
        month = pd.Timestamp(date).to_period(self.frequency)
        if month == self._last_signal_month:
            return False
        self._last_signal_month = month
        return True

    def liquid_symbols(self, view):
        """Symbols with volume > 0 on enough of the last `volume_window` sessions.

        Fast path: a precomputed causal `traded_days` column (see add_traded_days).
        Fallback: count from the history window (slow on large panels).
        """
        if "traded_days" in view.columns and self.volume_window == TRADED_WINDOW:
            t = view.today(columns=["traded_days"])
            return set(t.loc[t.traded_days.ge(self.min_volume_days), "symbol"])
        hist = view.history(columns=["volume"])
        dates = np.sort(hist.date.unique())[-self.volume_window:]
        recent = hist.loc[hist.date.isin(dates)]
        traded_days = recent.loc[recent.volume.gt(0)].groupby("symbol").size()
        return set(traded_days[traded_days >= self.min_volume_days].index)

    def select(self, view):
        """Ranked list of symbols to buy at the next close."""
        has_flag = self.skip_suspect and "mcap_suspect" in view.columns
        today = view.today(columns=["eligible", self.score_col] + (["mcap_suspect"] if has_flag else []))
        ok = today.eligible.fillna(False).astype(bool) & today[self.score_col].notna()
        if has_flag:
            ok &= ~today.mcap_suspect.fillna(False).astype(bool)
        cands = today.loc[ok & today.symbol.isin(self.liquid_symbols(view))]
        cands = cands.sort_values([self.score_col, "symbol"], ascending=[False, True])
        return cands.symbol.head(self.top_n).tolist()

    def plan_orders(self, date, view, state):
        if not self.is_signal_date(date):
            return {}
        picks = self.select(view)
        if not picks:
            return {}
        cohort = min(state.cash, self.cohort_fraction * state.nav)
        orders = {}
        for sym, w in zip(picks, self.weights(len(picks))):
            room = self.max_weight * state.nav - state.exposure.get(sym, 0.0)
            budget = min(cohort * w, max(room, 0.0))
            if budget >= self.min_order:
                orders[sym] = budget
        return orders

    def weights(self, n):
        """Cohort weights for picks ranked 1..n (best first); they sum to 1.

        equal: 1/n each.  rank: linear in rank, w_i = (n - i + 1) / (n(n+1)/2),
        so the best name gets n shares and the n-th gets 1 share.
        """
        if self.weighting == "equal":
            return np.full(n, 1.0 / n)
        raw = np.arange(n, 0, -1, dtype=float)
        return raw / raw.sum()


# =============================================================================
# Engine: advances sessions and coordinates strategy <-> order manager
# =============================================================================

@dataclass
class BacktestResult:
    daily: pd.DataFrame        # one row per session: cash, holdings, nav, activity
    fills: pd.DataFrame        # every BUY / SELL
    trades: pd.DataFrame       # every closed lot with its PnL
    open_lots: pd.DataFrame    # lots still open at the end, with their end marks
    diagnostics: pd.DataFrame  # cancelled orders, stale marks, overdue lots


class BacktestEngine:
    """Run one strategy over [start, end] on the market calendar.

    Session d, in this order (all at the close of d):
      1. MARK   update last known price for every symbol quoted on d
      2. SELL   lots the strategy releases (maturity rule) at close d
      3. BUY    orders planned at close d-1, at close d; no quote on d -> cancelled
      4. RECORD choose marks (stale = last real quote, flagged); OrderManager.record
                books cash, NAV, realized and unrealized PnL
      5. PLAN   strategy sees data <= d and returns budgets for close d+1
                (skipped on the last session: there is no d+1 inside the test)

    Guarantees enforced here (each raises if violated):
      - a plan never budgets more than the cash available at its signal close
      - no lot is sold before its due date (90-day holding rule)
      - only prices dated <= the current session are ever used; none after `end`
    """

    def __init__(self, panel, calendar, strategy, order_manager, start, end,
                 stale_warn_sessions=5):
        self.panel = panel
        self.strategy = strategy
        self.om = order_manager
        cal = pd.DatetimeIndex(calendar).sort_values()
        self.sessions = cal[(cal >= pd.Timestamp(start)) & (cal <= pd.Timestamp(end))]
        if len(self.sessions) < 2:
            raise ValueError("need at least two sessions between start and end")
        off = ~panel.date.isin(cal)
        if off.any():
            raise ValueError(f"{int(off.sum())} panel rows fall outside the market calendar")
        self.stale_warn_sessions = stale_warn_sessions
        # Quotes per session, only inside the test window: nothing after `end` exists here.
        quoted = panel.loc[panel.date.isin(self.sessions) & panel.adj_close.gt(0),
                           ["date", "symbol", "adj_close"]]
        self._quotes = {d: g.set_index("symbol").adj_close for d, g in quoted.groupby("date")}
        # Row positions per date, so MarketView.today() is a lookup, not a full scan.
        self._positions = panel.reset_index(drop=True).groupby("date").indices
        self.panel = panel.reset_index(drop=True)

    def run(self):
        om, strat = self.om, self.strategy
        last_mark, last_seen = {}, {}           # symbol -> price / session index of last quote
        pending = {}                            # orders planned at the previous close
        diags = []

        for k, d in enumerate(self.sessions):
            quotes = self._quotes.get(d, pd.Series(dtype=float))

            # 1. MARK
            for sym, px in quotes.items():
                last_mark[sym], last_seen[sym] = px, k

            # 2. SELL (maturity), using today's real quote only
            lots = tuple(om.open_lots.values())
            for lot_id in strat.lots_to_sell(d, lots, set(quotes.index)):
                lot = om.open_lots[lot_id]
                if d < lot.due_date:
                    raise RuntimeError(f"strategy tried to sell lot {lot_id} before its due date")
                om.sell(lot_id, quotes[lot.symbol], d)

            # 3. BUY orders planned at the previous close
            for sym, budget in sorted(pending.items()):
                if sym not in quotes.index:
                    diags.append({"date": d, "symbol": sym, "event": "order_cancelled_no_quote",
                                  "amount": budget})
                    continue
                om.buy(sym, budget, quotes[sym], d, holding_days=strat.holding_days)
            pending = {}

            # 4. RECORD: the engine chooses marks (stale = last real quote, never an
            #    invented price) and logs data incidents; the OrderManager keeps the books.
            held = {l.symbol for l in om.open_lots.values()}
            stale = {s: k - last_seen[s] for s in held if last_seen[s] < k}
            for s, age in stale.items():
                if age == self.stale_warn_sessions + 1:     # report once, when it crosses
                    units = sum(l.units for l in om.open_lots.values() if l.symbol == s)
                    diags.append({"date": d, "symbol": s, "event": "stale_mark_over_limit",
                                  "amount": units * last_mark[s]})
            for l in om.open_lots.values():
                if l.due_date <= d and l.symbol not in quotes.index:
                    diags.append({"date": d, "symbol": l.symbol, "event": "due_lot_no_quote",
                                  "amount": l.units * last_mark[l.symbol]})
            marks = {s: last_mark[s] for s in held}
            books = om.record(d, marks, stale_symbols=stale.keys())

            # 5. PLAN for the next session (none on the final session)
            if k == len(self.sessions) - 1:
                break
            state = PortfolioState(om.cash, books["nav"], om.exposure(marks),
                                   tuple(om.open_lots.values()))
            orders = strat.plan_orders(d, MarketView(self.panel, d, self._positions), state) or {}
            if any(not (np.isfinite(b) and b > 0) for b in orders.values()):
                raise ValueError(f"{d.date()}: order budgets must be positive")
            if sum(orders.values()) > om.cash + CASH_TOL:
                raise ValueError(f"{d.date()}: orders total {sum(orders.values()):,.2f} "
                                 f"> cash {om.cash:,.2f} (same cash spent twice)")
            pending = dict(orders)

        end_marks = {l.symbol: last_mark[l.symbol] for l in om.open_lots.values()}
        open_end = om.open_lots_frame()
        if len(open_end):
            open_end["end_mark"] = open_end.symbol.map(end_marks)
            open_end["end_value"] = open_end.units * open_end.end_mark
            open_end["mark_date"] = open_end.symbol.map(lambda s: self.sessions[last_seen[s]])
        return BacktestResult(om.daily_frame(), om.fills_frame(),
                              om.closed_frame(), open_end,
                              pd.DataFrame(diags, columns=["date", "symbol", "event", "amount"]))


# =============================================================================
# Analytics: returns, risk, trading costs, cash usage (strategy vs benchmark)
# =============================================================================

TRADING_DAYS = 252
DAYS_PER_YEAR = 365.25


def load_benchmark(database, end):
    """^SP500TR adjusted close (a total-return index) as a date-indexed Series."""
    import duckdb
    with duckdb.connect(str(database), read_only=True) as con:
        b = con.execute("SELECT date, adj_close FROM benchmark WHERE date <= ? ORDER BY date",
                        [end]).fetchdf()
    return b.assign(date=pd.to_datetime(b.date)).set_index("date").adj_close


def benchmark_nav(prices, dates, initial=1_000_000.0):
    """$initial in the benchmark at the close of dates[0], held through dates[-1].

    No fees on the benchmark (one buy and hold, stated as an assumption).
    Uses only benchmark closes on the given dates; a missing date is an error.
    """
    dates = pd.DatetimeIndex(dates)
    px = prices.reindex(dates)
    if px.isna().any():
        raise ValueError(f"benchmark has no close on {px[px.isna()].index[0].date()}")
    return (initial * px / px.iloc[0]).rename("nav")


def performance_metrics(nav, rf_annual=0.02):
    """Headline metrics for one daily NAV series (first row = starting capital).

    CAGR    = (end / start) ^ (365.25 / elapsed calendar days) - 1
    Sharpe  = mean(r - rf_d) / std(r) x sqrt(252),  rf_d = (1 + rf)^(1/252) - 1,
              r = daily simple returns; std uses ddof=1; zero std -> NaN
    MaxDD   = min over t of (nav_t / running peak - 1)
    """
    nav = nav.astype(float)
    r = nav.pct_change().dropna()
    years = (nav.index[-1] - nav.index[0]).days / DAYS_PER_YEAR
    rf_d = (1 + rf_annual) ** (1 / TRADING_DAYS) - 1
    sd = r.std(ddof=1)
    dd = nav / nav.cummax() - 1
    trough = dd.idxmin()
    peak = nav.loc[:trough].idxmax()
    return {
        "start": nav.index[0].date(), "end": nav.index[-1].date(),
        "final_value": nav.iloc[-1],
        "total_return": nav.iloc[-1] / nav.iloc[0] - 1,
        "cagr": (nav.iloc[-1] / nav.iloc[0]) ** (1 / years) - 1,
        "ann_vol": sd * np.sqrt(TRADING_DAYS),
        "sharpe": (r.mean() - rf_d) / sd * np.sqrt(TRADING_DAYS) if sd > 0 else np.nan,
        "max_drawdown": dd.min(),
        "dd_peak": peak.date(), "dd_trough": trough.date(),
    }


class PerformanceReport:
    """Summaries of one BacktestResult against a buy-and-hold benchmark."""

    def __init__(self, result, benchmark_prices, rf_annual=0.02):
        self.result = result
        self.daily = result.daily
        self.rf_annual = rf_annual
        self.nav = self.daily.nav.rename("strategy")
        self.bench = benchmark_nav(benchmark_prices, self.nav.index,
                                   initial=self.nav.iloc[0]).rename("benchmark")

    def summary(self):
        """Required metrics for both portfolios, side by side."""
        return pd.DataFrame({"Strategy": performance_metrics(self.nav, self.rf_annual),
                             "Benchmark": performance_metrics(self.bench, self.rf_annual)})

    def verdict(self):
        s, b = (performance_metrics(x, self.rf_annual) for x in (self.nav, self.bench))
        return {"beat_benchmark": s["final_value"] > b["final_value"],
                "excess_final_value": s["final_value"] - b["final_value"],
                "excess_cagr": s["cagr"] - b["cagr"],
                "excess_sharpe": s["sharpe"] - b["sharpe"],
                "drawdown_difference": s["max_drawdown"] - b["max_drawdown"]}

    def trading(self):
        """Costs, turnover, trade outcomes, and cash usage of the strategy."""
        d, t = self.daily, self.result.trades
        years = (d.index[-1] - d.index[0]).days / DAYS_PER_YEAR
        cash_w = d.cash / d.nav
        traded = d.bought.sum() + d.sold.sum()
        diag = self.result.diagnostics
        return {
            "fees_total": d.fees.sum(),
            "fees_pct_of_start": d.fees.sum() / d.nav.iloc[0],
            "turnover_per_year": traded / d.nav.mean() / years,   # (buys + sells) / avg NAV
            "lots_closed": len(t),
            "lots_open_at_end": len(self.result.open_lots),
            "win_rate": (t.pnl > 0).mean() if len(t) else np.nan,
            "mean_lot_return": t["return"].mean() if len(t) else np.nan,
            "median_lot_return": t["return"].median() if len(t) else np.nan,
            "realized_pnl": d.realized_pnl.iloc[-1],
            "unrealized_pnl_end": d.unrealized_pnl.iloc[-1],
            "mean_cash_weight": cash_w.mean(),
            "max_cash_weight_after_first_buy": cash_w[d.open_lots.cumsum().gt(0)].max()
                if d.open_lots.gt(0).any() else np.nan,
            "days_with_stale_marks": int(d.stale_symbols.gt(0).sum()),
            "orders_cancelled": int(diag.event.eq("order_cancelled_no_quote").sum()),
        }

    def plot(self, path, title="Cumulative return: strategy vs ^SP500TR"):
        """One chart, one y-axis: cumulative return of both portfolios. Saves a PNG."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import PercentFormatter

        ink, ink2, grid, surface = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
        series = [(self.nav, "Strategy", "#2a78d6"), (self.bench, "^SP500TR", "#eb6834")]
        fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
        fig.patch.set_facecolor(surface)
        ax.set_facecolor(surface)
        ends = []
        for s, label, color in series:
            cum = s / s.iloc[0] - 1
            ax.plot(cum.index, cum.values, color=color, lw=1.0, label=label)  # ≈2px at 150 dpi
            ends.append((cum.iloc[-1], cum.index[-1], label))
        # Direct end labels; push them apart vertically when the lines finish close together.
        (y_lo, x_lo, l_lo), (y_hi, x_hi, l_hi) = sorted(ends)
        span = max(abs(y_hi - y_lo), 1e-9)
        gap = 7 if span < 0.05 * (ax.get_ylim()[1] - ax.get_ylim()[0]) else 0
        for y, x, lab, dy in ((y_lo, x_lo, l_lo, -gap), (y_hi, x_hi, l_hi, gap)):
            ax.annotate(f"{lab} {y:+.0%}", (x, y), xytext=(6, dy), textcoords="offset points",
                        va="center", fontsize=9, color=ink)
        ax.axhline(0, color=ink2, lw=0.8)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        ax.grid(axis="y", color=grid, lw=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(grid)
        ax.tick_params(colors=ink2, labelsize=9)
        ax.set_title(title, loc="left", fontsize=11, color=ink)
        ax.set_ylabel("Cumulative return", color=ink2, fontsize=9)
        ax.legend(frameon=False, loc="upper left", fontsize=9, labelcolor=ink)
        ax.margins(x=0.0)
        fig.subplots_adjust(right=0.80)          # room for the direct labels
        fig.savefig(path, facecolor=surface)
        plt.close(fig)
        return path


def _check_price(price):
    if not (np.isfinite(price) and price > 0):
        raise ValueError(f"price must be positive and finite, got {price}")
