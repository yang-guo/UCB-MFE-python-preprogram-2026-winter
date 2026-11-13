"""HW4 backtester: long-only, fixed 90-calendar-day holding period.

Written for this homework. The split into strategy / account / daily loop /
analytics follows the structure of the Lecture 8 framework
(Lectures/modeling/backtest.py: Strategy, OrderManagementSystem, Backtester,
BacktestAnalytics); the code itself is new and does not copy that file.

Pieces
------
* ``Strategy``      decides order budgets after the close of a decision day.
* ``Account``       books cash, lots, fees and fills; never sees the future.
* ``Backtester``    the daily loop: sell matured lots, fill yesterday's orders,
                    mark positions, record history, ask the strategy for orders.
* ``performance``   final value, CAGR, max drawdown, Sharpe.
* ``accounting_checks`` identities that must hold on any finished run.

Timing and accounting rules
---------------------------
* Signals are made after the close of session t; orders fill at the close of
  the next session t+1. A strategy never sees t+1's price before deciding.
* Orders planned at t may spend at most the cash available at t's close. Sale
  proceeds that arrive on t+1 do not fund t+1's buys; they are available to
  orders planned at t+1's close or later. (Enforced: the planned budgets are
  checked against the t-close cash, and an error is raised if they exceed it.)
* Fee: ``fee_bps`` per side. The entry fee is inside the dollar budget:
  units = budget / (price * (1 + fee)), so a purchase costs exactly ``budget``.
  A sale returns units * price * (1 - fee).
* Each lot is due ``holding_days`` calendar days after its entry date. It is
  sold at the close of the first session on or after the due date that has a
  quote. Fills are never invented: no quote, no trade.
* Missing entry quote: the order is rejected and the cash stays.
* Missing marks: the last observed quote is carried and the lot is flagged as
  stale. If a held stock goes more than ``max_stale_sessions`` sessions without
  a quote, the ``long_gap_policy`` decides:
    - "raise" (default): ``StaleQuoteError`` stops the run for investigation.
    - "hold": the lot stays open at its last mark (no fill is invented), is
      recorded once in ``result.long_gaps``, keeps being flagged as stale, and
      is sold normally if quotes resume after its due date. Lots still frozen
      at the end are reported so a zero-recovery sensitivity can be shown.
* Implausible one-day moves on held stocks (|log move| > ln 5 by default) are
  recorded in ``result.jumps``. Prices are never altered.
* Cash earns 0%. No borrowing, no shorting, no leverage.
* Prices are adjusted closes, so dividends are already in the returns and are
  not added again.
* History starts with an initial-capital row on the first session (the first
  decision day, before any fill). Open lots at the end stay in NAV at their
  last mark.

Credits: component split modeled on Lectures/modeling/backtest.py; no code
copied. Written with AI assistance (Claude Code) and reviewed by the author.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


class InsufficientCashError(ValueError):
    """Orders or a purchase would spend more cash than is available."""


class StaleQuoteError(RuntimeError):
    """A held stock has gone too many sessions without a quote."""


# ----------------------------------------------------------------------------
# Account: cash, lots, fees, fills
# ----------------------------------------------------------------------------
@dataclass
class Lot:
    lot_id: int
    symbol: str
    units: float
    entry_date: pd.Timestamp
    entry_price: float
    cost: float               # cash paid, entry fee included
    signal_date: pd.Timestamp
    due_date: pd.Timestamp
    mark: float               # last observed quote
    mark_date: pd.Timestamp
    stale_sessions: int = 0


class Account:
    """Books every cash movement. Knows nothing about strategies or the future."""

    def __init__(self, initial_cash=1_000_000.0, fee_bps=10.0, holding_days=90):
        self.initial_cash = float(initial_cash)
        self.cash = float(initial_cash)
        self.fee = fee_bps / 10_000
        self.holding_days = holding_days
        self.lots = {}            # lot_id -> Lot (open lots only)
        self.closed = []          # dicts for sold lots
        self.fills = []           # every buy and sell
        self._next_id = 0

    def buy(self, symbol, budget, price, date, signal_date):
        if not (budget > 0 and np.isfinite(price) and price > 0):
            raise ValueError(f"Invalid buy: budget={budget}, price={price}")
        if budget > self.cash + 1e-6:
            raise InsufficientCashError(f"{date.date()} {symbol}: budget {budget:,.2f} > cash {self.cash:,.2f}")
        units = budget / (price * (1 + self.fee))
        fee = units * price * self.fee
        self.cash -= budget
        lot = Lot(self._next_id, symbol, units, date, price, budget, signal_date,
                  date + pd.Timedelta(days=self.holding_days), price, date)
        self.lots[lot.lot_id] = lot
        self._next_id += 1
        self.fills.append({"date": date, "side": "buy", "symbol": symbol, "lot_id": lot.lot_id,
                           "units": units, "price": price, "gross": units * price, "fee": fee,
                           "cash_flow": -budget, "signal_date": signal_date, "due_date": lot.due_date})
        return lot

    def sell(self, lot_id, price, date):
        lot = self.lots.pop(lot_id)
        gross = lot.units * price
        fee = gross * self.fee
        proceeds = gross - fee
        self.cash += proceeds
        self.fills.append({"date": date, "side": "sell", "symbol": lot.symbol, "lot_id": lot_id,
                           "units": lot.units, "price": price, "gross": gross, "fee": fee,
                           "cash_flow": proceeds, "signal_date": lot.signal_date, "due_date": lot.due_date})
        self.closed.append({"lot_id": lot_id, "symbol": lot.symbol, "entry_date": lot.entry_date,
                            "exit_date": date, "due_date": lot.due_date, "cost": lot.cost,
                            "proceeds": proceeds, "pnl": proceeds - lot.cost,
                            "return": proceeds / lot.cost - 1})
        return proceeds

    def holdings_value(self):
        return sum(lot.units * lot.mark for lot in self.lots.values())

    def holdings_by_symbol(self):
        out = {}
        for lot in self.lots.values():
            out[lot.symbol] = out.get(lot.symbol, 0.0) + lot.units * lot.mark
        return out


# ----------------------------------------------------------------------------
# Strategies
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class Snapshot:
    """What a strategy may see after the close of a decision day."""
    date: pd.Timestamp
    cash: float
    nav: float
    holdings: dict            # symbol -> market value at last marks


class Strategy:
    """Base class. ``lookback`` = number of sessions of prices the strategy needs.

    With lookback 0 the strategy receives ``window=None`` (precomputed scores).
    Otherwise it receives a *copy* of the last ``lookback`` sessions of adjusted
    closes ending at the decision date, so editing it cannot change the
    backtester's quotes.
    """
    lookback = 0

    def generate_orders(self, snapshot, window):
        """Return {symbol: dollar budget} to fill at the next session's close."""
        raise NotImplementedError


class TopNStrategy(Strategy):
    """Shared sizing: buy the top N names with a cohort budget.

    cohort budget   = min(cohort_fraction * NAV, cash at the decision close)
    per-name budget = min(cohort budget / N, per_name_cap * NAV - value already held in that name)
    The cap limits a name's TOTAL position (existing lots at their last marks plus
    the new order), so repeated purchases of the same top-ranked stock cannot pile
    up. Budget a capped name cannot use stays in cash (it is not redistributed).
    """

    def __init__(self, top_n=50, cohort_fraction=1 / 3, per_name_cap=0.02):
        self.top_n = top_n
        self.cohort_fraction = cohort_fraction
        self.per_name_cap = per_name_cap

    def rank(self, snapshot, window):
        """Return a Series of scores (higher is better) for this decision date."""
        raise NotImplementedError

    def generate_orders(self, snapshot, window):
        scores = self.rank(snapshot, window)
        if scores is None or scores.empty:
            return {}
        picks = scores.dropna().sort_values(ascending=False, kind="mergesort").head(self.top_n)
        if picks.empty:
            return {}
        cohort = min(self.cohort_fraction * snapshot.nav, snapshot.cash)
        base = cohort / len(picks)
        cap_value = self.per_name_cap * snapshot.nav
        orders = {}
        for symbol in picks.index:
            budget = min(base, cap_value - snapshot.holdings.get(symbol, 0.0))
            if budget > 0.01:            # skip dust orders
                orders[symbol] = budget
        return orders


class ScoreStrategy(TopNStrategy):
    """Uses precomputed model scores: {date: Series(symbol -> score)}. No prices passed."""
    lookback = 0

    def __init__(self, scores_by_date, **kwargs):
        super().__init__(**kwargs)
        self.scores_by_date = scores_by_date

    def rank(self, snapshot, window):
        return self.scores_by_date.get(snapshot.date)


class TrailingReturnStrategy(TopNStrategy):
    """Simple price-based example (used in tests): rank by trailing return."""

    def __init__(self, lookback=21, **kwargs):
        super().__init__(**kwargs)
        self.lookback = lookback

    def rank(self, snapshot, window):
        if window is None or len(window) < self.lookback:
            return None
        return (window.iloc[-1] / window.iloc[0] - 1).dropna()


class FixedOrdersStrategy(Strategy):
    """Places hard-coded orders {decision date: {symbol: budget}} (used in tests)."""

    def __init__(self, orders):
        self.orders = {pd.Timestamp(k): v for k, v in orders.items()}

    def generate_orders(self, snapshot, window):
        return dict(self.orders.get(snapshot.date, {}))


# ----------------------------------------------------------------------------
# Daily loop
# ----------------------------------------------------------------------------
@dataclass
class BacktestResult:
    history: pd.DataFrame
    fills: pd.DataFrame
    orders: pd.DataFrame
    closed_lots: pd.DataFrame
    open_lots: pd.DataFrame
    stale: pd.DataFrame
    jumps: pd.DataFrame
    long_gaps: pd.DataFrame
    initial_cash: float
    params: dict = field(default_factory=dict)


class Backtester:
    """Run a strategy over a wide price table (dates x symbols, adjusted closes, NaN = no quote)."""

    def __init__(self, prices, strategy, initial_cash=1_000_000.0, fee_bps=10.0, holding_days=90,
                 entry_every=21, max_stale_sessions=5, jump_threshold=np.log(5), long_gap_policy="raise"):
        if long_gap_policy not in ("raise", "hold"):
            raise ValueError("long_gap_policy must be 'raise' or 'hold'")
        prices = prices.sort_index()
        if not prices.index.is_unique:
            raise ValueError("Price dates must be unique")
        values = prices.to_numpy(dtype=float, copy=True)
        values[~(np.isfinite(values) & (values > 0))] = np.nan   # no zero/negative quotes
        self._values = values
        self._values.setflags(write=False)                       # backtester's own read-only copy
        self.dates = pd.DatetimeIndex(prices.index)
        self.symbols = list(prices.columns)
        self._col = {s: i for i, s in enumerate(self.symbols)}
        self.strategy = strategy
        self.params = dict(initial_cash=initial_cash, fee_bps=fee_bps, holding_days=holding_days,
                           entry_every=entry_every, max_stale_sessions=max_stale_sessions,
                           jump_threshold=float(jump_threshold), long_gap_policy=long_gap_policy)

    def quote(self, i, symbol):
        j = self._col.get(symbol)
        if j is None:
            return np.nan
        return self._values[i, j]

    def _window(self, i):
        n = self.strategy.lookback
        if not n:
            return None
        lo = max(0, i - n + 1)
        return pd.DataFrame(self._values[lo:i + 1].copy(), index=self.dates[lo:i + 1], columns=self.symbols)

    def run(self):
        p = self.params
        account = Account(p["initial_cash"], p["fee_bps"], p["holding_days"])
        history, orders_log, stale_log, jump_log, long_gaps = [], [], [], [], {}
        pending, pending_signal = {}, None

        for i, date in enumerate(self.dates):
            buys = sells = rejected = 0

            # 1. Jump diagnostic: compare today's quote with each held symbol's last mark.
            seen = set()
            for lot in account.lots.values():
                if lot.symbol in seen:
                    continue
                q = self.quote(i, lot.symbol)
                if np.isfinite(q):
                    move = np.log(q / lot.mark)
                    if abs(move) > p["jump_threshold"]:
                        jump_log.append({"date": date, "symbol": lot.symbol, "previous_mark": lot.mark,
                                         "previous_mark_date": lot.mark_date, "quote": q, "log_move": move})
                    seen.add(lot.symbol)

            # 2. Sell matured lots that have a quote today (first quoted close on/after due date).
            for lot in sorted(account.lots.values(), key=lambda x: x.lot_id):
                if date >= lot.due_date:
                    q = self.quote(i, lot.symbol)
                    if np.isfinite(q):
                        account.sell(lot.lot_id, q, date)
                        sells += 1

            # 3. Fill the orders planned at the previous close (budgets fixed then).
            for symbol, budget in pending.items():
                q = self.quote(i, symbol)
                status = "filled"
                if np.isfinite(q):
                    account.buy(symbol, budget, q, date, pending_signal)
                    buys += 1
                else:
                    status = "rejected: no entry quote"
                    rejected += 1
                orders_log.append({"signal_date": pending_signal, "fill_date": date, "symbol": symbol,
                                   "budget": budget, "status": status})
            pending, pending_signal = {}, None

            # 4. Mark open lots; carry the last mark when there is no quote.
            stale_today = 0
            for lot in account.lots.values():
                q = self.quote(i, lot.symbol)
                if np.isfinite(q):
                    lot.mark, lot.mark_date, lot.stale_sessions = q, date, 0
                else:
                    lot.stale_sessions += 1
                    stale_today += 1
                    stale_log.append({"date": date, "symbol": lot.symbol, "lot_id": lot.lot_id,
                                      "mark_date": lot.mark_date, "stale_sessions": lot.stale_sessions,
                                      "past_due": date >= lot.due_date})
                    if lot.stale_sessions > p["max_stale_sessions"] and p["long_gap_policy"] == "hold":
                        long_gaps.setdefault(lot.lot_id, {"symbol": lot.symbol, "lot_id": lot.lot_id,
                                                          "last_quote_date": lot.mark_date, "flagged_on": date,
                                                          "last_mark": lot.mark, "units": lot.units})
                    elif lot.stale_sessions > p["max_stale_sessions"]:
                        raise StaleQuoteError(
                            f"{lot.symbol} (lot {lot.lot_id}) has no quote for {lot.stale_sessions} sessions "
                            f"through {date.date()}; last quote {lot.mark_date.date()}. Investigate before trusting the run.")

            # 5. Record the day.
            holdings = account.holdings_value()
            history.append({"date": date, "cash": account.cash, "holdings": holdings,
                            "nav": account.cash + holdings, "open_lots": len(account.lots),
                            "stale_lots": stale_today, "buys": buys, "sells": sells, "rejected": rejected})

            # 6. Decide after the close (not on the last session: nothing could fill).
            if i % p["entry_every"] == 0 and i + 1 < len(self.dates):
                snap = Snapshot(date, account.cash, account.cash + holdings, account.holdings_by_symbol())
                orders = self.strategy.generate_orders(snap, self._window(i)) or {}
                orders = {s: float(b) for s, b in orders.items() if b and b > 0}
                if sum(orders.values()) > snap.cash + 1e-6:
                    raise InsufficientCashError(
                        f"{date.date()}: orders total {sum(orders.values()):,.2f} > cash {snap.cash:,.2f}")
                pending, pending_signal = orders, date

        open_lots = pd.DataFrame([{**vars(l), "market_value": l.units * l.mark} for l in account.lots.values()])
        gaps = pd.DataFrame(list(long_gaps.values()),
                            columns=["symbol", "lot_id", "last_quote_date", "flagged_on", "last_mark", "units"])
        gaps["still_open_at_end"] = gaps.lot_id.isin(list(account.lots.keys()))
        gaps["value_at_end"] = np.where(gaps.still_open_at_end, gaps.units * gaps.last_mark, 0.0)
        return BacktestResult(
            history=pd.DataFrame(history).set_index("date"),
            fills=pd.DataFrame(account.fills, columns=["date", "side", "symbol", "lot_id", "units", "price", "gross",
                                                       "fee", "cash_flow", "signal_date", "due_date"]),
            orders=pd.DataFrame(orders_log, columns=["signal_date", "fill_date", "symbol", "budget", "status"]),
            closed_lots=pd.DataFrame(account.closed),
            open_lots=open_lots,
            stale=pd.DataFrame(stale_log, columns=["date", "symbol", "lot_id", "mark_date", "stale_sessions", "past_due"]),
            jumps=pd.DataFrame(jump_log, columns=["date", "symbol", "previous_mark", "previous_mark_date", "quote", "log_move"]),
            long_gaps=gaps,
            initial_cash=account.initial_cash,
            params=dict(p),
        )


# ----------------------------------------------------------------------------
# Benchmark and analytics
# ----------------------------------------------------------------------------
def benchmark_nav(benchmark, dates, invest_date, initial_cash=1_000_000.0):
    """Hold cash until `invest_date`'s close, then all in the benchmark (no fee)."""
    level = benchmark.reindex(dates)
    if level.isna().any():
        raise ValueError("Benchmark is missing a session in the backtest window")
    nav = pd.Series(initial_cash, index=dates, dtype=float)
    after = dates >= invest_date
    nav[after] = initial_cash * level[after] / level[invest_date]
    return nav


def performance(nav, annual_rf=0.02, periods=252):
    """Final value, CAGR (elapsed days / 365.25), max drawdown, Sharpe on daily returns."""
    nav = nav.astype(float)
    days = (nav.index[-1] - nav.index[0]).days
    cagr = (nav.iloc[-1] / nav.iloc[0]) ** (365.25 / days) - 1
    drawdown = nav / nav.cummax() - 1
    daily = nav.pct_change().dropna()
    rf_daily = (1 + annual_rf) ** (1 / periods) - 1
    excess = daily - rf_daily
    sharpe = excess.mean() / excess.std(ddof=1) * np.sqrt(periods)
    return pd.Series({"final_value": nav.iloc[-1], "CAGR": cagr, "max_drawdown": drawdown.min(),
                      "sharpe": sharpe, "ann_volatility": daily.std(ddof=1) * np.sqrt(periods),
                      "start": nav.index[0].date(), "end": nav.index[-1].date()})


def accounting_checks(result, tol=1e-6):
    """Identities every finished run must satisfy. Returns {check: passed}."""
    h, f = result.history, result.fills
    checks = {}
    flows = f.groupby("date").cash_flow.sum().reindex(h.index, fill_value=0.0) if len(f) else 0.0
    rebuilt = result.initial_cash + (flows.cumsum() if len(f) else 0.0)
    checks["cash rebuilt from fills == history cash"] = bool(np.allclose(rebuilt, h.cash, atol=tol * 1e3, rtol=0))
    checks["NAV == cash + holdings every day"] = bool(np.allclose(h.nav, h.cash + h.holdings, atol=tol, rtol=0))
    checks["cash never negative"] = bool((h.cash >= -tol).all())
    buys, sells = f.loc[f.side.eq("buy")], f.loc[f.side.eq("sell")]
    checks["every buy: signal date < fill date"] = bool((buys.signal_date < buys.date).all())
    checks["every sale on/after its due date"] = bool((sells.date >= sells.due_date).all())
    checks["each lot sold at most once, same units"] = bool(
        sells.lot_id.is_unique and set(sells.lot_id) <= set(buys.lot_id)
        and np.allclose(sells.set_index("lot_id").units, buys.set_index("lot_id").units.reindex(sells.lot_id)))
    open_ids = set(result.open_lots.lot_id) if len(result.open_lots) else set()
    checks["open lots = bought - sold"] = open_ids == set(buys.lot_id) - set(sells.lot_id)
    if len(result.open_lots):
        last = h.iloc[-1]
        checks["final holdings == open lots at last mark"] = bool(
            np.isclose(result.open_lots.market_value.sum(), last.holdings, atol=tol))
    return checks
