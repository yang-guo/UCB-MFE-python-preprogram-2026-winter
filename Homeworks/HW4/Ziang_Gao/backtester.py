"""Own backtester for HW4 (long only, no leverage, 90-calendar-day holds).

Who does what
-------------
* The STRATEGY decides what to buy and how much (class `Strategy`, example
  `TopNStrategy`). It only proposes dollar orders. It sees one row of scores and a
  small view of the portfolio, both known at the previous close.
* The BACKTESTER (`run_backtest`) enforces the rules of the market and does the
  accounting. It checks every order, fills at the close, charges fees, sells after
  90 calendar days, and tracks cash and value. A strategy cannot break these rules.
* A `Ledger` holds cash and open lots. `audit`, `performance` and `compare` check
  and measure the result.

One session, in order
---------------------
(1) ask the strategy for orders using information through the PREVIOUS close,
(2) update marks, (3) sell lots that reached 90 calendar days, (4) book the
orders at today's close, (5) record the closing portfolio value.

Timing rules
------------
* Signals come from the close of session t-1. Fills happen at the close of t.
* Orders must be covered by the cash known at the t-1 close. Cash from sales made
  on day t is NOT used for day t purchases, so no cash is spent twice.
* A lot bought on date d is due on d + 90 calendar days. It is sold at the first
  session on or after the due date that has a quote.
* A held stock with no quote keeps its last observed mark (flagged in the
  events table). No fill is ever invented at a missing price.
* Prices are adjusted close prices, so dividends are already inside the return.
  Dividends are never added again.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict, field

import numpy as np
import pandas as pd


@dataclass
class Lot:
    """One purchase. Units are fractional adjusted-price units."""

    lot_id: int
    symbol: str
    units: float
    signal_date: pd.Timestamp
    entry_date: pd.Timestamp
    entry_price: float
    entry_fee: float
    due_date: pd.Timestamp


@dataclass(frozen=True)
class Config:
    """Rules of the world. These come from the assignment, not from strategy choices."""

    initial_cash: float = 1_000_000.0
    fee_bps: float = 5.0            # cost per side, in basis points of notional (my assumption)
    hold_days: int = 90             # calendar days, set by the assignment

    def __post_init__(self):
        if self.initial_cash <= 0 or self.fee_bps < 0 or self.hold_days < 1:
            raise ValueError("Need positive cash, nonnegative fees, positive holding days")


@dataclass(frozen=True)
class PortfolioView:
    """What the strategy may see. All values are from the previous close."""

    cash: float
    nav: float
    exposure: dict        # symbol -> dollar value of all open lots in that symbol
    # symbol -> calendar days since the NEWEST open lot in that symbol was bought (0 = bought at
    # the previous close). A symbol with no open lot is not in this dict.
    days_since_buy: dict = field(default_factory=dict)


class Strategy(ABC):
    """Decides what to buy. It never trades. It returns dollar budgets."""

    @abstractmethod
    def orders(self, signal_date: pd.Timestamp, session_number: int,
               scores: pd.Series, portfolio: PortfolioView) -> dict:
        """Return {symbol: dollars to spend, fee included}. An empty dict means no trade.

        signal_date is the previous session. scores is the row of scores for that date
        only (NaN = not eligible). session_number counts test sessions from 0.
        """


class TopNStrategy(Strategy):
    """Every `entry_every` sessions, buy the top_n scores in equal parts.

    One entry spends at most cohort_fraction of NAV, and never more than the cash known
    at the signal close. Each name is trimmed so that its total exposure, old lots
    included, stays under max_name_weight of NAV.

    Position awareness (round 2). With `rebuy_wait_days=None` the strategy ignores how old
    its positions are. With a number, for example 45, it skips a stock whose newest open lot
    was bought fewer than 45 calendar days before the signal date. It then takes the next
    best score instead, so the whole budget is still spent, but on other names. A stock
    can be bought again once its newest lot has aged, which means close to the end of the
    holding period. This spreads the risk over more names and avoids stacking a stock
    that we bought a few days ago.
    """

    def __init__(self, top_n=20, cohort_fraction=0.08, entry_every=5, max_name_weight=0.05,
                 rebuy_wait_days=None):
        if top_n < 1 or entry_every < 1:
            raise ValueError("top_n and entry_every must be at least 1")
        if not 0 < cohort_fraction <= 1 or not 0 < max_name_weight <= 1:
            raise ValueError("Fractions must be in (0, 1]")
        if rebuy_wait_days is not None and rebuy_wait_days < 0:
            raise ValueError("rebuy_wait_days must be None or not negative")
        self.top_n = top_n
        self.cohort_fraction = cohort_fraction
        self.entry_every = entry_every
        self.max_name_weight = max_name_weight
        self.rebuy_wait_days = rebuy_wait_days

    def _picks(self, scores, portfolio):
        candidates = scores.dropna()
        if self.rebuy_wait_days is None:
            return candidates.nlargest(self.top_n)
        ranked = candidates.sort_values(ascending=False, kind="mergesort")
        age = portfolio.days_since_buy
        chosen = [s for s in ranked.index if age.get(s, np.inf) >= self.rebuy_wait_days]
        return ranked.loc[chosen[: self.top_n]]

    def orders(self, signal_date, session_number, scores, portfolio):
        if session_number % self.entry_every != 0:
            return {}
        picks = self._picks(scores, portfolio)
        if picks.empty or portfolio.cash <= 0 or portfolio.nav <= 0:
            return {}
        budget = min(portfolio.cash, self.cohort_fraction * portfolio.nav)
        per_name = budget / len(picks)
        out = {}
        for symbol in picks.index:
            room = self.max_name_weight * portfolio.nav - portfolio.exposure.get(symbol, 0.0)
            amount = min(per_name, room)
            if amount > 1e-8:
                out[symbol] = amount
        return out


TRADE_COLUMNS = ["date", "signal_date", "lot_id", "symbol", "side", "price", "units",
                 "notional", "fee", "cash_change", "due_date", "realized_pnl"]


class Ledger:
    """Cash, open lots, and the trade log. It does the accounting and nothing else."""

    def __init__(self, cash: float, fee_rate: float):
        self.initial_cash = float(cash)
        self.cash = float(cash)
        self.fee_rate = float(fee_rate)
        self.lots: dict[int, Lot] = {}
        self.trades: list[dict] = []
        self._next_id = 0

    def buy(self, symbol, date, signal_date, price, dollars, hold_days):
        """Spend `dollars` INCLUDING the fee. Returns the new lot id."""
        if not (np.isfinite(price) and price > 0 and np.isfinite(dollars) and dollars > 0):
            raise ValueError("Price and dollars must be finite and positive")
        if pd.Timestamp(signal_date) >= pd.Timestamp(date):
            raise ValueError("The signal date must be before the fill date")
        if dollars > self.cash + 1e-6:
            raise ValueError("Not enough cash. Cash cannot be spent twice")
        units = dollars / (price * (1 + self.fee_rate))
        notional = units * price
        fee = notional * self.fee_rate
        self.cash -= notional + fee
        if -1e-6 < self.cash < 0:
            self.cash = 0.0
        lot = Lot(self._next_id, symbol, units, pd.Timestamp(signal_date), pd.Timestamp(date),
                  float(price), fee, pd.Timestamp(date) + pd.Timedelta(days=hold_days))
        self.lots[lot.lot_id] = lot
        self._next_id += 1
        self._log(lot, "BUY", date, price, units, notional, fee, -(notional + fee), 0.0)
        return lot.lot_id

    def sell(self, lot_id, date, price):
        """Sell a whole lot. Returns realized PnL after both fees."""
        if not (np.isfinite(price) and price > 0):
            raise ValueError("Price must be finite and positive")
        lot = self.lots[lot_id]
        if pd.Timestamp(date) < lot.due_date:
            raise ValueError("A lot cannot be sold before its due date")
        notional = lot.units * price
        fee = notional * self.fee_rate
        pnl = lot.units * (price - lot.entry_price) - lot.entry_fee - fee
        self.cash += notional - fee
        del self.lots[lot_id]
        self._log(lot, "SELL", date, price, lot.units, notional, fee, notional - fee, pnl)
        return pnl

    def holdings_value(self, marks: dict) -> float:
        return float(sum(lot.units * marks[lot.symbol] for lot in self.lots.values()))

    def exposure(self, marks: dict) -> dict:
        out: dict[str, float] = {}
        for lot in self.lots.values():
            out[lot.symbol] = out.get(lot.symbol, 0.0) + lot.units * marks[lot.symbol]
        return out

    def days_since_buy(self, as_of) -> dict:
        """Calendar days since the newest open lot of each symbol was bought."""
        out: dict[str, int] = {}
        for lot in self.lots.values():
            age = (pd.Timestamp(as_of) - lot.entry_date).days
            out[lot.symbol] = min(out.get(lot.symbol, age), age)
        return out

    def _log(self, lot, side, date, price, units, notional, fee, cash_change, pnl):
        self.trades.append({"date": pd.Timestamp(date), "signal_date": lot.signal_date,
                            "lot_id": lot.lot_id, "symbol": lot.symbol, "side": side,
                            "price": float(price), "units": float(units), "notional": float(notional),
                            "fee": float(fee), "cash_change": float(cash_change),
                            "due_date": lot.due_date, "realized_pnl": float(pnl)})


@dataclass
class Result:
    nav: pd.DataFrame        # one row per session, plus an initial-capital row
    trades: pd.DataFrame
    open_lots: pd.DataFrame
    events: pd.DataFrame     # missing quotes, delayed sales, stale marks


def check_orders(orders, cash: float) -> dict:
    """The backtester refuses orders that break the rules, whatever the strategy wants."""
    if not isinstance(orders, dict):
        raise ValueError("A strategy must return a dict {symbol: dollars}")
    clean = {}
    for symbol, dollars in orders.items():
        if not np.isfinite(dollars) or dollars < 0:
            raise ValueError(f"Order for {symbol} must be finite and not negative (no short selling)")
        if dollars > 1e-8:
            clean[symbol] = float(dollars)
    if sum(clean.values()) > cash + 1e-6:
        raise ValueError("Orders are larger than the cash known at the signal close (no leverage)")
    return clean


def run_backtest(prices: pd.DataFrame, scores: pd.DataFrame, strategy: Strategy, start, end,
                 cfg: Config = Config(), max_stale: int = 5) -> Result:
    """Replay the strategy under the market rules.

    prices: adjusted closes, index = market calendar, columns = symbols, NaN = no quote.
    scores: index = SIGNAL dates, columns = symbols, NaN = not eligible / no score.
    """
    calendar = pd.DatetimeIndex(prices.index)
    if not calendar.is_monotonic_increasing or calendar.has_duplicates:
        raise ValueError("The calendar must be sorted and unique")
    days = calendar[(calendar >= pd.Timestamp(start)) & (calendar <= pd.Timestamp(end))]
    if len(days) == 0:
        raise ValueError("No sessions between start and end")
    first = calendar.get_loc(days[0])
    if first == 0:
        raise ValueError("Need one session before the start for the first signal")

    px = prices.to_numpy(dtype=float)
    col = {s: j for j, s in enumerate(prices.columns)}
    ledger = Ledger(cfg.initial_cash, cfg.fee_bps / 10_000)
    marks: dict[str, float] = {}
    stale_run: dict[str, int] = {}
    events: list[dict] = []
    prev_cash, prev_nav = ledger.cash, ledger.cash
    rows = [{"date": calendar[first - 1], "cash": ledger.cash, "holdings": 0.0, "nav": ledger.cash,
             "open_lots": 0, "fees": 0.0, "traded": 0.0, "stale_marks": 0}]

    for k, day in enumerate(days):
        i = first + k
        signal_date = calendar[i - 1]
        today = px[i]

        # 1. Ask the strategy, using information up to the previous close only.
        row = scores.loc[signal_date] if signal_date in scores.index else pd.Series(dtype=float)
        view = PortfolioView(cash=prev_cash, nav=prev_nav, exposure=ledger.exposure(marks),
                             days_since_buy=ledger.days_since_buy(signal_date))
        orders = check_orders(strategy.orders(signal_date, k, row, view), prev_cash)

        # 2. Update marks for held stocks. A missing quote keeps the last mark.
        stale = 0
        for symbol in {lot.symbol for lot in ledger.lots.values()}:
            price = today[col[symbol]] if symbol in col else np.nan
            if np.isfinite(price):
                marks[symbol], stale_run[symbol] = float(price), 0
            else:
                stale += 1
                stale_run[symbol] = stale_run.get(symbol, 0) + 1
                if stale_run[symbol] == max_stale + 1:
                    events.append({"date": day, "symbol": symbol, "event": "stale_mark_over_limit"})

        n_before = len(ledger.trades)

        # 3. Sell lots that reached maturity, if the stock has a quote today.
        for lot in list(ledger.lots.values()):
            if lot.due_date <= day:
                price = today[col[lot.symbol]] if lot.symbol in col else np.nan
                if np.isfinite(price):
                    ledger.sell(lot.lot_id, day, price)
                else:
                    events.append({"date": day, "symbol": lot.symbol, "event": "sale_delayed_no_quote"})

        # 4. Book the planned purchases at today's close.
        for symbol, dollars in orders.items():
            price = today[col[symbol]] if symbol in col else np.nan
            if not np.isfinite(price):
                events.append({"date": day, "symbol": symbol, "event": "buy_skipped_no_quote"})
                continue
            ledger.buy(symbol, day, signal_date, price, dollars, cfg.hold_days)
            marks[symbol], stale_run[symbol] = float(price), 0

        # 5. Record the closing value.
        holdings = ledger.holdings_value(marks)
        nav = ledger.cash + holdings
        todays = ledger.trades[n_before:]
        rows.append({"date": day, "cash": ledger.cash, "holdings": holdings, "nav": nav,
                     "open_lots": len(ledger.lots), "fees": sum(t["fee"] for t in todays),
                     "traded": sum(t["notional"] for t in todays), "stale_marks": stale})
        prev_cash, prev_nav = ledger.cash, nav

    return Result(
        nav=pd.DataFrame(rows).set_index("date"),
        trades=pd.DataFrame(ledger.trades, columns=TRADE_COLUMNS),
        open_lots=pd.DataFrame([asdict(l) for l in ledger.lots.values()],
                               columns=list(Lot.__dataclass_fields__)),
        events=pd.DataFrame(events, columns=["date", "symbol", "event"]),
    )


def benchmark_curve(bench: pd.Series, nav_index: pd.DatetimeIndex, initial_cash: float) -> pd.Series:
    """Value of initial_cash invested at the first test close and held.

    nav_index[0] is the initial-capital row (the session before the first test
    close). It stays at initial_cash. The purchase happens at nav_index[1].
    """
    bench = bench.sort_index()
    curve = initial_cash * bench.reindex(nav_index) / bench.loc[nav_index[1]]
    curve.iloc[0] = initial_cash
    return curve.rename("benchmark")


def performance(nav: pd.Series, annual_rf: float = 0.02) -> pd.Series:
    """Final value, CAGR, max drawdown, Sharpe (daily returns, 252 sessions a year)."""
    nav = nav.sort_index()
    if len(nav) < 3 or not np.isfinite(nav).all() or (nav <= 0).any():
        raise ValueError("Need at least three positive, finite NAV values")
    returns = nav.pct_change().dropna()
    daily_rf = (1 + annual_rf) ** (1 / 252) - 1
    std = returns.std(ddof=1)
    years = (nav.index[-1] - nav.index[0]).days / 365.25
    return pd.Series({
        "final_value": nav.iloc[-1],
        "cagr": (nav.iloc[-1] / nav.iloc[0]) ** (1 / years) - 1,
        "max_drawdown": (nav / nav.cummax() - 1).min(),
        "sharpe": np.sqrt(252) * (returns - daily_rf).mean() / std if std > 1e-12 else np.nan,
    })


def compare(result: Result, bench: pd.Series, cfg: Config = Config(), annual_rf: float = 0.02) -> pd.DataFrame:
    """Side by side metrics for the strategy and the benchmark."""
    curve = benchmark_curve(bench, result.nav.index, cfg.initial_cash)
    return pd.DataFrame({"strategy": performance(result.nav.nav, annual_rf),
                         "S&P 500 TR": performance(curve, annual_rf)})


def audit(result: Result, cfg: Config = Config(), tol: float = 1e-6) -> dict:
    """Re-check the books. Raises AssertionError on the first failure."""
    nav, trades = result.nav, result.trades
    checks = {}

    assert np.allclose(nav.nav, nav.cash + nav.holdings, atol=tol)
    checks["nav = cash + holdings"] = True

    flows = trades.groupby("date").cash_change.sum().reindex(nav.index, fill_value=0.0)
    assert np.allclose(nav.cash, cfg.initial_cash + flows.cumsum(), atol=1e-4)
    checks["cash = start cash + all trade cash flows"] = True

    assert nav.cash.min() >= -tol
    checks["cash never negative (no borrowing, no double spend)"] = True

    buys, sells = trades[trades.side == "BUY"], trades[trades.side == "SELL"]
    assert (buys.signal_date < buys.date).all()
    checks["every buy comes after its signal date"] = True

    assert (sells.date >= sells.due_date).all()
    assert ((buys.set_index("lot_id").due_date - buys.set_index("lot_id").date)
            == pd.Timedelta(days=cfg.hold_days)).all()
    checks["every sale is on or after 90 calendar days"] = True

    assert sells.lot_id.is_unique and buys.lot_id.is_unique
    assert set(sells.lot_id) | set(result.open_lots.lot_id) == set(buys.lot_id)
    assert not (set(sells.lot_id) & set(result.open_lots.lot_id))
    checks["every lot is sold once or still open at the end"] = True

    return checks
