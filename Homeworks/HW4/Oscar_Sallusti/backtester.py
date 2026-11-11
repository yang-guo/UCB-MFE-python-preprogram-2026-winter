"""Daily long-only backtester for HW4.

Written for this homework (not copied from Lectures/modeling/backtest.py). It
follows the same conventions as the lecture framework: fractional
adjusted-price units, fees charged on both sides with the entry fee inside the
dollar budget, and orders decided at one close and filled at the next.

Timeline for every market session ``d`` (all events happen "at the close"):

1. Matured lots (entry + 90 calendar days <= d) are sold at d's close, if the
   stock has a quote on d. Sale cash is available immediately.
2. Orders that were decided at the previous session's close are filled at d's
   close. If the stock has no quote on d the order is cancelled, not filled.
3. Every open lot is marked to its latest observed close and the day is
   recorded (cash, holdings value, NAV).
4. The strategy sees the portfolio as of d's close and returns dollar budgets
   for tomorrow's buys. The total cannot exceed today's cash, so the same cash
   is never promised twice. No orders are taken on the last session.

Data-gap policy (no invented fills):
- A held stock with no quote keeps its last observed close as its mark and is
  flagged as stale. If it is due for sale it is sold at its next quote.
- If quotes never come back (for example a delisting with no proceeds in the
  data), the lot stays open at its last mark. ``nav_ex_stale`` in the history
  values such lots at zero, so the size of the problem can be reported.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping
import warnings

import duckdb
import numpy as np
import pandas as pd

CASH_TOLERANCE = 1e-6  # dollars; absorbs floating-point dust


@dataclass
class Lot:
    """One purchase. Several lots of the same stock may be open at once."""

    lot_id: int
    symbol: str
    units: float
    signal_date: pd.Timestamp
    entry_date: pd.Timestamp
    entry_price: float
    entry_fee: float
    due_date: pd.Timestamp


@dataclass(frozen=True)
class PortfolioView:
    """What the strategy is allowed to see when it plans tomorrow's orders."""

    date: pd.Timestamp
    cash: float
    nav: float
    holdings: pd.Series  # market value by symbol, at today's marks
    lots: tuple[Lot, ...]


@dataclass(frozen=True)
class BacktestResult:
    history: pd.DataFrame    # one row per session
    fills: pd.DataFrame      # every buy and sell
    cancelled: pd.DataFrame  # orders that could not be filled
    open_lots: pd.DataFrame  # lots still open on the last session
    params: dict


# A strategy is any callable: PortfolioView -> {symbol: dollar budget}.
Strategy = Callable[[PortfolioView], Mapping[str, float]]


class Backtester:
    def __init__(self, prices: pd.DataFrame, initial_cash: float = 1_000_000,
                 fee_bps: float = 10, holding_days: int = 90,
                 stale_flag_sessions: int = 5):
        """``prices``: adjusted closes, index = market calendar, one column per
        symbol, NaN where the stock has no quote that session."""
        if not prices.index.is_monotonic_increasing or prices.index.has_duplicates:
            raise ValueError("Price index must be a sorted market calendar")
        if (prices <= 0).any().any():
            raise ValueError("Prices must be positive; quarantine bad rows first")
        if initial_cash <= 0 or fee_bps < 0 or holding_days < 1:
            raise ValueError("Bad capital, fee, or holding period")
        self.prices = prices
        self.initial_cash = float(initial_cash)
        self.fee_rate = fee_bps / 10_000
        self.holding_days = holding_days
        self.stale_flag_sessions = stale_flag_sessions

    def run(self, strategy: Strategy, start, end) -> BacktestResult:
        start, end = pd.Timestamp(start), pd.Timestamp(end)
        window = self.prices.loc[start:end]
        if window.empty:
            raise ValueError("No market sessions in the requested window")
        dates = window.index
        values = window.to_numpy(dtype=float)
        col = {s: j for j, s in enumerate(window.columns)}

        cash = self.initial_cash
        lots: dict[int, Lot] = {}
        next_id = 0
        pending: list[tuple[str, float, pd.Timestamp]] = []  # symbol, budget, signal date
        last_px: dict[str, float] = {}       # last observed close of held symbols
        last_quote_i: dict[str, int] = {}    # session index of that close
        fills, cancelled, history = [], [], []

        for i, d in enumerate(dates):
            row = values[i]

            def quote(symbol):
                p = row[col[symbol]]
                return p if np.isfinite(p) else None

            # Refresh marks of held symbols that trade today.
            for symbol in {lot.symbol for lot in lots.values()}:
                p = quote(symbol)
                if p is not None:
                    last_px[symbol], last_quote_i[symbol] = p, i

            # 1. Sell matured lots that have a quote today.
            for lot in sorted(lots.values(), key=lambda x: x.lot_id):
                if d < lot.due_date:
                    continue
                p = quote(lot.symbol)
                if p is None:
                    continue  # overdue; sold at next quote
                gross = lot.units * p
                fee = gross * self.fee_rate
                cash += gross - fee
                pnl = gross - fee - lot.units * lot.entry_price - lot.entry_fee
                fills.append(dict(date=d, side="SELL", lot_id=lot.lot_id, symbol=lot.symbol,
                                  units=lot.units, price=p, notional=gross, fee=fee,
                                  cash_change=gross - fee, signal_date=lot.signal_date,
                                  entry_date=lot.entry_date, due_date=lot.due_date,
                                  realized_pnl=pnl))
                del lots[lot.lot_id]

            # 2. Fill yesterday's orders at today's close.
            for symbol, budget, signal_date in pending:
                p = quote(symbol)
                if p is None:
                    cancelled.append(dict(date=d, signal_date=signal_date, symbol=symbol,
                                          budget=budget, reason="no quote at fill close"))
                    continue
                if budget > cash + CASH_TOLERANCE:  # cannot happen if step 4 is honoured
                    raise RuntimeError(f"{d.date()}: budget {budget:.2f} exceeds cash {cash:.2f}")
                units = budget / (p * (1 + self.fee_rate))
                gross = units * p
                fee = gross * self.fee_rate
                cash = max(0.0, cash - gross - fee)
                lot = Lot(next_id, symbol, units, signal_date, d, p, fee,
                          d + pd.Timedelta(days=self.holding_days))
                lots[next_id] = lot
                next_id += 1
                last_px[symbol], last_quote_i[symbol] = p, i
                fills.append(dict(date=d, side="BUY", lot_id=lot.lot_id, symbol=symbol,
                                  units=units, price=p, notional=gross, fee=fee,
                                  cash_change=-(gross + fee), signal_date=signal_date,
                                  entry_date=d, due_date=lot.due_date, realized_pnl=0.0))
            pending = []

            # 3. Mark to market.
            holdings: dict[str, float] = {}
            stale_value = 0.0
            n_stale = 0
            for lot in lots.values():
                value = lot.units * last_px[lot.symbol]
                holdings[lot.symbol] = holdings.get(lot.symbol, 0.0) + value
                if i - last_quote_i[lot.symbol] > self.stale_flag_sessions:
                    stale_value += value
                    n_stale += 1
            holdings_value = float(sum(holdings.values()))
            nav = cash + holdings_value
            history.append(dict(date=d, cash=cash, holdings_value=holdings_value, nav=nav,
                                n_lots=len(lots), n_symbols=len(holdings),
                                stale_lots=n_stale, stale_value=stale_value,
                                nav_ex_stale=nav - stale_value))

            # 4. Plan tomorrow's orders (none on the last session).
            if i == len(dates) - 1:
                continue
            view = PortfolioView(d, cash, nav, pd.Series(holdings, dtype=float),
                                 tuple(lots.values()))
            orders = dict(strategy(view) or {})
            for symbol, budget in orders.items():
                if symbol not in col:
                    raise KeyError(f"Unknown symbol {symbol!r}")
                if not np.isfinite(budget) or budget < 0:
                    raise ValueError(f"Bad budget {budget!r} for {symbol}")
            total = sum(orders.values())
            if total > cash + CASH_TOLERANCE:
                raise ValueError(f"{d.date()}: orders total {total:,.2f} > cash {cash:,.2f}")
            pending = [(s, float(b), d) for s, b in orders.items() if b > 0]

        open_rows = []
        for lot in lots.values():
            open_rows.append(dict(lot_id=lot.lot_id, symbol=lot.symbol, units=lot.units,
                                  entry_date=lot.entry_date, entry_price=lot.entry_price,
                                  due_date=lot.due_date, mark=last_px[lot.symbol],
                                  last_quote=dates[last_quote_i[lot.symbol]],
                                  value=lot.units * last_px[lot.symbol]))
        fill_cols = ["date", "side", "lot_id", "symbol", "units", "price", "notional", "fee",
                     "cash_change", "signal_date", "entry_date", "due_date", "realized_pnl"]
        return BacktestResult(
            history=pd.DataFrame(history).set_index("date"),
            fills=pd.DataFrame(fills, columns=fill_cols),
            cancelled=pd.DataFrame(cancelled, columns=["date", "signal_date", "symbol",
                                                       "budget", "reason"]),
            open_lots=pd.DataFrame(open_rows, columns=["lot_id", "symbol", "units", "entry_date",
                                                       "entry_price", "due_date", "mark",
                                                       "last_quote", "value"]),
            params=dict(initial_cash=self.initial_cash, fee_bps=self.fee_rate * 10_000,
                        holding_days=self.holding_days, start=dates[0], end=dates[-1]),
        )


# ---------------------------------------------------------------- analytics

def performance_metrics(nav: pd.Series, rf_annual: float = 0.02) -> pd.Series:
    """Final value, CAGR, max drawdown, and Sharpe of a daily value series.

    CAGR uses elapsed calendar days / 365.25. Sharpe uses daily simple returns
    minus the daily equivalent of ``rf_annual`` and annualises with sqrt(252).
    """
    nav = nav.dropna()
    years = (nav.index[-1] - nav.index[0]).days / 365.25
    cagr = (nav.iloc[-1] / nav.iloc[0]) ** (1 / years) - 1
    drawdown = nav / nav.cummax() - 1
    rf_daily = (1 + rf_annual) ** (1 / 252) - 1
    excess = nav.pct_change().dropna() - rf_daily
    sharpe = np.sqrt(252) * excess.mean() / excess.std(ddof=1)
    return pd.Series({"final_value": nav.iloc[-1], "cagr": cagr,
                      "max_drawdown": drawdown.min(), "sharpe": sharpe,
                      "volatility": nav.pct_change().std(ddof=1) * np.sqrt(252)})


def benchmark_nav(benchmark: pd.Series, start, end, capital: float = 1_000_000) -> pd.Series:
    """Value of ``capital`` invested in the benchmark at the first close >= start."""
    series = benchmark.loc[pd.Timestamp(start):pd.Timestamp(end)]
    return capital * series / series.iloc[0]


def check_accounting(result: BacktestResult, prices: pd.DataFrame) -> pd.Series:
    """Independent re-computation of the books; returns the largest errors.

    1. Cash: initial cash + cumulative fill cash flows == recorded cash.
    2. Holdings: rebuild positions from the fill log and mark them with
       forward-filled prices; compare with the recorded holdings value.
    3. Timing: every buy fills strictly after its signal date, on the next
       session; every sale happens on/after the due date.
    """
    h, f = result.history, result.fills
    flows = f.groupby("date").cash_change.sum().reindex(h.index, fill_value=0.0)
    cash_err = (result.params["initial_cash"] + flows.cumsum() - h.cash).abs().max()

    marks = prices.loc[h.index[0]:h.index[-1]].ffill()
    signed = f.assign(u=np.where(f.side == "BUY", f.units, -f.units))
    units = (signed.pivot_table(index="date", columns="symbol", values="u", aggfunc="sum")
             .reindex(h.index, fill_value=0.0).fillna(0.0).cumsum())
    rebuilt = (units * marks[units.columns]).where(units.abs() > 1e-12, 0.0).sum(axis=1)
    holdings_err = (rebuilt - h.holdings_value).abs().max()

    cal = prices.index
    buys, sells = f[f.side == "BUY"], f[f.side == "SELL"]
    next_session = cal[cal.searchsorted(buys.signal_date, side="right")]
    bad_buy_timing = int((buys.date.to_numpy() != next_session.to_numpy()).sum())
    bad_sell_timing = int((sells.date < sells.due_date).sum())
    return pd.Series({"max_cash_error": cash_err, "max_holdings_error": holdings_err,
                      "min_cash": h.cash.min(), "bad_buy_timing": bad_buy_timing,
                      "bad_sell_timing": bad_sell_timing,
                      "nav_identity_error": (h.cash + h.holdings_value - h.nav).abs().max()})


# ------------------------------------------------------------------ data

def load_prices(database: str | Path, start=None, end="2026-09-30", symbols=None):
    """Wide adjusted-close matrix on the supplied market calendar, plus the benchmark.

    Applies the data guide's quarantine policy (as in Lectures/modeling/
    course_data.py:load_data): non-positive adjusted prices and rows dated
    off the market calendar are dropped with a warning, never filled. Rows
    with zero volume are also dropped (no trade; see features.quarantine_quotes).
    """
    where, params = ["date <= ?"], [end]
    if start is not None:
        where.append("date >= ?"); params.append(start)
    if symbols is not None:
        where.append("symbol IN (SELECT unnest(?))"); params.append(list(symbols))
    with duckdb.connect(str(database), read_only=True) as con:
        long = con.execute(f"SELECT symbol, date, adj_close, volume FROM prices WHERE {' AND '.join(where)}",
                           params).fetchdf()
        bench = con.execute("SELECT date, adj_close FROM benchmark WHERE date <= ?"
                            + (" AND date >= ?" if start else "") + " ORDER BY date",
                            [end] + ([start] if start else [])).fetchdf()
    long["date"] = pd.to_datetime(long.date)
    bench["date"] = pd.to_datetime(bench.date)
    calendar = pd.DatetimeIndex(bench.date)
    bad = ~np.isfinite(long.adj_close) | (long.adj_close <= 0)
    off = ~long.date.isin(calendar)
    no_trade = ~(long.volume > 0) & ~bad & ~off
    bad |= no_trade
    for mask, what in [(bad, "non-positive adjusted prices or zero volume"),
                       (off, "off-calendar dates")]:
        if mask.any():
            warnings.warn(f"Quarantined {int(mask.sum())} price rows with {what}", stacklevel=2)
    long = long[~bad & ~off]
    wide = long.pivot(index="date", columns="symbol", values="adj_close").reindex(calendar)
    wide.index.name = "date"
    return wide.astype("float64"), bench.set_index("date").adj_close.rename("^SP500TR")
