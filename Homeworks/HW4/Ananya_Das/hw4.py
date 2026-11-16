"""HW4: stock-return model, strategy, and my own backtester (Ananya Das).

Usage, from this folder:
    ../../../.venv/bin/python hw4.py dev     # validation + 2012-2015 backtest (data through Mar 2016)
    ../../../.venv/bin/python hw4.py final   # final test; refuses unless RUN_FINAL_TEST is True

Reused course code (Lectures/modeling/course_data.py): the loader follows load_data /
load_membership; the features follow price_features, financial_events, and make_labels.
The backtester's plan -> execute -> mark structure follows Lectures/modeling/backtest.py.
"""

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import warnings

import duckdb
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

plt.switch_backend("Agg")


# Settings

HW_DIR = Path(__file__).resolve().parent
REPO_ROOT = HW_DIR.parents[2]
DATA_DIR = REPO_ROOT / "Lectures" / "modeling" / "data"
CLASS_DB = DATA_DIR / "equities_class.duckdb"
FULL_DB = DATA_DIR / "equities_top2000.duckdb"
DATABASE = CLASS_DB  # switch to FULL_DB for the real experiment
OUTPUT_DIR = HW_DIR / "outputs"

WARMUP_START = pd.Timestamp("2000-01-01")
DEV_START = pd.Timestamp("2010-01-01")
DEV_END = pd.Timestamp("2015-12-31")
BUFFER_END = pd.Timestamp("2016-03-31")
TEST_START = pd.Timestamp("2016-04-01")
TEST_END = pd.Timestamp("2026-09-30")

# The final test was run once before submitting homework (results in final_results.md).
# Keep False: any re-run needs fresh approval and must be reported alongside the original.
RUN_FINAL_TEST = False

HOLDING_DAYS = 90
INITIAL_CAPITAL = 1_000_000
FEE_BPS = 10
FEE_SENSITIVITY_BPS = (5, 20)
TOP_N = 20
COHORT_FRACTION = 1 / 3
NAME_CAP = 0.05
SECTOR_CAP = 0.25
MIN_PRICE = 5.0
LIQUIDITY_WINDOW = 63
LIQUIDITY_MIN_QUOTES = 60
RAMP_MONTHS = 3
MAX_STALE_SESSIONS = 5
FEATURE_START = pd.Timestamp("2008-01-01")  # lookback for 2010 features
VALIDATION_YEARS = (2012, 2013, 2014, 2015)
RIDGE_ALPHAS = (10, 1_000, 100_000, 10_000_000)  # fixed before looking at any results
TARGET_WINSOR = 0.01  # clip each date's training targets at its 1st/99th percentiles
MAX_DAILY_MOVE = 4.0  # skip stocks with a one-day move above x4 or below /4 in the last 63 sessions
ANNUAL_RISK_FREE = 0.02
CASH_INTEREST = 0.0


# Data loading with the final-test date lock (adapted from course_data.load_data / load_membership)

class LookaheadError(RuntimeError):
    pass


def allowed_end():
    return TEST_END if RUN_FINAL_TEST else BUFFER_END


def check_end(end):
    end = pd.Timestamp(end)
    if pd.isna(end):
        raise ValueError("end must be a real date")
    limit = allowed_end()
    if end > limit:
        reason = ("the final test has not been approved (RUN_FINAL_TEST is False)"
                  if not RUN_FINAL_TEST else "the data stops at the test end")
        raise LookaheadError(f"Refusing to load data through {end.date()}: the limit is "
                             f"{limit.date()} because {reason}.")
    return end


def check_no_rows_after(frame, columns, end, name):
    for column in columns:
        latest = frame[column].max()
        if pd.notna(latest) and latest > end:
            raise LookaheadError(f"{name}.{column} contains {latest.date()}, after {end.date()}")


def _to_datetime(frame, columns):
    for column in columns:
        frame[column] = pd.to_datetime(frame[column]).astype("datetime64[ns]")


def load_data(end, database=None, symbols=None):
    """Return prices, fundamentals, universe, benchmark, membership through `end`."""
    end = check_end(end)  # before the database is even opened
    database = DATABASE if database is None else database
    if not database.is_file():
        raise FileNotFoundError(f"Download the database from bCourses to {database}")

    symbol_filter = " AND symbol IN (SELECT unnest(?))" if symbols is not None else ""
    symbol_params = [list(symbols)] if symbols is not None else []
    with duckdb.connect(str(database), read_only=True) as con:
        prices = con.execute(f"""
            SELECT symbol, date, close, adj_close, volume FROM prices
            WHERE date <= ? {symbol_filter} ORDER BY symbol, date""",
            [end] + symbol_params).fetchdf()
        # Filter on filing_date: that's when the numbers became public.
        fundamentals = con.execute(f"""
            SELECT symbol, fiscal_date, period, filing_date, reported_currency,
                   revenue, net_income, operating_income, total_assets, total_debt,
                   cash_and_equivalents, free_cash_flow, operating_cash_flow
            FROM fundamentals
            WHERE filing_date IS NOT NULL AND filing_date <= ? AND fiscal_date <= ?
            {symbol_filter} ORDER BY symbol, fiscal_date""",
            [end, end] + symbol_params).fetchdf()
        # Only symbol and sector: active/delisted flags would leak the future.
        universe = con.execute(f"""
            SELECT symbol, sector FROM universe WHERE true {symbol_filter}
            ORDER BY symbol""", symbol_params).fetchdf()
        benchmark = con.execute("""
            SELECT date, adj_close FROM benchmark WHERE date <= ? ORDER BY date""",
            [end]).fetchdf()
        membership = con.execute(f"""
            SELECT symbol, quarter_start, quarter_end, selection_date,
                   market_cap_proxy, universe_rank
            FROM universe_membership
            WHERE quarter_start <= ? AND selection_date <= ? {symbol_filter}
            ORDER BY quarter_start, universe_rank""",
            [end, end] + symbol_params).fetchdf()

    _to_datetime(prices, ["date"])
    _to_datetime(fundamentals, ["fiscal_date", "filing_date"])
    _to_datetime(benchmark, ["date"])
    _to_datetime(membership, ["quarter_start", "quarter_end", "selection_date"])

    check_no_rows_after(prices, ["date"], end, "prices")
    check_no_rows_after(fundamentals, ["fiscal_date", "filing_date"], end, "fundamentals")
    check_no_rows_after(benchmark, ["date"], end, "benchmark")
    check_no_rows_after(membership, ["quarter_start", "selection_date"], end, "membership")

    assert not prices.duplicated(["symbol", "date"]).any(), "duplicate price rows"
    assert not benchmark.date.duplicated().any(), "duplicate benchmark dates"
    assert (np.isfinite(benchmark.adj_close) & benchmark.adj_close.gt(0)).all()
    assert not membership.duplicated(["symbol", "quarter_start"]).any(), "duplicate membership"

    invalid = ~np.isfinite(prices.adj_close) | prices.adj_close.le(0)
    if invalid.any():
        warnings.warn(f"Quarantining {int(invalid.sum())} non-positive/non-finite adjusted prices.",
                      stacklevel=2)
        prices = prices.loc[~invalid].copy()
    off_calendar = ~prices.date.isin(benchmark.date)
    if off_calendar.any():
        warnings.warn(f"Quarantining {int(off_calendar.sum())} price rows not on the market calendar.",
                      stacklevel=2)
        prices = prices.loc[~off_calendar].copy()

    return prices, fundamentals, universe, benchmark, membership


# Backtester (structure follows Lectures/modeling/backtest.py; implementation is my own)

CASH_TOLERANCE = 1e-6


class InsufficientCashError(ValueError):
    pass


class OrderError(ValueError):
    pass


@dataclass(frozen=True)
class Lot:
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
        return self.units * self.entry_price + self.entry_fee


@dataclass(frozen=True)
class PortfolioSnapshot:
    date: pd.Timestamp
    cash: float
    nav: float
    holdings: pd.Series
    sale_proceeds: float = 0.0


class Portfolio:
    def __init__(self, initial_cash=INITIAL_CAPITAL, fee_bps=FEE_BPS):
        if not initial_cash > 0 or not fee_bps >= 0:
            raise ValueError("Capital must be positive and fees nonnegative")
        self.initial_cash = float(initial_cash)
        self.cash = float(initial_cash)
        self.fee_rate = fee_bps / 10_000
        self.lots = {}
        self.fills = []
        self.realized_pnl = 0.0
        self._next_id = 0

    def buy(self, symbol, budget, price, date, signal_date, holding_days=HOLDING_DAYS):
        """Spend `budget` (fee included) on `symbol` at `price`."""
        if not (np.isfinite(budget) and budget > 0 and np.isfinite(price) and price > 0):
            raise ValueError("Budget and price must be positive and finite")
        if not pd.Timestamp(signal_date) < pd.Timestamp(date):
            raise ValueError("The signal must come before the fill")
        units = budget / (price * (1 + self.fee_rate))
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
        values = {}
        for lot in self.lots.values():
            values[lot.symbol] = values.get(lot.symbol, 0.0) + lot.units * marks[lot.symbol]
        return pd.Series(values, dtype=float)

    def snapshot(self, date, marks, sale_proceeds=0.0):
        h = self.holdings(marks)
        return PortfolioSnapshot(pd.Timestamp(date), self.cash, self.cash + h.sum(), h, sale_proceeds)

    def reconcile(self, marks):
        """Starting cash + realized + unrealized PnL must equal NAV."""
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
    history: pd.DataFrame
    fills: pd.DataFrame
    open_lots: pd.DataFrame
    diagnostics: pd.DataFrame
    summary: pd.Series


class Backtester:
    """`strategy` needs generate_orders(signal_date, execution_date, snapshot) -> Series of budgets."""

    def __init__(self, prices, calendar, initial_cash=INITIAL_CAPITAL,
                 fee_bps=FEE_BPS, holding_days=HOLDING_DAYS,
                 max_stale_sessions=MAX_STALE_SESSIONS):
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
        self.quotes = prices.pivot(index="date", columns="symbol", values="adj_close").reindex(self.calendar)
        self.initial_cash = initial_cash
        self.fee_bps = fee_bps
        self.holding_days = holding_days
        self.max_stale = max_stale_sessions
        self.portfolio = None

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
        end = check_end(end)
        start = pd.Timestamp(start)
        last_entry = end if last_entry is None else pd.Timestamp(last_entry)
        cal = self.calendar
        days = cal[(cal >= start) & (cal <= end)]
        if not len(days) or cal.get_loc(days[0]) == 0:
            raise ValueError("Need at least one session, plus one session before it for signals")

        pf = self.portfolio = Portfolio(self.initial_cash, self.fee_bps)
        last_price, last_seen = {}, {}
        frozen, delayed_logged, diagnostics = set(), set(), []

        prev = cal[cal.get_loc(days[0]) - 1]
        snapshot = pf.snapshot(prev, last_price)
        history = [self._history_row(prev, pf, last_price, frozen, 0)]

        for day in days:
            i = cal.get_loc(day)
            signal_date = cal[i - 1]

            if day <= last_entry:
                orders = strategy.generate_orders(signal_date, day, snapshot)
                self._validate_orders(orders, snapshot.cash)
            else:
                orders = pd.Series(dtype=float)

            # Held stocks with no quote keep their last price; past max_stale
            # sessions they're frozen (can't be sold) until quotes come back.
            observed = self.quotes.iloc[i].dropna()
            stale_today = 0
            for sym in sorted({lot.symbol for lot in pf.lots.values()}):
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

            fills_before_sales = len(pf.fills)
            for lot in list(pf.lots.values()):
                if day < lot.due_date:
                    continue
                if lot.symbol in observed.index:
                    first_exit_session = cal[cal.searchsorted(lot.due_date)]
                    pf.sell(lot.lot_id, observed[lot.symbol], day, delayed=day > first_exit_session)
                elif lot.lot_id not in delayed_logged:
                    delayed_logged.add(lot.lot_id)
                    diagnostics.append((day, lot.symbol, "exit_delayed"))

            for sym, budget in orders.items():
                if budget <= CASH_TOLERANCE:
                    continue
                if sym not in observed.index:
                    diagnostics.append((day, sym, "rejected_no_quote"))
                    continue
                pf.buy(sym, budget, observed[sym], day, signal_date, self.holding_days)
                last_price[sym], last_seen[sym] = observed[sym], i

            proceeds = sum(f["cash_change"] for f in pf.fills[fills_before_sales:] if f["side"] == "SELL")
            snapshot = pf.snapshot(day, last_price, proceeds)
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
                "frozen_value": frozen_value, "nav_frozen_at_zero": nav - frozen_value,
                "open_lots": len(pf.lots), "stale_marks": stale_today}


# Performance metrics and the ^SP500TR benchmark

TRADING_DAYS_PER_YEAR = 252
DAYS_PER_YEAR = 365.25


def _check_nav(nav):
    if not isinstance(nav, pd.Series) or not isinstance(nav.index, pd.DatetimeIndex):
        raise ValueError("nav must be a pandas Series indexed by date")
    if len(nav) < 2 or not nav.index.is_monotonic_increasing or not nav.index.is_unique:
        raise ValueError("nav needs at least two values on sorted, unique dates")
    if not (np.isfinite(nav) & nav.gt(0)).all():
        raise ValueError("nav values must be positive and finite")
    return nav.astype(float)


def daily_returns(nav):
    nav = _check_nav(nav)
    return (nav / nav.shift(1) - 1).iloc[1:]


def daily_risk_free(annual_rf=ANNUAL_RISK_FREE):
    return (1 + annual_rf) ** (1 / TRADING_DAYS_PER_YEAR) - 1


def years_between(nav):
    nav = _check_nav(nav)
    return (nav.index[-1] - nav.index[0]).days / DAYS_PER_YEAR


def cagr(nav):
    nav = _check_nav(nav)
    return (nav.iloc[-1] / nav.iloc[0]) ** (1 / years_between(nav)) - 1


def drawdowns(nav):
    nav = _check_nav(nav)
    return nav / nav.cummax() - 1


def max_drawdown(nav):
    return drawdowns(nav).min()


def annualized_volatility(nav):
    return daily_returns(nav).std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)


def sharpe(nav, annual_rf=ANNUAL_RISK_FREE):
    r = daily_returns(nav)
    std = r.std(ddof=1)
    if not std > 1e-12:
        return np.nan
    return np.sqrt(TRADING_DAYS_PER_YEAR) * (r - daily_risk_free(annual_rf)).mean() / std


def performance(nav, annual_rf=ANNUAL_RISK_FREE):
    nav = _check_nav(nav)
    dd = drawdowns(nav)
    trough = dd.idxmin()
    return pd.Series({
        "start": nav.index[0], "end": nav.index[-1], "years": years_between(nav),
        "start_value": nav.iloc[0], "final_value": nav.iloc[-1],
        "total_return": nav.iloc[-1] / nav.iloc[0] - 1,
        "CAGR": cagr(nav),
        "max_drawdown": dd.min(),
        "max_drawdown_peak": nav.loc[:trough].idxmax(),
        "max_drawdown_trough": trough,
        "annualized_volatility": annualized_volatility(nav),
        "Sharpe": sharpe(nav, annual_rf),
    }, dtype=object)


def benchmark_nav(benchmark, dates, capital=INITIAL_CAPITAL):
    """dates[0] is the pre-trading row; the index is bought at the dates[1] close."""
    prices = benchmark.set_index("date").adj_close if isinstance(benchmark, pd.DataFrame) else benchmark
    prices = prices.copy()
    prices.index = pd.to_datetime(prices.index)
    dates = pd.DatetimeIndex(dates)
    if len(dates) < 2:
        raise ValueError("Need the pre-trading row plus at least one session")
    missing = dates[1:].difference(prices.index)
    if len(missing):
        raise ValueError(f"No benchmark price on {len(missing)} date(s), e.g. {missing[0].date()}")
    held = prices.reindex(dates[1:])
    if not (np.isfinite(held) & held.gt(0)).all():
        raise ValueError("Benchmark prices must be positive and finite")
    units = capital / held.iloc[0]
    nav = pd.concat([pd.Series([float(capital)], index=dates[:1]), units * held])
    nav.name = "benchmark_nav"
    return nav


def compare(strategy_nav, benchmark, annual_rf=ANNUAL_RISK_FREE, capital=INITIAL_CAPITAL):
    bench = benchmark_nav(benchmark, strategy_nav.index, capital)
    table = pd.DataFrame({"Strategy": performance(strategy_nav, annual_rf),
                          "^SP500TR buy-and-hold": performance(bench, annual_rf)})
    numeric = ["final_value", "total_return", "CAGR", "max_drawdown", "annualized_volatility", "Sharpe"]
    table["Strategy minus benchmark"] = None
    table.loc[numeric, "Strategy minus benchmark"] = (
        table.loc[numeric, "Strategy"].astype(float) - table.loc[numeric, "^SP500TR buy-and-hold"].astype(float))
    return table


# Purchase schedule, eligibility screens, and caps

def liquid_symbols(quotes, signal_date, window=LIQUIDITY_WINDOW,
                   min_quotes=LIQUIDITY_MIN_QUOTES):
    """`quotes` must have a row for every session, with NaN for missing prices."""
    past = quotes.loc[:signal_date].tail(window)
    if len(past) < window:
        return set()
    counts = past.notna().sum()
    return set(counts.index[counts >= min_quotes])


def universe_members(membership, signal_date, execution_date):
    quarter = pd.Timestamp(execution_date).to_period("Q").start_time
    rows = membership.quarter_start.eq(quarter) & membership.selection_date.le(signal_date)
    return set(membership.loc[rows, "symbol"])


def capped_allocation(ranked, budget, holdings, sectors, nav, top_n=TOP_N,
                      name_cap=NAME_CAP, sector_cap=SECTOR_CAP):
    """Equal slices of budget / top_n down the ranked list.

    A stock is skipped (and we move down the list) if a full slice would push
    it past name_cap or its sector past sector_cap of NAV.
    """
    per_name = budget / top_n
    by_name = holdings.to_dict()
    by_sector = holdings.groupby(sectors.reindex(holdings.index).fillna("Unknown")).sum().to_dict()
    orders = {}
    for sym in ranked:
        if len(orders) == top_n:
            break
        sector = sectors.get(sym, "Unknown")
        if (by_name.get(sym, 0.0) + per_name > name_cap * nav
                or by_sector.get(sector, 0.0) + per_name > sector_cap * nav):
            continue
        orders[sym] = per_name
        by_name[sym] = by_name.get(sym, 0.0) + per_name
        by_sector[sector] = by_sector.get(sector, 0.0) + per_name
    return pd.Series(orders, dtype=float)


class CohortStrategy(ABC):
    """Use a fresh instance per backtest; it counts the ramp-up buys."""

    def __init__(self, membership, quotes, calendar, ramp_months=RAMP_MONTHS,
                 cohort_fraction=COHORT_FRACTION):
        self.membership = membership
        self.quotes = quotes.reindex(pd.DatetimeIndex(calendar))
        self.ramp_months = ramp_months
        self.cohort_fraction = cohort_fraction
        self.ramp_buys_done = 0
        self.log = []

    def budget(self, signal_date, execution_date, snapshot):
        if self.ramp_buys_done < self.ramp_months:
            if execution_date.to_period("M") != signal_date.to_period("M"):
                self.ramp_buys_done += 1
                return min(snapshot.cash, snapshot.nav * self.cohort_fraction), "ramp-up"
            return 0.0, None
        if snapshot.sale_proceeds > 0:
            return min(snapshot.cash, snapshot.sale_proceeds), "reinvest"
        return 0.0, None

    def eligible(self, signal_date, execution_date):
        return (universe_members(self.membership, signal_date, execution_date)
                & liquid_symbols(self.quotes, signal_date))

    @abstractmethod
    def pick(self, signal_date, eligible):
        """Return the symbols to buy, chosen from `eligible`."""

    def allocate(self, signal_date, budget, snapshot, eligible):
        names = sorted(self.pick(signal_date, eligible))
        return pd.Series(budget / len(names), index=names) if names else pd.Series(dtype=float)

    def generate_orders(self, signal_date, execution_date, snapshot):
        budget, reason = self.budget(signal_date, execution_date, snapshot)
        if budget <= 0:
            return pd.Series(dtype=float)
        orders = self.allocate(signal_date, budget, snapshot, self.eligible(signal_date, execution_date))
        if orders.empty:
            return orders
        self.log.append((execution_date, reason, budget))
        return orders


# Features and 90-day targets (adapted from course_data.price_features / financial_events / make_labels)

PRICE_FEATURES = ["mom_21", "mom_252_21", "vol_63", "drawdown_252"]
FINANCIAL_FEATURES = ["operating_roa", "accruals_to_assets", "debt_to_assets"]
FEATURES = PRICE_FEATURES + FINANCIAL_FEATURES + ["log_cap"]
MAX_EXIT_DELAY = 5
STALE_FINANCIAL_DAYS = 550


def financial_events(fundamentals):
    f = fundamentals.loc[fundamentals.period.eq("quarter")
                         & fundamentals.filing_date.gt(fundamentals.fiscal_date)].copy()
    # A filing date has no time of day, so use it from the next day on.
    f["available_date"] = f.filing_date + pd.Timedelta(days=1)
    assets = f.total_assets.where(f.total_assets.gt(0))
    f["operating_roa"] = f.operating_income / assets
    f["accruals_to_assets"] = (f.net_income - f.operating_cash_flow) / assets
    f["debt_to_assets"] = f.total_debt / assets
    f[FINANCIAL_FEATURES] = f[FINANCIAL_FEATURES].replace([np.inf, -np.inf], np.nan)
    f = f.sort_values(["symbol", "available_date", "fiscal_date"])
    # A late filing for an older quarter must not replace a newer one.
    f = f.loc[f.fiscal_date.eq(f.groupby("symbol").fiscal_date.cummax())]
    f = f.drop_duplicates(["symbol", "available_date"], keep="last")
    return f[["symbol", "fiscal_date", "available_date"] + FINANCIAL_FEATURES].reset_index(drop=True)


class FeatureBuilder:
    """Holds wide price tables; features_on(t) uses rows up to t only."""

    def __init__(self, prices, fundamentals, membership, universe, calendar, start=None):
        cal = pd.DatetimeIndex(calendar)
        if start is not None:
            cal = cal[cal >= pd.Timestamp(start)]
        self.calendar = cal
        self.adj = prices.pivot(index="date", columns="symbol", values="adj_close").reindex(cal)
        self.raw = prices.pivot(index="date", columns="symbol", values="close").reindex(cal)
        returns = self.adj.pct_change(fill_method=None)
        # Bad prints (e.g. $0.19 -> $26,000 overnight) show up as huge one-day ratios,
        # including across a gap, so compare with the last observed price.
        ratio = self.adj / self.adj.ffill().shift(1)
        jump = (ratio > MAX_DAILY_MOVE) | (ratio < 1 / MAX_DAILY_MOVE)
        self.recent_jump = jump.rolling(LIQUIDITY_WINDOW, min_periods=1).max().astype(bool)
        self.wide = {
            "mom_21": self.adj / self.adj.shift(21) - 1,
            "mom_252_21": self.adj.shift(21) / self.adj.shift(252) - 1,
            "vol_63": returns.rolling(63, min_periods=60).std() * np.sqrt(252),
            "drawdown_252": self.adj / self.adj.rolling(252, min_periods=240).max() - 1,
        }
        self.events = financial_events(fundamentals)
        self.membership = membership
        self.sectors = universe.set_index("symbol").sector.fillna("Unknown")

    def next_session(self, t):
        i = self.calendar.get_loc(pd.Timestamp(t))
        return self.calendar[i + 1] if i + 1 < len(self.calendar) else None

    def eligible(self, t, execution_date=None):
        t = pd.Timestamp(t)
        execution_date = execution_date if execution_date is not None else self.next_session(t)
        if execution_date is None:
            return set()
        members = universe_members(self.membership, t, execution_date)
        liquid = liquid_symbols(self.adj, t)
        raw = self.raw.loc[t]
        priced = set(raw.index[raw.ge(MIN_PRICE)])
        jumped = self.recent_jump.loc[t]
        sane = set(jumped.index[~jumped])
        return members & liquid & priced & sane

    def features_on(self, t, symbols=None, ranked=True):
        t = pd.Timestamp(t)
        syms = sorted(self.eligible(t) if symbols is None else symbols)
        X = pd.DataFrame({name: table.loc[t].reindex(syms) for name, table in self.wide.items()}, index=syms)

        known = self.events.loc[self.events.available_date.le(t)]
        latest = known.groupby("symbol").last().reindex(syms)
        stale = (t - latest.fiscal_date).dt.days.gt(STALE_FINANCIAL_DAYS) | latest.fiscal_date.isna()
        latest.loc[stale, FINANCIAL_FEATURES] = np.nan
        X[FINANCIAL_FEATURES] = latest[FINANCIAL_FEATURES]

        execution_date = self.next_session(t)
        quarter = execution_date.to_period("Q").start_time if execution_date is not None else None
        m = self.membership
        caps = m.loc[m.quarter_start.eq(quarter) & m.selection_date.le(t)].set_index("symbol").market_cap_proxy
        X["log_cap"] = np.log(caps.where(caps.gt(0)).reindex(syms))
        X.index.name = "symbol"
        return X.rank(pct=True) if ranked else X

    def labels_on(self, t, symbols):
        """Next-session entry to the first close on/after entry + 90 days (uses future prices)."""
        t = pd.Timestamp(t)
        syms = list(symbols)
        out = pd.DataFrame({"entry_date": pd.NaT, "label_end": pd.NaT, "target": np.nan},
                           index=pd.Index(syms, name="symbol"))
        i = self.calendar.get_loc(t)
        if i + 1 >= len(self.calendar):
            return out
        entry_date = self.calendar[i + 1]
        entry = self.adj.iloc[i + 1].reindex(syms)
        j = self.calendar.searchsorted(entry_date + pd.Timedelta(days=HOLDING_DAYS))
        block = self.adj.iloc[j:j + MAX_EXIT_DELAY + 1].reindex(columns=syms)
        if block.empty:
            return out
        has_exit = block.notna().any()
        first = block.notna().to_numpy().argmax(axis=0)
        exit_price = pd.Series(block.to_numpy()[first, np.arange(len(syms))], index=syms).where(has_exit)
        # No quote within 5 sessions of maturity: the target stays missing, never a guessed return.
        exit_date = pd.Series(block.index[first], index=syms).where(has_exit)
        out["entry_date"] = entry_date
        out["label_end"] = pd.to_datetime(exit_date).astype("datetime64[ns]")
        out["target"] = (exit_price / entry - 1).astype(float)
        return out

    def panel(self, dates):
        """Training/validation rows: ranked features, sector, target, and excess target."""
        frames = []
        for t in dates:
            X = self.features_on(t)
            if X.empty:
                continue
            y = self.labels_on(t, X.index)
            frame = X.join(y)
            frame["date"] = pd.Timestamp(t)
            frame["sector"] = self.sectors.reindex(frame.index).fillna("Unknown").to_numpy()
            frame["excess"] = frame.target - frame.target.median()
            frames.append(frame.reset_index())
        panel = pd.concat(frames, ignore_index=True)
        panel["excess_w"] = winsorize_by_date(panel, "excess")
        return panel


def winsorize_by_date(frame, column, q=TARGET_WINSOR):
    """Clip each date's values to that date's q and 1-q quantiles."""
    grouped = frame.groupby("date")[column]
    return frame[column].clip(grouped.transform("quantile", q), grouped.transform("quantile", 1 - q))


def month_end_sessions(calendar, start, end):
    cal = pd.DatetimeIndex(calendar)
    cal = cal[(cal >= pd.Timestamp(start)) & (cal <= pd.Timestamp(end))]
    s = pd.Series(cal, index=cal)
    return pd.DatetimeIndex(s.groupby(cal.to_period("M")).max().to_numpy())


# Model, walk-forward validation, and the model strategy

def make_model(alpha):
    # Features are percentile ranks, so a missing value is filled with the middle rank.
    # Nothing is learned in the imputer, so there's no preprocessing leakage.
    return Pipeline([("impute", SimpleImputer(strategy="constant", fill_value=0.5, keep_empty_features=True)),
                     ("ridge", Ridge(alpha=alpha))])


def training_rows(panel, fit_date, train_start=DEV_START):
    fit_date = pd.Timestamp(fit_date)
    rows = panel.loc[panel.date.ge(train_start) & panel.date.lt(fit_date)
                     & panel.label_end.lt(fit_date) & panel.excess.notna()]
    assert rows.label_end.max() < fit_date, "training target ends after the fit date"
    return rows


def fit(panel, fit_date, alpha, train_start=DEV_START):
    rows = training_rows(panel, fit_date, train_start)
    model = make_model(alpha).fit(rows[FEATURES], rows.excess_w)
    model.fit_date_, model.n_train_ = pd.Timestamp(fit_date), len(rows)
    model.train_label_end_ = rows.label_end.max()
    return model


def daily_scores(frame, predictions):
    """Mean rank IC and top-minus-bottom decile spread, computed date by date."""
    df = frame[["date", "excess_w"]].rename(columns={"excess_w": "excess"})
    df = df.assign(pred=np.asarray(predictions)).dropna()
    ics, spreads = [], []
    for _, g in df.groupby("date"):
        if len(g) < 20 or g.pred.nunique() < 2:
            continue
        ics.append(g.pred.corr(g.excess, method="spearman"))
        q = g.pred.rank(pct=True)
        spreads.append(g.excess[q > 0.9].mean() - g.excess[q <= 0.1].mean())
    if not ics:
        return pd.Series({"rank_IC": np.nan, "decile_spread": np.nan, "IC_hit_rate": np.nan, "dates": 0})
    return pd.Series({"rank_IC": np.mean(ics), "decile_spread": np.mean(spreads),
                      "IC_hit_rate": np.mean(np.array(ics) > 0), "dates": len(ics)})


def walk_forward(panel, calendar, years=VALIDATION_YEARS, alphas=RIDGE_ALPHAS):
    """Fit at the first session of each validation year; score that year's month-ends."""
    cal = pd.DatetimeIndex(calendar)
    rows = []
    for year in years:
        fit_date = cal[cal >= pd.Timestamp(f"{year}-01-01")][0]
        valid = panel.loc[panel.date.dt.year.eq(year) & panel.excess.notna()]
        for alpha in alphas:
            m = fit(panel, fit_date, alpha)
            assert m.train_label_end_ < valid.date.min()
            scores = daily_scores(valid, m.predict(valid[FEATURES]))
            rows.append({"year": year, "alpha": alpha, "fit_date": fit_date.date(),
                         "n_train": m.n_train_, **scores.to_dict()})
    return pd.DataFrame(rows)


def refit_dates(calendar, first_trade_day, last_trade_day):
    """The first signal date (session before the first trade), then the first session of each January."""
    cal = pd.DatetimeIndex(calendar)
    first_signal = cal[cal < pd.Timestamp(first_trade_day)][-1]
    januaries = pd.Series(cal, index=cal).groupby(cal.year).min()
    januaries = januaries[januaries.dt.month.eq(1) & januaries.gt(first_signal)
                          & januaries.le(pd.Timestamp(last_trade_day))]
    return sorted(set(januaries) | {first_signal})


class ModelStrategy(CohortStrategy):
    """Buy the top-ranked stocks by model score, within the name and sector caps."""

    def __init__(self, builder, models, calendar):
        super().__init__(builder.membership, builder.adj, calendar)
        self.builder = builder
        self.models = sorted(models, key=lambda m: m.fit_date_)
        self.picks = []

    def eligible(self, signal_date, execution_date):
        return self.builder.eligible(signal_date, execution_date)

    def model_for(self, signal_date):
        usable = [m for m in self.models if m.fit_date_ <= signal_date]
        if not usable:
            raise ValueError(f"No model fitted on or before {signal_date.date()}")
        return usable[-1]

    def pick(self, signal_date, eligible):
        if not eligible:
            return []
        X = self.builder.features_on(signal_date, symbols=eligible)
        model = self.model_for(signal_date)
        assert model.train_label_end_ < signal_date
        scores = pd.Series(model.predict(X[FEATURES]), index=X.index)
        ranked = scores.sort_values(ascending=False, kind="mergesort")
        self.picks.append((signal_date, model.fit_date_, len(eligible)))
        return list(ranked.index)

    def allocate(self, signal_date, budget, snapshot, eligible):
        ranked = self.pick(signal_date, eligible)
        return capped_allocation(ranked, budget, snapshot.holdings, self.builder.sectors, snapshot.nav)


# Accounting audit, charts, and markdown tables


FIG_DIR = HW_DIR / "figures"
STRATEGY_COLOR, BENCH_COLOR = "#2a78d6", "#eb6834"
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def audit(result, prices, calendar, plan_log, last_entry, end, min_price=None):
    h, f = result.history, result.fills
    px = prices.pivot(index="date", columns="symbol", values="adj_close").reindex(calendar)
    buys, sells = f[f.side.eq("BUY")], f[f.side.eq("SELL")]
    prev_session = {calendar[i]: calendar[i - 1] for i in range(1, len(calendar))}

    def actual_close(rows):
        return np.array([px.at[d, s] for d, s in zip(rows.date, rows.symbol)])

    spend = -buys.groupby("date").cash_change.sum()
    cash_at_signal = pd.Series(h.cash.reindex(spend.index.map(prev_session)).to_numpy(), index=spend.index)
    first_quote_after_due = [px[s].loc[due:].dropna().index[0] for s, due in zip(sells.symbol, sells.due_date)]
    window_counts = px.notna().rolling(LIQUIDITY_WINDOW, min_periods=LIQUIDITY_WINDOW).sum()
    liq_counts = [int(window_counts.at[sd, s]) for sd, s in zip(buys.signal_date, buys.symbol)]
    ramp_days = [d for d, why, _ in plan_log if why == "ramp-up"]
    reinvest_days = [d for d, why, _ in plan_log if why == "reinvest"]
    sale_cash = sells.groupby("date").cash_change.sum()
    proceeds_at_signal = [sale_cash.get(prev_session[d], 0.0) for d in reinvest_days]
    max_ratio = max((spend.get(d, 0.0) / p for d, p in zip(reinvest_days, proceeds_at_signal) if p > 0),
                    default=0.0)
    open_due = result.open_lots[result.open_lots.due_date.le(h.index[-1])]
    open_due_quoted = [px[s].loc[due:h.index[-1]].notna().any()
                       for s, due in zip(open_due.symbol, open_due.due_date)]
    held = (sells.date - sells.due_date).dt.days + HOLDING_DAYS
    exits = sells.set_index("lot_id").date
    ratio = px / px.ffill().shift(1)
    worst_moves = []
    for lot in buys.itertuples():
        window = ratio[lot.symbol].loc[lot.date:exits.get(lot.lot_id, h.index[-1])].iloc[1:]
        worst_moves.append(max(window.max(), 1 / window.min()) if window.notna().any() else 1.0)
    jumps = [m for m in worst_moves if m > MAX_DAILY_MOVE]

    checks = [
        ("NAV = cash + holdings, every day", f"max gap ${(h.nav - h.cash - h.holdings_value).abs().max():.2e}",
         np.allclose(h.nav, h.cash + h.holdings_value, rtol=0, atol=1e-6)),
        ("Cash never negative", f"min cash ${h.cash.min():,.2f}", h.cash.min() >= 0),
        ("Every BUY at that day's real adjusted close", f"{len(buys):,} buys",
         np.allclose(buys.price, actual_close(buys), rtol=0, atol=0)),
        ("Every BUY's signal is the previous session", "signal_date vs fill date",
         all(prev_session[d] == s for d, s in zip(buys.date, buys.signal_date))),
        ("No BUY after the last entry date", f"last buy {buys.date.max().date()}", buys.date.max() <= last_entry),
        ("Each day's buys fit in the previous close's cash", f"max spend/cash {(spend / cash_at_signal).max():.6f}",
         (spend <= cash_at_signal + CASH_TOLERANCE).all()),
        ("Every SELL at that day's real adjusted close", f"{len(sells):,} sells",
         np.allclose(sells.price, actual_close(sells), rtol=0, atol=0)),
        ("Every SELL on/after its 90-day due date", f"calendar days held: {held.min()}–{held.max()}",
         (sells.date >= sells.due_date).all()),
        ("Every SELL on the first quoted session on/after due", "no early or late exits",
         all(a == b for a, b in zip(sells.date, first_quote_after_due))),
        ("Each lot sold at most once; buys − sells = open lots",
         f"{len(buys)} − {len(sells)} = {len(result.open_lots)}",
         sells.lot_id.is_unique and len(buys) - len(sells) == len(result.open_lots)),
        ("Open lots past due only if no quote since due", f"{len(open_due)} open lots past due",
         not any(open_due_quoted)),
        ("Sum of fill cash flows = final cash − initial cash", f"${f.cash_change.sum():,.2f}",
         np.isclose(f.cash_change.sum(), h.cash.iloc[-1] - INITIAL_CAPITAL, rtol=0, atol=1e-4)),
        ("Initial capital + realized + unrealized PnL = final NAV", f"final NAV ${h.nav.iloc[-1]:,.2f}",
         np.isclose(result.summary.nav, h.nav.iloc[-1], rtol=0, atol=1e-6)),
        ("No date after the run end", f"last date {h.index[-1].date()}",
         h.index[-1] <= end and f.date.max() <= end),
        ("Liquidity: every BUY had ≥60 prices in the 63 sessions to its signal", f"min count {min(liq_counts)}",
         min(liq_counts) >= LIQUIDITY_MIN_QUOTES),
        ("Ramp-up: exactly 3 buy days, each the first session of a month",
         ", ".join(str(d.date()) for d in ramp_days),
         len(ramp_days) == RAMP_MONTHS
         and all(d.to_period("M") != prev_session[d].to_period("M") for d in ramp_days)),
        ("After ramp-up: every buy day follows a sale day", f"{len(reinvest_days)} reinvest days",
         all(p > 0 for p in proceeds_at_signal)),
        ("After ramp-up: spend ≤ the previous close's sale proceeds", f"max spend/proceeds {max_ratio:.6f}",
         max_ratio <= 1 + 1e-9),
        (f"No held position moved beyond x{MAX_DAILY_MOVE:.0f} or /{MAX_DAILY_MOVE:.0f} "
         "in one day (bad-print check)", f"largest one-day move while held: x{max(worst_moves):.2f}; "
         f"{len(jumps)} flagged", not jumps),
    ]
    if min_price is not None:
        raw = prices.pivot(index="date", columns="symbol", values="close").reindex(calendar)
        raw_at_signal = np.array([raw.at[sd, s] for sd, s in zip(buys.signal_date, buys.symbol)])
        checks.append((f"Every BUY had a raw close ≥ ${min_price:.0f} at its signal",
                       f"min ${raw_at_signal.min():,.2f}", raw_at_signal.min() >= min_price))
    return pd.DataFrame(checks, columns=["check", "detail", "passed"])


def audit_markdown(checks, result):
    d = result.diagnostics.event.value_counts()
    invested = (1 - result.history.cash / result.history.nav).iloc[1:].mean()
    return ["| Check | Detail | Passed |", "|---|---|:-:|",
            *[f"| {c} | {d_} | {'✅' if p else '❌'} |" for c, d_, p in checks.itertuples(index=False)], "",
            f"**{int(checks.passed.sum())} of {len(checks)} checks pass.** Diagnostics: "
            f"{int(d.get('stale_mark', 0))} stale marks, {int(d.get('frozen', 0))} freezes, "
            f"{int(d.get('rejected_no_quote', 0))} rejected orders, {int(d.get('exit_delayed', 0))} delayed exits. "
            f"Total fees ${result.summary.total_fees:,.2f}. Average share of NAV invested: {invested:.1%}.", ""]


def _fmt(row, value):
    if isinstance(value, pd.Timestamp):
        return str(value.date())
    if row in ("final_value", "start_value"):
        return f"${value:,.0f}"
    if row in ("Sharpe", "years"):
        return f"{value:.2f}"
    return f"{value:.2%}"


def performance_markdown(table, label="Strategy"):
    rows = ["start", "end", "final_value", "total_return", "CAGR", "max_drawdown", "annualized_volatility", "Sharpe"]
    return [f"| Metric | {label} | ^SP500TR |", "|---|---:|---:|",
            *[f"| {r} | {_fmt(r, table.at[r, 'Strategy'])} | {_fmt(r, table.at[r, '^SP500TR buy-and-hold'])} |"
              for r in rows], ""]


def _style(ax, title, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=11, color=INK, pad=10)
    ax.set_ylabel(ylabel, color=INK_2, fontsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.tick_params(colors=INK_2, labelsize=8, length=0)
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=0))


def plot_run(result, benchmark, prefix, label, note):
    FIG_DIR.mkdir(exist_ok=True)
    nav = result.history.nav
    bench = benchmark_nav(benchmark, nav.index)
    series = [(label, nav, STRATEGY_COLOR), ("^SP500TR buy-and-hold", bench, BENCH_COLOR)]
    period = f"{nav.index[1]:%b %Y} – {nav.index[-1]:%b %Y}"
    paths = {}

    fig, ax = plt.subplots(figsize=(9, 4.6), facecolor=SURFACE)
    for name, s, color in series:
        cum = s / s.iloc[0] - 1
        ax.plot(cum.index, cum, color=color, linewidth=1.8, label=name)
        ax.annotate(f"{cum.iloc[-1]:+.0%}", (cum.index[-1], cum.iloc[-1]), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=9, color=INK)
    ax.axhline(0, color=INK_2, linewidth=0.8)
    _style(ax, f"Cumulative return, {period}", "Cumulative return")
    ax.legend(frameon=False, fontsize=9, loc="upper left", labelcolor=INK)
    fig.text(0.01, 0.01, note, fontsize=8, color=INK_2)
    fig.tight_layout(rect=(0, 0.03, 0.97, 1))
    paths["cumulative"] = FIG_DIR / f"{prefix}_cumulative.png"
    fig.savefig(paths["cumulative"], dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 3.6), facecolor=SURFACE)
    for name, s, color in series:
        dd = drawdowns(s)
        ax.plot(dd.index, dd, color=color, linewidth=1.6, label=f"{name} (worst {dd.min():.1%}, {dd.idxmin():%b %Y})")
    _style(ax, "Drawdown from running peak", "Drawdown")
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.05))
    ax.set_ylim(top=0.01)
    ax.legend(frameon=False, fontsize=9, loc="lower left", labelcolor=INK)
    fig.text(0.01, 0.01, note, fontsize=8, color=INK_2)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    paths["drawdown"] = FIG_DIR / f"{prefix}_drawdown.png"
    fig.savefig(paths["drawdown"], dpi=150)
    plt.close(fig)

    h = result.history.iloc[1:]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 4.8), sharex=True, facecolor=SURFACE)
    a1.plot(h.index, 1 - h.cash / h.nav, color=STRATEGY_COLOR, linewidth=1.6)
    _style(a1, "Share of NAV invested", "Invested")
    a1.set_ylim(0, 1.05)
    a2.step(h.index, h.open_lots, where="post", color=STRATEGY_COLOR, linewidth=1.4)
    _style(a2, "Open lots", "Lots")
    a2.yaxis.set_major_formatter(mticker.ScalarFormatter())
    fig.text(0.01, 0.01, f"Average invested: {(1 - h.cash / h.nav).mean():.0%}. {note}", fontsize=8, color=INK_2)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    paths["invested"] = FIG_DIR / f"{prefix}_invested.png"
    fig.savefig(paths["invested"], dpi=150)
    plt.close(fig)
    return paths


# Development run (Jan 2010 - Mar 2016)

DEV_RESULTS_FILE = HW_DIR / "development_results.md"
FROZEN_FILE = HW_DIR / "frozen_config.json"
BACKTEST_START = pd.Timestamp("2012-01-01")


def load(end=BUFFER_END):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        prices, fundamentals, universe, benchmark, membership = load_data(end, database=FULL_DB)
    calendar = pd.DatetimeIndex(benchmark.date)
    builder = FeatureBuilder(prices, fundamentals, membership, universe, calendar, start=FEATURE_START)
    notes = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
    return prices, fundamentals, benchmark, calendar, builder, notes


def screened_out(builder, t):
    """Stocks that pass every screen except the price-sanity one."""
    raw = builder.raw.loc[t]
    others = (universe_members(builder.membership, t, builder.next_session(t))
              & liquid_symbols(builder.adj, t) & set(raw.index[raw.ge(MIN_PRICE)]))
    return len(others - builder.eligible(t))


def backtest(prices, calendar, builder, models, fee_bps, start, end, last_entry):
    first_trade = calendar[calendar >= pd.Timestamp(start)][0]
    strat = ModelStrategy(builder, models, calendar)
    result = Backtester(prices, calendar, fee_bps=fee_bps).run(strat, first_trade, end, last_entry=last_entry)
    return result, strat


def frozen_settings(alpha):
    return {
        "frozen_on": "2026-11-16", "ridge_alpha": alpha, "features": FEATURES,
        "train_start": str(DEV_START.date()), "training_dates": "month-end sessions",
        "target": "90-day return minus the date's median, capped at the date's 1st/99th percentiles",
        "target_winsor": TARGET_WINSOR, "max_daily_move": MAX_DAILY_MOVE,
        "refit": "first signal date, then the first session of each January",
        "top_n": TOP_N, "cohort_fraction": COHORT_FRACTION, "ramp_months": RAMP_MONTHS,
        "name_cap": NAME_CAP, "sector_cap": SECTOR_CAP, "min_price": MIN_PRICE,
        "liquidity_window": LIQUIDITY_WINDOW, "liquidity_min_quotes": LIQUIDITY_MIN_QUOTES,
        "fee_bps": FEE_BPS, "max_stale_sessions": MAX_STALE_SESSIONS,
    }


def develop():
    prices, fundamentals, benchmark, calendar, builder, load_notes = load()
    dates = month_end_sessions(calendar, DEV_START, DEV_END)
    panel = builder.panel(dates)

    wf = walk_forward(panel, calendar)
    by_alpha = wf.groupby("alpha")[["rank_IC", "decile_spread", "IC_hit_rate"]].mean()
    alpha = float(by_alpha.rank_IC.idxmax())

    first_trade = calendar[calendar >= BACKTEST_START][0]
    models = [fit(panel, d, alpha) for d in refit_dates(calendar, first_trade, DEV_END)]
    runs = {fee: backtest(prices, calendar, builder, models, fee, BACKTEST_START, BUFFER_END, DEV_END)
            for fee in (FEE_BPS, *FEE_SENSITIVITY_BPS)}
    result, strat = runs[FEE_BPS]
    checks = audit(result, prices, calendar, strat.log, DEV_END, BUFFER_END,
                          min_price=MIN_PRICE)
    coefs = pd.DataFrame({f"fit {m.fit_date_.date()}": m.named_steps["ridge"].coef_ for m in models},
                         index=FEATURES)
    fees = pd.DataFrame({f"{fee} bps": {**performance(res.history.nav)[["final_value", "CAGR",
                                                                                 "max_drawdown", "Sharpe"]],
                                         "fees_paid": res.summary.total_fees}
                         for fee, (res, _) in runs.items()}).T
    return SimpleNamespace(
        prices=prices, fundamentals=fundamentals, benchmark=benchmark, calendar=calendar, builder=builder,
        load_notes=load_notes, dates=dates, panel=panel, wf=wf, by_alpha=by_alpha, alpha=alpha, models=models,
        runs=runs, result=result, strategy=strat, checks=checks, fees=fees,
        table=compare(result.history.nav, benchmark),
        weights=coefs / coefs.abs().sum(),
        screened=[screened_out(builder, t) for t in dates])


def write_report(r):
    FROZEN_FILE.write_text(json.dumps(frozen_settings(r.alpha), indent=2) + "\n")
    paths = plot_run(r.result, r.benchmark, "dev", "Model strategy (10 bps)",
                            "Development backtest on the full database; alpha chosen on 2012-2015 validation.")
    per_date = r.panel.groupby("date").size()
    w = r.weights
    lines = [
        "# Development results (full database, Jan 2010 – Mar 2016 only)", "",
        "Generated by `python hw4.py dev`. No data after 2016-03-31 was loaded.", "",
        "## Data", "",
        f"- Prices: {len(r.prices):,} rows, {r.prices.symbol.nunique():,} symbols; "
        f"fundamentals: {len(r.fundamentals):,} rows.",
        *[f"- Loader warning: {n}" for n in r.load_notes],
        f"- Training/validation panel: {len(r.panel):,} rows on {len(r.dates)} month-end signal dates, "
        f"{per_date.min()}–{per_date.max()} eligible stocks per date (median {int(per_date.median())}); "
        f"{r.panel.excess.notna().mean():.1%} have a completed 90-day target.",
        f"- Price-sanity screen (a one-day move above x{MAX_DAILY_MOVE:.0f} or below "
        f"/{MAX_DAILY_MOVE:.0f} in the last 63 sessions) removed {sum(r.screened):,} stock-dates "
        f"across the {len(r.dates)} month-ends (max {max(r.screened)} on one date) that passed every other screen.",
        f"- Training targets are capped at each date's 1st/99th percentiles. Before capping, the largest "
        f"90-day return in the panel was {r.panel.target.max():+,.0%}; the 99.9th percentile was "
        f"{r.panel.target.quantile(0.999):+.0%}.",
        "- Feature coverage (non-missing): " + ", ".join(f"{c} {r.panel[c].notna().mean():.0%}" for c in FEATURES),
        "",
        "## Walk-forward validation", "",
        "The model is fitted at the first session of each year on month-end rows from 2010 whose targets had "
        "already ended, then scored on that year's month-ends. The alpha grid was fixed in `hw4.py` (RIDGE_ALPHAS) before "
        "any results were seen.", "",
        "| alpha | mean rank IC | mean decile spread (90d) | share of dates with IC > 0 |", "|---:|---:|---:|---:|",
        *[f"| {a:,.0f} | {x.rank_IC:.4f} | {x.decile_spread:.2%} | {x.IC_hit_rate:.0%} |"
          for a, x in r.by_alpha.iterrows()],
        "", f"**Chosen alpha: {r.alpha:,.0f}** (highest mean rank IC).", "",
        "| year | n_train | rank IC | decile spread |", "|---:|---:|---:|---:|",
        *[f"| {x.year} | {x.n_train:,} | {x.rank_IC:.4f} | {x.decile_spread:.2%} |"
          for x in r.wf[r.wf.alpha.eq(r.alpha)].itertuples()], "",
        "## Feature weights (coefficient / sum of |coefficients|; features are percentile ranks)", "",
        "With a large penalty only the relative sizes matter for ranking stocks. Positive means a higher "
        "rank predicts a higher 90-day market-relative return.", "",
        "| feature | " + " | ".join(w.columns) + " |", "|---|" + "---:|" * len(w.columns),
        *[f"| {f} | " + " | ".join(f"{v:+.1%}" for v in row) + " |" for f, row in w.iterrows()], "",
        "## Development backtest, Jan 2012 – Mar 2016 (last entry Dec 31, 2015)", "",
        "This period was also used to choose alpha, so it is not an out-of-sample result. After the last "
        "entry, the portfolio winds down to cash during the Jan–Mar 2016 buffer while the benchmark stays "
        "invested.", "",
        *performance_markdown(r.table, "Model strategy"),
        "### Transaction-cost sensitivity", "",
        "| cost per side | final value | CAGR | max drawdown | Sharpe | fees paid |", "|---|---:|---:|---:|---:|---:|",
        *[f"| {k} | ${x.final_value:,.0f} | {x.CAGR:.2%} | {x.max_drawdown:.2%} | {x.Sharpe:.2f} | "
          f"${x.fees_paid:,.0f} |" for k, x in r.fees.iterrows()], "",
        "### Accounting audit", "",
        *audit_markdown(r.checks, r.result),
        "## Plots", "",
        *[f"![{k}](figures/{p.name})\n" for k, p in paths.items()],
    ]
    DEV_RESULTS_FILE.write_text("\n".join(lines))
    return paths


# Final test (Apr 2016 - Sep 2026), only when RUN_FINAL_TEST is True

FINAL_RESULTS_FILE = HW_DIR / "final_results.md"


def check_frozen():
    if not RUN_FINAL_TEST:
        raise SystemExit("Final test not approved: RUN_FINAL_TEST is False. Nothing was loaded.")
    if not FROZEN_FILE.exists():
        raise SystemExit("frozen_config.json is missing; run `python hw4.py dev` first.")
    frozen = json.loads(FROZEN_FILE.read_text())
    current = frozen_settings(frozen["ridge_alpha"])
    changed = {k: (frozen.get(k), v) for k, v in current.items() if k != "frozen_on" and frozen.get(k) != v}
    if changed:
        raise SystemExit(f"Settings changed since they were frozen: {changed}")
    return frozen


def run_final():
    frozen = check_frozen()
    prices, fundamentals, benchmark, calendar, builder, load_notes = load(TEST_END)
    panel = builder.panel(month_end_sessions(calendar, DEV_START, TEST_END))
    first_trade = calendar[calendar >= TEST_START][0]
    models = [fit(panel, d, frozen["ridge_alpha"])
              for d in refit_dates(calendar, first_trade, TEST_END)]

    runs = {fee: backtest(prices, calendar, builder, models, fee, TEST_START, TEST_END,
                             TEST_END)
            for fee in (FEE_BPS, *FEE_SENSITIVITY_BPS)}
    result, strat = runs[FEE_BPS]
    checks = audit(result, prices, calendar, strat.log, TEST_END, TEST_END,
                          min_price=MIN_PRICE)
    table = compare(result.history.nav, benchmark)
    paths = plot_run(result, benchmark, "final", "Model strategy (10 bps)",
                            "Final test with settings frozen on 2026-11-16 in development.")
    at_zero = performance(result.history.nav_frozen_at_zero)
    diff = table.at["final_value", "Strategy minus benchmark"]

    lines = [
        "# Final test results, Apr 2016 – Sep 2026", "",
        f"Generated by `python hw4.py final` with the settings in `frozen_config.json` "
        f"(alpha {frozen['ridge_alpha']:,.0f}). Models refit at: "
        + ", ".join(str(m.fit_date_.date()) for m in models) + ".", "",
        *[f"- Loader warning: {n}" for n in load_notes], "",
        "## Strategy vs $1,000,000 in ^SP500TR at the first test close", "",
        *performance_markdown(table, "Model strategy"),
        f"**{'Beat' if diff > 0 else 'Did not beat'} the benchmark by ${abs(diff):,.0f} in final value "
        f"({table.at['CAGR', 'Strategy minus benchmark']:+.2%} CAGR).**", "",
        f"If every frozen (no-quote) position were worth $0, the final value would be ${at_zero.final_value:,.0f}.",
        "",
        "### Transaction-cost sensitivity", "",
        "| cost per side | final value | CAGR | max drawdown | Sharpe |", "|---|---:|---:|---:|---:|",
        *[f"| {fee} bps | ${p.final_value:,.0f} | {p.CAGR:.2%} | {p.max_drawdown:.2%} | {p.Sharpe:.2f} |"
          for fee, p in ((fee, performance(res.history.nav)) for fee, (res, _) in runs.items())], "",
        "### Accounting audit", "",
        *audit_markdown(checks, result),
        "## Plots", "",
        *[f"![{k}](figures/{p.name})\n" for k, p in paths.items()],
    ]
    FINAL_RESULTS_FILE.write_text("\n".join(lines))
    print(table.loc[["final_value", "CAGR", "max_drawdown", "Sharpe"]].to_string())
    print("audit:", "all checks pass" if checks.passed.all() else checks[~checks.passed].to_string())


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "dev"
    if command == "dev":
        r = develop()
        write_report(r)
        print(r.by_alpha.round(4).to_string())
        print("alpha", r.alpha)
        print(r.table.loc[["final_value", "CAGR", "max_drawdown", "Sharpe"]].to_string())
        print("audit:", "all checks pass" if r.checks.passed.all() else r.checks[~r.checks.passed].to_string())
        print(r.fees.to_string())
    elif command == "final":
        run_final()
    else:
        raise SystemExit("usage: hw4.py [dev|final]")
