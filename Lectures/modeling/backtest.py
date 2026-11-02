"""Class-based backtesting framework shared by the Lecture 8 notebooks.

Strategies decide what to trade, the backtester advances the market calendar,
and the order management system owns cash, lots, fees, and PnL. Adjusted-price
units model total-return exposure rather than historical broker share counts.
"""

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, replace

import numpy as np
import pandas as pd


class HeldPriceGapError(ValueError):
    """Stop when a holding has gone too long without an observed price."""

    def __init__(self, symbol, date, last_quote_date):
        self.symbol = symbol
        self.date = pd.Timestamp(date)
        self.last_quote_date = pd.Timestamp(last_quote_date)
        super().__init__(f"Unresolved held-price gap: {symbol} at {self.date.date()}")


@dataclass(frozen=True)
class Lot:
    """One purchase, with its own cost basis and scheduled maturity."""

    lot_id: int
    symbol: str
    units: float
    signal_date: pd.Timestamp
    entry_date: pd.Timestamp
    entry_price: float
    entry_fee: float
    due_date: pd.Timestamp


@dataclass(frozen=True)
class PortfolioSnapshot:
    """A detached view of the account that a strategy can use to plan orders."""

    cash: float
    nav: float
    holdings: pd.Series  # Aggregate market value by symbol.
    fee_rate: float


class Strategy(ABC):
    """The interface every trading strategy implements."""

    def __init__(self, holding_days=90):
        if not isinstance(holding_days, int) or holding_days < 1:
            raise ValueError("holding_days must be a positive integer")
        self.holding_days = holding_days

    @abstractmethod
    def generate_orders(self, signal_date, execution_date, portfolio, history):
        """Return fee-inclusive buy budgets indexed by symbol.

        history contains adjusted closes through signal_date, with dates as
        rows and symbols as columns. The portfolio is from that same close.
        execution_date is known, but its prices are not included.
        """

    def should_sell(self, lot, date):
        """Sell at the lot's scheduled calendar-day maturity."""
        return date >= lot.due_date


class MonthlyTopNStrategy(Strategy):
    """Share monthly scheduling and allocation across different signals."""

    def __init__(self, top_n=3, cohort_fraction=1 / 3, holding_days=90):
        super().__init__(holding_days)
        if not isinstance(top_n, int) or top_n < 1 or not 0 < cohort_fraction <= 1:
            raise ValueError("Invalid stock count or allocation fraction")
        self.top_n = top_n
        self.cohort_fraction = cohort_fraction

    @abstractmethod
    def score_stocks(self, signal_date, execution_date, history):
        """Return scores for eligible stocks using information known at the signal close."""

    def generate_orders(self, signal_date, execution_date, portfolio, history):
        if execution_date.to_period("M") == signal_date.to_period("M"):
            return pd.Series(dtype=float)
        scores = self.score_stocks(signal_date, execution_date, history)
        if (not isinstance(scores, pd.Series) or not scores.index.is_unique
                or scores.index.hasnans or not np.isfinite(scores).all()):
            raise ValueError("Stock scores must be finite with unique, nonmissing symbols")
        return self.allocate_budgets(scores, portfolio)

    def allocate_budgets(self, scores, portfolio):
        """Split the available cohort budget equally among the highest scores."""
        ranked = scores.rename_axis("symbol").rename("score").reset_index().sort_values(
            ["score", "symbol"], ascending=[False, True]).head(self.top_n)
        budget = min(portfolio.cash, portfolio.nav * self.cohort_fraction)
        per_name = budget / len(ranked) if len(ranked) else 0.
        return pd.Series(per_name, index=ranked.symbol, name="budget", dtype=float)


class MomentumStrategy(MonthlyTopNStrategy):
    """Calculate trailing returns from the history supplied by the backtester."""

    def __init__(self, membership, top_n=3, cohort_fraction=1 / 3,
                 holding_days=90, lookback_sessions=63):
        super().__init__(top_n, cohort_fraction, holding_days)
        if not isinstance(lookback_sessions, int) or lookback_sessions < 1:
            raise ValueError("lookback_sessions must be a positive integer")
        membership = membership[["quarter_start", "symbol", "selection_date"]].copy()
        for column in ["quarter_start", "selection_date"]:
            membership[column] = pd.to_datetime(membership[column])
        if membership.isna().any().any() or membership.duplicated(["quarter_start", "symbol"]).any():
            raise ValueError("Membership needs dates and one row per quarter and symbol")
        self.membership = membership
        self.lookback_sessions = lookback_sessions

    def score_stocks(self, signal_date, execution_date, history):
        window = history.tail(self.lookback_sessions + 1)
        if len(window) < self.lookback_sessions + 1:
            return pd.Series(dtype=float)
        quarter = execution_date.to_period("Q").start_time
        members = self.membership.loc[
            self.membership.quarter_start.eq(quarter)
            & self.membership.selection_date.le(signal_date), "symbol"]
        complete = window.notna().all()
        eligible = complete & window.columns.isin(members)
        window = window.loc[:, eligible]
        return window.iloc[-1] / window.iloc[0] - 1


class PredictedScoreStrategy(MonthlyTopNStrategy):
    """Use model predictions prepared with the research notebook's fit schedule."""

    def __init__(self, scores, top_n=3, cohort_fraction=1 / 3, holding_days=90):
        super().__init__(top_n, cohort_fraction, holding_days)
        scores = scores[["date", "symbol", "score"]].copy()
        scores["date"] = pd.to_datetime(scores.date)
        if scores[["date", "symbol"]].isna().any().any():
            raise ValueError("Scores need dates and symbols")
        if scores.duplicated(["date", "symbol"]).any() or not np.isfinite(scores.score).all():
            raise ValueError("Scores must be finite with one row per date and symbol")
        self.scores = {date: group.set_index("symbol").score for date, group in scores.groupby("date")}

    def score_stocks(self, signal_date, execution_date, history):
        return self.scores.get(signal_date, pd.Series(dtype=float)).copy()


class OrderManagementSystem:
    """Book fills, maintain independent lots, and calculate trade PnL."""

    FILL_COLUMNS = ["date", "signal_date", "lot_id", "symbol", "side", "price", "units",
                    "notional", "fee", "cash_change", "due_date", "realized_pnl"]

    def __init__(self, initial_cash=1_000_000, fee_bps=5):
        if not np.isfinite([initial_cash, fee_bps]).all() or initial_cash <= 0 or fee_bps < 0:
            raise ValueError("Capital must be positive and fees nonnegative")
        self.initial_cash = float(initial_cash)
        self.cash = float(initial_cash)
        self.fee_rate = fee_bps / 10_000
        self.lots = {}
        self.fills = []
        self.total_fees = 0.
        self.realized_pnl = 0.
        self._next_lot_id = 0

    def size_for_budget(self, budget, price):
        """Reserve the entry fee inside a dollar budget."""
        if not np.isfinite([budget, price]).all() or budget < 0 or price <= 0:
            raise ValueError("Budget must be nonnegative and price positive")
        return budget / (price * (1 + self.fee_rate))

    def buy(self, size, price, *, symbol, date, signal_date, holding_days=90):
        """Buy units at a supplied fill price and return the new lot ID."""
        if not np.isfinite([size, price]).all() or size <= 0 or price <= 0:
            raise ValueError("Size and price must be finite and positive")
        date, signal_date = pd.Timestamp(date), pd.Timestamp(signal_date)
        if pd.isna(date) or pd.isna(signal_date) or signal_date >= date:
            raise ValueError("A buy needs a signal date before its fill date")
        if not symbol or not isinstance(holding_days, int) or holding_days < 1:
            raise ValueError("A buy needs a symbol and a positive holding period")
        notional = size * price
        fee = notional * self.fee_rate
        if notional + fee > self.cash + 1e-8:
            raise ValueError("Insufficient cash for the purchase and its fee")
        lot = Lot(self._next_lot_id, symbol, size, signal_date, date, price, fee,
                  date + pd.Timedelta(days=holding_days))
        self.cash = max(0., self.cash - notional - fee)  # Remove floating-point dust.
        self.lots[lot.lot_id] = lot
        self._next_lot_id += 1
        self._record_fill(lot, "BUY", date, size, price, fee, -(notional + fee), 0.)
        return lot.lot_id

    def sell(self, size, price, *, lot_id, date):
        """Sell all or part of a lot, booking cash and PnL immediately."""
        if not np.isfinite([size, price]).all() or size <= 0 or price <= 0:
            raise ValueError("Size and price must be finite and positive")
        lot = self.lots[lot_id]
        date = pd.Timestamp(date)
        if pd.isna(date) or date < lot.entry_date:
            raise ValueError("A sale cannot precede its purchase")
        if size > lot.units:
            raise ValueError("Cannot sell more units than the lot owns")
        notional = size * price
        fee = notional * self.fee_rate
        entry_fee = lot.entry_fee * size / lot.units
        pnl = size * (price - lot.entry_price) - entry_fee - fee
        self.cash += notional - fee
        self.realized_pnl += pnl
        if size == lot.units:
            del self.lots[lot_id]
        else:
            self.lots[lot_id] = replace(lot, units=lot.units - size, entry_fee=lot.entry_fee - entry_fee)
        self._record_fill(lot, "SELL", date, size, price, fee, notional - fee, pnl)
        return pnl

    def _record_fill(self, lot, side, date, size, price, fee, cash_change, pnl):
        self.total_fees += fee
        self.fills.append({"date": date, "signal_date": lot.signal_date, "lot_id": lot.lot_id,
                           "symbol": lot.symbol, "side": side, "price": price, "units": size,
                           "notional": size * price, "fee": fee, "cash_change": cash_change,
                           "due_date": lot.due_date, "realized_pnl": pnl})

    def snapshot(self, marks):
        """Value all open lots and return cash, NAV, and exposure by symbol."""
        holdings = {}
        for lot in self.lots.values():
            price = marks[lot.symbol]
            if not np.isfinite(price) or price <= 0:
                raise ValueError(f"Invalid mark for {lot.symbol}")
            holdings[lot.symbol] = holdings.get(lot.symbol, 0.) + lot.units * price
        holdings = pd.Series(holdings, dtype=float)
        return PortfolioSnapshot(self.cash, self.cash + holdings.sum(), holdings, self.fee_rate)

    def pnl(self, marks):
        """Reconcile realized and unrealized PnL with the account's value."""
        portfolio = self.snapshot(marks)
        unrealized = sum(lot.units * (marks[lot.symbol] - lot.entry_price) - lot.entry_fee
                         for lot in self.lots.values())
        return pd.Series({"cash": self.cash, "holdings_value": portfolio.holdings.sum(),
                          "nav": portfolio.nav, "realized_pnl": self.realized_pnl,
                          "unrealized_pnl": unrealized, "total_pnl": portfolio.nav - self.initial_cash,
                          "total_fees": self.total_fees})


@dataclass(frozen=True)
class BacktestResult:
    """Tables to inspect, plot, or save after a run."""

    history: pd.DataFrame
    fills: pd.DataFrame
    open_lots: pd.DataFrame
    diagnostics: pd.DataFrame
    pnl: pd.Series


class Backtester:
    """Advance one session at a time and coordinate a strategy with its OMS."""

    def __init__(self, prices, benchmark, strategy, oms=None, max_stale_sessions=5):
        if not isinstance(strategy, Strategy):
            raise TypeError("strategy must inherit from Strategy")
        if not isinstance(max_stale_sessions, int) or max_stale_sessions < 0:
            raise ValueError("max_stale_sessions must be a nonnegative integer")
        self.strategy = strategy
        self.oms = oms if oms is not None else OrderManagementSystem()
        if self.oms.fills or self.oms.lots or self.oms.cash != self.oms.initial_cash:
            raise ValueError("Start each backtest with a fresh OMS")
        self.max_stale_sessions = max_stale_sessions
        prices = prices[["date", "symbol", "adj_close"]].copy()
        benchmark = benchmark[["date", "adj_close"]].copy()
        prices["date"] = pd.to_datetime(prices.date)
        benchmark["date"] = pd.to_datetime(benchmark.date)
        for frame, keys in [(prices, ["date", "symbol"]), (benchmark, ["date"])]:
            if frame[keys].isna().any().any() or frame.duplicated(keys).any():
                raise ValueError("Quotes need unique, nonmissing dates and symbols")
            if not (np.isfinite(frame.adj_close) & frame.adj_close.gt(0)).all():
                raise ValueError("Prices must be finite and positive; omit unavailable quotes")
        self.benchmark = benchmark.set_index("date").adj_close.sort_index()
        self.calendar = pd.DatetimeIndex(self.benchmark.index)
        if not prices.date.isin(self.calendar).all():
            raise ValueError("Price dates must belong to the benchmark calendar")
        self.quotes = prices.pivot(index="date", columns="symbol", values="adj_close").reindex(self.calendar)
        self._has_run = False

    def market_history(self, signal_date):
        """Copy prices through the signal close, keeping market-session gaps."""
        return self.quotes.loc[:signal_date].dropna(axis=1, how="all").copy()

    @staticmethod
    def _validate_orders(orders, cash):
        if not isinstance(orders, pd.Series) or not orders.index.is_unique or orders.index.hasnans:
            raise ValueError("Orders must be a Series of budgets with unique symbols")
        if not np.isfinite(orders).all() or orders.lt(0).any() or orders.sum() > cash + 1e-8:
            raise ValueError("Order budgets must be finite, nonnegative, and covered by signal-date cash")

    def _check_held_marks(self, day, day_index, observed, last_seen, diagnostics):
        stale = 0
        for symbol in sorted({lot.symbol for lot in self.oms.lots.values()}):
            if day_index - last_seen[symbol] > self.max_stale_sessions:
                raise HeldPriceGapError(symbol, day, self.calendar[last_seen[symbol]])
            if symbol not in observed:
                stale += 1
                diagnostics.append({"date": day, "symbol": symbol, "event": "stale_mark"})
        return stale

    def run_backtest(self, start, end):
        """Plan with yesterday's information, execute, then record today's value."""
        days = self.calendar[(self.calendar >= pd.Timestamp(start)) & (self.calendar <= pd.Timestamp(end))]
        if not len(days) or self.calendar.get_loc(days[0]) == 0:
            raise ValueError("Need test sessions and one preceding signal session")
        if self._has_run:
            raise ValueError("Create a new Backtester and OMS for each run")
        self._has_run = True
        last_marks, last_seen, diagnostics = {}, {}, []
        portfolio = self.oms.snapshot(last_marks)
        previous_day = self.calendar[self.calendar.get_loc(days[0]) - 1]
        history = [{"date": previous_day, "cash": portfolio.cash, "holdings_value": 0.,
                    "nav": portfolio.nav, "open_lots": 0, "fees": 0., "gross_traded": 0., "stale_marks": 0}]
        for day in days:
            day_index = self.calendar.get_loc(day)
            signal_date = self.calendar[day_index - 1]
            market_history = self.market_history(signal_date)
            orders = self.strategy.generate_orders(signal_date, day, portfolio, market_history)
            self._validate_orders(orders, portfolio.cash)
            observed = self.quotes.loc[day].dropna()
            for symbol, price in observed.items():
                last_marks[symbol], last_seen[symbol] = price, day_index
            stale = self._check_held_marks(day, day_index, observed, last_seen, diagnostics)
            fill_start = len(self.oms.fills)
            for lot in list(self.oms.lots.values()):
                if self.strategy.should_sell(lot, day) and lot.symbol in observed:
                    self.oms.sell(lot.units, observed[lot.symbol], lot_id=lot.lot_id, date=day)
            for symbol, budget in orders.items():
                if budget <= 1e-8:
                    continue
                if symbol not in observed:
                    diagnostics.append({"date": day, "symbol": symbol, "event": "rejected_missing_quote"})
                    continue
                price = observed[symbol]
                size = self.oms.size_for_budget(budget, price)
                self.oms.buy(size, price, symbol=symbol, date=day, signal_date=signal_date,
                             holding_days=self.strategy.holding_days)
            portfolio = self.oms.snapshot(last_marks)
            daily_fills = self.oms.fills[fill_start:]
            history.append({"date": day, "cash": portfolio.cash, "holdings_value": portfolio.holdings.sum(),
                            "nav": portfolio.nav, "open_lots": len(self.oms.lots),
                            "fees": sum(fill["fee"] for fill in daily_fills),
                            "gross_traded": sum(fill["notional"] for fill in daily_fills), "stale_marks": stale})
        history = pd.DataFrame(history).set_index("date")
        history["benchmark_nav"] = self.oms.initial_cash
        history.loc[days, "benchmark_nav"] = self.oms.initial_cash * self.benchmark.loc[days] / self.benchmark.loc[days[0]]
        return BacktestResult(
            history=history,
            fills=pd.DataFrame(self.oms.fills, columns=self.oms.FILL_COLUMNS),
            open_lots=pd.DataFrame([asdict(lot) for lot in self.oms.lots.values()], columns=Lot.__dataclass_fields__),
            diagnostics=pd.DataFrame(diagnostics, columns=["date", "symbol", "event"]),
            pnl=self.oms.pnl(last_marks),
        )


class BacktestAnalytics:
    """Report portfolio performance from daily NAV, including open positions."""

    def __init__(self, result):
        self.result = result

    @staticmethod
    def performance(nav, annual_rf=0.02):
        """Daily simple returns, sample volatility, and a calendar-time CAGR."""
        nav = nav.sort_index()
        if (not isinstance(nav.index, pd.DatetimeIndex) or nav.index.has_duplicates
                or nav.index.hasnans or len(nav) < 2
                or not np.isfinite(nav).all() or nav.le(0).any()):
            raise ValueError("Need at least two positive, finite NAVs on unique dates")
        if not np.isfinite(annual_rf) or annual_rf <= -1:
            raise ValueError("The annual risk-free rate must be finite and greater than -1")
        returns = nav.pct_change(fill_method=None).dropna()
        excess = returns - ((1 + annual_rf) ** (1 / 252) - 1)
        volatility = returns.std(ddof=1)
        years = (nav.index[-1] - nav.index[0]).total_seconds() / (365.25 * 24 * 60 * 60)
        return pd.Series({"final_value": nav.iloc[-1], "total_return": nav.iloc[-1] / nav.iloc[0] - 1,
                          "CAGR": (nav.iloc[-1] / nav.iloc[0]) ** (1 / years) - 1,
                          "max_drawdown": (nav / nav.cummax() - 1).min(),
                          "annualized_volatility": volatility * np.sqrt(252),
                          "Sharpe": np.sqrt(252) * excess.mean() / volatility if volatility > 1e-12 else np.nan})

    def summary(self, annual_rf=0.02):
        h = self.result.history
        return pd.DataFrame({"strategy": self.performance(h.nav, annual_rf),
                             "S&P 500 total return": self.performance(h.benchmark_nav, annual_rf)})

    def diagnostics(self):
        h = self.result.history.iloc[1:]  # The initial-capital row is not a trading session.
        return pd.Series({"average_cash_fraction": (h.cash / h.nav).mean(),
                          "fees": h.fees.sum(),
                          "gross_traded_over_average_nav": h.gross_traded.sum() / h.nav.mean(),
                          "open_lots_at_end": len(self.result.open_lots),
                          "stale_marks": int(h.stale_marks.sum())})


def capped_budgets(scores, holdings, sectors, cash, nav, top_n=20,
                   cohort_fraction=1 / 3, name_cap=.05, sector_cap=.25, fee_bps=5):
    """Allocate equal budgets, reducing purchases to fit name and sector limits."""
    if not scores.index.is_unique or not holdings.index.is_unique or not sectors.index.is_unique:
        raise ValueError("Aggregate duplicate tickers before allocating")
    if not np.isfinite([nav, cash, fee_bps]).all() or nav <= 0 or cash < 0 or not np.isclose(cash + holdings.sum(), nav):
        raise ValueError("Holdings and cash must reconcile to NAV")
    if not np.isfinite(scores).all() or not np.isfinite(holdings).all() or holdings.lt(0).any():
        raise ValueError("Scores/holdings must be finite; holdings nonnegative")
    if not isinstance(top_n, int) or top_n < 1 or fee_bps < 0 or not all(0 < cap <= 1 for cap in [cohort_fraction, name_cap, sector_cap]):
        raise ValueError("Invalid allocation setting")
    sectors = sectors.fillna("Unknown")
    if not set(scores.index).union(holdings.index).issubset(sectors.index):
        raise ValueError("Every scored/held ticker needs an explicit sector bucket")
    ranked = scores.rename_axis("symbol").rename("score").reset_index().sort_values(
        ["score", "symbol"], ascending=[False, True]).head(top_n)
    fee = fee_bps / 10_000
    budget = min(cash, nav * cohort_fraction)
    equal_budget = budget / len(ranked) if len(ranked) else 0.
    planned_names = holdings.to_dict()
    planned_sectors = holdings.groupby(sectors.reindex(holdings.index)).sum().to_dict()
    orders = {}
    for symbol in ranked.symbol:
        sector = sectors.loc[symbol]
        name_room = max(0., name_cap * nav - planned_names.get(symbol, 0.))
        sector_room = max(0., sector_cap * nav - planned_sectors.get(sector, 0.))
        inclusive = min(equal_budget, budget, name_room * (1 + fee), sector_room * (1 + fee))
        notional = inclusive / (1 + fee)
        if inclusive > 1e-8:
            orders[symbol] = inclusive
            budget -= inclusive
            planned_names[symbol] = planned_names.get(symbol, 0.) + notional
            planned_sectors[sector] = planned_sectors.get(sector, 0.) + notional
    return pd.Series(orders, dtype=float, name="budget")


class CappedMomentumStrategy(MomentumStrategy):
    """Add concentration limits without changing execution or accounting."""

    def __init__(self, membership, sectors, top_n=20, cohort_fraction=1 / 3,
                 holding_days=90, lookback_sessions=63, name_cap=.05, sector_cap=.25):
        super().__init__(membership, top_n, cohort_fraction, holding_days, lookback_sessions)
        if not all(0 < cap <= 1 for cap in [name_cap, sector_cap]) or not sectors.index.is_unique:
            raise ValueError("Invalid caps or duplicate sector symbols")
        self.sectors = sectors.copy()
        self.name_cap = name_cap
        self.sector_cap = sector_cap

    def allocate_budgets(self, scores, portfolio):
        return capped_budgets(scores, portfolio.holdings, self.sectors, portfolio.cash, portfolio.nav,
                              top_n=self.top_n, cohort_fraction=self.cohort_fraction,
                              name_cap=self.name_cap, sector_cap=self.sector_cap,
                              fee_bps=portfolio.fee_rate * 10_000)
