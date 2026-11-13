"""HW4's own backtester: forecasts -> orders -> lots -> daily value.

Design reference: Lecture 8 separates Strategy, execution, and analysis. This
implementation is written for the HW4 rules; it does not import the lecture
backtester. Prices are adjusted closes. Shares are fractional adjusted units.
Future return targets never enter this module's trading decisions.

The final strategy uses InverseVolatilityStrategy with 30 stocks, the "cash"
purchase schedule, the volatility screen, the no-quote rule and the
suspect-quote rule; run_final_test.py sets them explicitly. The defaults below
(5 stocks, equal weights, monthly schedule) are the first experiment's and are
kept so the earlier development notebooks and tests still run unchanged.
"""
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BacktestConfig:
    """Editable experiment assumptions. Rates use decimal units."""
    initial_cash: float = 1_000_000.0
    top_n: int = 5
    budget_fraction: float = 1 / 3
    holding_days: int = 90
    stock_limit: float = 0.30
    sector_limit: float = 0.50
    fee_rate: float = 0.001
    risk_free_rate: float = 0.02

    def __post_init__(self):
        if not np.isfinite(list(asdict(self).values())).all():
            raise ValueError("Configuration values must be finite.")
        if self.initial_cash <= 0 or not 0 <= self.fee_rate < 1:
            raise ValueError("Require positive initial cash and a fee rate in [0, 1).")
        if not isinstance(self.top_n, int) or self.top_n < 1:
            raise ValueError("top_n must be a positive integer.")
        if not isinstance(self.holding_days, int) or self.holding_days < 1:
            raise ValueError("holding_days must be a positive integer.")
        if not all(0 < x <= 1 for x in (self.budget_fraction, self.stock_limit, self.sector_limit)):
            raise ValueError("Budget fraction and exposure limits must be in (0, 1].")
        if self.risk_free_rate <= -1:
            raise ValueError("The annual risk-free rate must exceed -1.")


@dataclass
class PortfolioSnapshot:
    """Signal-close inputs only. holdings sums every open lot of each stock."""
    date: pd.Timestamp
    cash: float
    holdings: pd.Series

    @property
    def nav(self):
        return self.cash + self.holdings.sum()


@dataclass
class PurchasePlan:
    symbols: list = field(default_factory=list)
    cash_per_stock: float = 0.0  # Includes the entry fee.
    budget: float = 0.0
    reason: str = ""
    rejections: list = field(default_factory=list)
    cash_by_symbol: dict = field(default_factory=dict)

    def cash_for(self, symbol):
        """Use a weighted order when present; preserve the old equal allocator."""
        return self.cash_by_symbol[symbol] if self.cash_by_symbol else self.cash_per_stock

    @property
    def planned_cash(self):
        return sum(self.cash_by_symbol.values()) if self.cash_by_symbol else len(self.symbols) * self.cash_per_stock


class Strategy(ABC):
    """A common interface lets later strategies use the same execution engine."""
    @abstractmethod
    def generate_orders(self, snapshot, scores, sectors):
        """Return a PurchasePlan using information known at the signal close."""


class MonthlyRidgeStrategy(Strategy):
    """Choose five feasible ranked stocks with equal fee-inclusive budgets.

    Greedy ranking keeps the strongest feasible stock first. When a later,
    weaker stock breaches a limit, skip it and try the next ranked stock.
    Fixed equal allocations make this equivalent to replacing the weakest
    candidate in the violating sector. Skip the batch if it cannot reach five.
    """
    def __init__(self, config):
        self.config = config

    def generate_orders(self, snapshot, scores, sectors):
        cfg = self.config
        budget = min(snapshot.cash, snapshot.nav * cfg.budget_fraction)
        plan = PurchasePlan(budget=budget)
        if budget <= 1e-8:
            plan.reason = "no_cash"
            return plan
        cash_each = budget / cfg.top_n
        notional_each = cash_each / (1 + cfg.fee_rate)
        # Entry fees reduce NAV. Check limits against the full planned fee bill.
        nav_after_fees = snapshot.nav - budget * cfg.fee_rate / (1 + cfg.fee_rate)
        stock_values = snapshot.holdings.to_dict()
        sector_values = {}
        for symbol, value in stock_values.items():
            sector = sectors.get(symbol)
            if pd.isna(sector) or not str(sector).strip():
                raise ValueError(f"Missing sector for held stock {symbol}.")
            sector_values[sector] = sector_values.get(sector, 0.0) + value
        if (any(v > cfg.stock_limit * nav_after_fees + 1e-8 for v in stock_values.values())
                or any(v > cfg.sector_limit * nav_after_fees + 1e-8 for v in sector_values.values())):
            plan.reason = "existing_exposure_limit"
            return plan
        ranked = scores.sort_values(["prediction", "symbol"], ascending=[False, True])
        for row in ranked.itertuples(index=False):
            sector = sectors.get(row.symbol)
            reason = None
            if pd.isna(sector) or not str(sector).strip():
                reason = "missing_sector"
            elif stock_values.get(row.symbol, 0) + notional_each > cfg.stock_limit * nav_after_fees + 1e-8:
                reason = "stock_limit"
            elif sector_values.get(sector, 0) + notional_each > cfg.sector_limit * nav_after_fees + 1e-8:
                reason = "sector_limit"
            if reason:
                plan.rejections.append({"symbol": row.symbol, "prediction": row.prediction, "reason": reason})
                continue
            plan.symbols.append(row.symbol)
            stock_values[row.symbol] = stock_values.get(row.symbol, 0) + notional_each
            sector_values[sector] = sector_values.get(sector, 0) + notional_each
            if len(plan.symbols) == cfg.top_n:
                break
        if len(plan.symbols) < cfg.top_n:
            plan.symbols = []
            plan.reason = "fewer_than_required_stocks"
        else:
            plan.cash_per_stock = cash_each
            plan.reason = "planned"
        return plan


def inverse_volatility_weights(volatility):
    """Normalize inverse positive, finite volatility. Invalid values are errors.

    The strategy removes missing/zero/negative values before calling this.
    Scaling by the minimum volatility avoids overflowing 1 / tiny values.
    Weights equalize new-purchase weight * stand-alone volatility, not full
    portfolio risk contributions. Correlations and existing lots still matter.
    """
    if volatility.empty or volatility.index.has_duplicates:
        raise ValueError("Require unique stock identifiers and nonempty volatility.")
    if not (np.isfinite(volatility) & volatility.gt(0)).all():
        raise ValueError("Inverse-volatility inputs must be finite and positive.")
    inverse = volatility.min() / volatility
    return inverse / inverse.sum()


class InverseVolatilityStrategy(Strategy):
    """Top-N forecast selection with signal-close inverse-volatility budgets.

    Start with the highest N valid candidates. Recompute all weights after
    every replacement. If stock limits fail, remove the weakest forecast among
    those stocks. If sector limits fail, remove the weakest selected forecast
    in a violating sector. Try the next unused ranked candidate. Stop when a
    complete feasible group is found or the ranked list is exhausted. This is
    the specified deterministic replacement policy, not a global optimizer.

    Count all existing lots at signal-close prices. Deduct all planned entry
    fees from the NAV denominator. Keep no partial batch when N cannot be met.
    Missing sectors and nonpositive/nonfinite volatility are excluded. No
    future execution quote is used in selection or in the volatility input.
    """
    def __init__(self, config):
        self.config = config

    def generate_orders(self, snapshot, scores, sectors):
        cfg = self.config
        budget = min(snapshot.cash, snapshot.nav * cfg.budget_fraction)
        plan = PurchasePlan(budget=budget)
        if budget <= 1e-8:
            plan.reason = "no_cash"
            return plan
        if "volatility_63" not in scores:
            raise ValueError("Inverse allocation requires raw signal-date volatility_63.")
        ranked = scores.sort_values(["prediction", "symbol"], ascending=[False, True]).copy()
        ranked["sector"] = ranked.symbol.map(sectors)
        valid_vol = np.isfinite(ranked.volatility_63) & ranked.volatility_63.gt(0)
        valid_sector = ranked.sector.notna() & ranked.sector.fillna("").astype(str).str.strip().ne("")
        for row in ranked.loc[~(valid_vol & valid_sector)].itertuples():
            plan.rejections.append({"symbol": row.symbol, "prediction": row.prediction,
                                    "reason": "invalid_volatility" if not np.isfinite(row.volatility_63) or row.volatility_63 <= 0 else "missing_sector"})
        ranked = ranked.loc[valid_vol & valid_sector].reset_index(drop=True)
        holdings = snapshot.holdings.to_dict()
        held_sectors = {}
        for symbol, value in holdings.items():
            sector = sectors.get(symbol)
            if pd.isna(sector) or not str(sector).strip():
                raise ValueError(f"Missing sector for held stock {symbol}.")
            held_sectors[sector] = held_sectors.get(sector, 0.0) + value
        nav_after_fees = snapshot.nav - budget * cfg.fee_rate / (1 + cfg.fee_rate)
        if (any(v > cfg.stock_limit * nav_after_fees + 1e-8 for v in holdings.values())
                or any(v > cfg.sector_limit * nav_after_fees + 1e-8 for v in held_sectors.values())):
            plan.reason = "existing_exposure_limit"
            return plan
        selected = ranked.head(cfg.top_n).copy()
        next_index = cfg.top_n
        while len(selected) == cfg.top_n:
            weights = inverse_volatility_weights(selected.set_index("symbol").volatility_63)
            purchases = budget * weights / (1 + cfg.fee_rate)
            stock_bad = {s for s in weights.index
                         if holdings.get(s, 0) + purchases[s] > cfg.stock_limit * nav_after_fees + 1e-8}
            sector_values = held_sectors.copy()
            for row in selected.itertuples():
                sector_values[row.sector] = sector_values.get(row.sector, 0.0) + purchases[row.symbol]
            sector_bad = {s for s, value in sector_values.items() if value > cfg.sector_limit * nav_after_fees + 1e-8}
            if not stock_bad and not sector_bad:
                plan.symbols = selected.symbol.tolist()
                plan.cash_by_symbol = (budget * weights).to_dict()
                plan.reason = "planned"
                return plan
            violations = selected.symbol.isin(stock_bad) if stock_bad else selected.sector.isin(sector_bad)
            removed = selected.loc[violations].iloc[-1]  # Lowest forecast; symbol breaks ties.
            plan.rejections.append({"symbol": removed.symbol, "prediction": removed.prediction,
                                    "reason": "stock_limit" if removed.symbol in stock_bad else "sector_limit"})
            selected = selected.loc[selected.symbol.ne(removed.symbol)]
            if next_index >= len(ranked):
                break
            selected = pd.concat([selected, ranked.iloc[[next_index]]], ignore_index=True)
            next_index += 1
        plan.reason = "fewer_than_required_stocks"
        return plan


@dataclass
class Lot:
    """One purchase. A second purchase of the same stock has its own clock."""
    lot_id: int
    symbol: str
    signal_date: pd.Timestamp
    entry_date: pd.Timestamp
    due_date: pd.Timestamp
    units: float
    entry_price: float
    entry_fee: float
    cash_cost: float


@dataclass
class BacktestResult:
    daily: pd.DataFrame
    trades: pd.DataFrame
    orders: pd.DataFrame
    decisions: pd.DataFrame
    rejections: pd.DataFrame
    open_lots: pd.DataFrame
    valuation_flags: pd.DataFrame = field(default_factory=pd.DataFrame)
    suspect_quotes: pd.DataFrame = field(default_factory=pd.DataFrame)


class MissingHeldPriceError(ValueError):
    """Stop rather than invent a mark or an exit fill for a held stock."""


def remove_suspect_quotes(quotes, low=0.2, high=5.0):
    """Treat an implausible one-day price jump as a missing quote.

    quotes: sessions x symbols table of adjusted closes, NaN where unquoted.
    For each stock, walk forward in time and compare every quote with the last
    ACCEPTED quote. ratio = price / last accepted price. A ratio below `low`
    (a fall of more than 80% with the default) or above `high` (a rise of more
    than 400%) is suspect: the quote is removed and logged. Normal moves,
    including large ones such as +20% or -50%, are kept.

    Only past quotes decide, so the rule can be applied on the day itself.
    A removed quote is handled like any other missing quote: no entry, a held
    lot is marked at its last accepted close, and the no-quote exit rule applies.
    Limitation: a genuine jump beyond the bounds is also removed, and the stock
    stays unquoted until its price returns within the bounds.
    """
    if not 0 < low < 1 < high:
        raise ValueError("Use bounds with 0 < low < 1 < high.")
    clean = quotes.copy()
    filled = quotes.ffill()
    ratio = quotes / filled.shift(1)
    flagged = ratio.columns[((ratio < low) | (ratio > high)).any()]
    rows = []
    for symbol in flagged:  # Only stocks with at least one raw violation need the walk.
        values = quotes[symbol].to_numpy()
        column = clean.columns.get_loc(symbol)
        last = np.nan
        for k, price in enumerate(values):
            if np.isnan(price):
                continue
            if not np.isnan(last) and not low <= price / last <= high:
                rows.append({"date": quotes.index[k], "symbol": symbol, "price": price,
                             "last_accepted_price": last, "ratio": price / last})
                clean.iat[k, column] = np.nan
                continue
            last = price
    log = pd.DataFrame(rows, columns=["date", "symbol", "price", "last_accepted_price", "ratio"])
    return clean, log


class Backtester:
    """Run daily accounting. Build monthly orders after the signal close.

    1. Require observed quotes for all held stocks.
    2. Sell matured lots at today's close.
    3. Execute yesterday's preset dollar orders at today's close.
    4. Mark open lots and record NAV.
    5. On the month's first market session, plan next-session orders.

    Sales at the execution close do not increase an already planned budget.
    Missing entry quotes cancel only those orders; no same-close replacement.
    Exposure limits constrain signal-close plans. Price moves can cause actual
    weights to breach a limit. Record these breaches; do not force early sales.

    schedule="monthly" (default) decides on the month's first market session.
    schedule="cash" decides after the close of any session where cash is at
    least `min_cash_weight` of NAV and at least `min_sessions_between` market
    sessions have passed since the last purchase decision. It reinvests cash
    from matured lots without waiting for the next calendar month. The budget,
    next-session entry, and every accounting rule are identical in both.

    suspect_ratio=(low, high) removes implausible one-day price jumps before
    the run; see remove_suspect_quotes. None keeps every supplied quote.
    """
    def __init__(self, config, strategy=None, allow_stale_quotes=False, no_quote_exit_sessions=10,
                 no_quote_recovery=1.0, volatility_quantile=None, schedule="monthly",
                 min_cash_weight=0.05, min_sessions_between=21, suspect_ratio=None):
        if schedule not in ("monthly", "cash"):
            raise ValueError("schedule must be 'monthly' or 'cash'.")
        if not 0 < min_cash_weight <= 1:
            raise ValueError("min_cash_weight must be in (0, 1].")
        if not isinstance(min_sessions_between, int) or min_sessions_between < 1:
            raise ValueError("min_sessions_between must be a positive integer.")
        self.schedule = schedule
        self.min_cash_weight = min_cash_weight
        self.min_sessions_between = min_sessions_between
        self.suspect_ratio = suspect_ratio
        self.config = config
        self.strategy = strategy if strategy is not None else MonthlyRidgeStrategy(config)
        self.allow_stale_quotes = allow_stale_quotes
        if not isinstance(no_quote_exit_sessions, int) or no_quote_exit_sessions < 1:
            raise ValueError("The no-quote deadline requires a positive session count.")
        self.no_quote_exit_sessions = no_quote_exit_sessions
        if no_quote_recovery not in (0.0, 1.0):
            raise ValueError("Use last-close recovery (1) or the zero-recovery stress case (0).")
        self.no_quote_recovery = no_quote_recovery
        if volatility_quantile is not None and not 0 < volatility_quantile <= 1:
            raise ValueError("Use a volatility quantile in (0,1].")
        self.volatility_quantile = volatility_quantile

    def run(self, prices, calendar, forecasts, sectors, membership, start, end):
        cfg = self.config
        calendar = pd.DatetimeIndex(calendar)
        if calendar.empty or calendar.has_duplicates or calendar.hasnans or not calendar.is_monotonic_increasing:
            raise ValueError("Use an ordered calendar with unique valid sessions.")
        sessions = calendar[(calendar >= pd.Timestamp(start)) & (calendar <= pd.Timestamp(end))]
        if len(sessions) < 2:
            raise ValueError("The backtest needs at least two market sessions.")
        if prices.duplicated(["date", "symbol"]).any():
            raise ValueError("Duplicate stock/session prices.")
        if not prices.date.isin(calendar).all():
            raise ValueError("Prices must belong to the market calendar.")
        if not (np.isfinite(prices.adj_close) & prices.adj_close.gt(0)).all():
            raise ValueError("Use finite positive observed prices; omit missing quotes.")
        # Deliberately drop all target, future-entry, and future-exit fields.
        signal_columns = ["date", "symbol", "prediction"]
        if "volatility_63" in forecasts:
            signal_columns.append("volatility_63")  # Raw historical input, never a target.
        signals = forecasts[signal_columns].copy()
        if signals.duplicated(["date", "symbol"]).any() or not np.isfinite(signals.prediction).all():
            raise ValueError("Forecasts need unique keys and finite predictions.")
        if signals[["date", "symbol"]].isna().any().any() or not signals.date.isin(calendar).all():
            raise ValueError("Forecasts need valid stock identifiers and market dates.")
        if membership.duplicated(["symbol", "quarter_start"]).any():
            raise ValueError("Duplicate quarterly membership keys.")
        quotes = prices.pivot(index="date", columns="symbol", values="adj_close").reindex(sessions)
        suspect_log = pd.DataFrame(columns=["date", "symbol", "price", "last_accepted_price", "ratio"])
        if self.suspect_ratio is not None:
            quotes, suspect_log = remove_suspect_quotes(quotes, *self.suspect_ratio)
        first_sessions = set(calendar.to_series().groupby(calendar.to_period("M")).first())
        last_decision = None  # Session index of the previous purchase decision.
        cash = cfg.initial_cash
        lots = []
        pending = []
        daily, trades, orders, decisions, rejections = [], [], [], [], []
        valuation_flags = []
        last_quotes, last_quote_dates = {}, {}
        next_lot_id = 1
        for i, date in enumerate(sessions):
            today = quotes.loc[date]
            for symbol, price in today.dropna().items():
                last_quotes[symbol] = price
                last_quote_dates[symbol] = date
            missing = sorted({lot.symbol for lot in lots if pd.isna(today.get(lot.symbol, np.nan))})
            if missing and not self.allow_stale_quotes:
                raise MissingHeldPriceError(f"Missing held price on {date.date()}: {missing}. No fill or mark was invented.")
            # Record one mark per held lot/session, including a timeout exit day.
            for lot in lots:
                if lot.symbol in missing:
                    valuation_flags.append({"date": date, "lot_id": lot.lot_id, "symbol": lot.symbol,
                                            "price_date": last_quote_dates[lot.symbol],
                                            "mark_price": last_quotes[lot.symbol],
                                            "market_value": lot.units * last_quotes[lot.symbol]})
            day_fees = 0.0
            remaining = []
            for lot in lots:
                if date < lot.due_date:
                    remaining.append(lot)
                    continue
                observed = pd.notna(today.get(lot.symbol, np.nan))
                deadline_index = calendar.searchsorted(lot.due_date, side="right") + self.no_quote_exit_sessions - 1
                deadline = calendar[deadline_index] if deadline_index < len(calendar) else pd.NaT
                if not observed and (pd.isna(deadline) or date < deadline):
                    remaining.append(lot)
                    continue
                price = today[lot.symbol] if observed else last_quotes[lot.symbol] * self.no_quote_recovery
                value = lot.units * price
                fee = value * cfg.fee_rate
                cash += value - fee
                day_fees += fee
                trades.append({**asdict(lot), "date": date, "side": "SELL", "price": price,
                               "notional": value, "fee": fee, "cash_change": value - fee,
                               "net_pnl": value - fee - lot.cash_cost,
                               "no_quote_exit": not observed, "price_date": last_quote_dates[lot.symbol],
                               "exit_deadline": deadline})
            lots = remaining
            for order in pending:
                price = today.get(order["symbol"], np.nan)
                if pd.isna(price):
                    orders.append({**order, "status": "cancelled_missing_entry"})
                    continue
                cost = order["cash_budget"]
                if cost > cash + 1e-7:
                    raise AssertionError("Planned purchase exceeds available cash.")
                value = cost / (1 + cfg.fee_rate)
                fee = cost - value
                lot = Lot(next_lot_id, order["symbol"], order["signal_date"], date,
                          date + pd.Timedelta(days=cfg.holding_days), value / price, price, fee, cost)
                next_lot_id += 1
                lots.append(lot)
                cash -= cost
                day_fees += fee
                orders.append({**order, "status": "filled", "lot_id": lot.lot_id})
                trades.append({**asdict(lot), "date": date, "side": "BUY", "price": price,
                               "notional": value, "fee": fee, "cash_change": -cost, "net_pnl": np.nan,
                               "no_quote_exit": False, "price_date": date, "exit_deadline": pd.NaT})
            pending = []
            values = {}
            for lot in lots:
                values[lot.symbol] = values.get(lot.symbol, 0.0) + lot.units * last_quotes[lot.symbol]
            holdings = pd.Series(values, dtype=float)
            snapshot = PortfolioSnapshot(date, cash, holdings)
            sector_values = {}
            for symbol, value in values.items():
                sector = sectors.get(symbol)
                if pd.isna(sector) or not str(sector).strip():
                    raise ValueError(f"Missing sector for held stock {symbol}.")
                sector_values[sector] = sector_values.get(sector, 0) + value
            max_stock = max(values.values(), default=0.0) / snapshot.nav
            max_sector = max(sector_values.values(), default=0.0) / snapshot.nav
            daily.append({"date": date, "cash": cash, "holdings_value": holdings.sum(), "nav": snapshot.nav,
                          "open_lots": len(lots), "held_stocks": len(values), "fees": day_fees,
                          "max_stock_weight": max_stock, "max_sector_weight": max_sector,
                          "exposure_breach": max_stock > cfg.stock_limit + 1e-10 or max_sector > cfg.sector_limit + 1e-10})
            if cash < -1e-7 or snapshot.nav <= 0:
                raise AssertionError("Invalid cash or NAV.")
            if i + 1 >= len(sessions):
                continue
            if self.schedule == "monthly":
                if date not in first_sessions:
                    continue
            else:
                # Cash schedule: today's close decides, using today's cash and NAV only.
                if cash < self.min_cash_weight * snapshot.nav:
                    continue
                if last_decision is not None and i - last_decision < self.min_sessions_between:
                    continue
            last_decision = i
            execution_date = sessions[i + 1]
            candidates = signals.loc[signals.date.eq(date)].copy()
            quarter = execution_date.to_period("Q").start_time
            known_members = membership.loc[
                membership.quarter_start.eq(quarter) & membership.selection_date.le(date), "symbol"
            ]
            # Only signal-day information filters this list, never tomorrow's quote.
            candidates = candidates.loc[candidates.symbol.isin(known_members)
                                        & candidates.symbol.isin(today.dropna().index)]
            before_screen = len(candidates)
            volatility_threshold = np.nan
            if self.volatility_quantile is not None and len(candidates):
                from feature_engineering import signal_volatility_screen
                screened = signal_volatility_screen(candidates, self.volatility_quantile)
                volatility_threshold = screened.volatility_threshold.iloc[0]
                rejections.extend({"signal_date": date, "symbol": row.symbol,
                    "prediction": row.prediction, "reason": "above_signal_volatility_percentile"}
                    for row in screened.loc[~screened.passes_volatility_screen].itertuples())
                candidates = screened.loc[screened.passes_volatility_screen]
            plan = self.strategy.generate_orders(snapshot, candidates, sectors)
            decisions.append({"signal_date": date, "execution_date": execution_date,
                              "cash": cash, "nav": snapshot.nav, "budget_cap": plan.budget,
                              "candidate_count": len(candidates), "selected_count": len(plan.symbols),
                              "planned_cash": plan.planned_cash,
                              "candidates_before_screen": before_screen,
                              "volatility_threshold": volatility_threshold,
                              "symbols": ", ".join(plan.symbols), "reason": plan.reason})
            rejections.extend({"signal_date": date, **item} for item in plan.rejections)
            pending = [{"signal_date": date, "execution_date": execution_date, "symbol": symbol,
                        "cash_budget": plan.cash_for(symbol)} for symbol in plan.symbols]
        lot_columns = list(Lot.__dataclass_fields__)
        open_lots = pd.DataFrame([
            {**asdict(lot), "mark_date": sessions[-1], "mark_price": last_quotes[lot.symbol],
             "market_value": lot.units * last_quotes[lot.symbol],
             "unrealized_pnl": lot.units * last_quotes[lot.symbol] - lot.cash_cost}
            for lot in lots
        ], columns=[*lot_columns, "mark_date", "mark_price", "market_value", "unrealized_pnl"])
        # Keep headers even when a log has no events, so saved CSVs stay readable.
        trade_frame = pd.DataFrame(trades, columns=[*lot_columns, "date", "side", "price", "notional", "fee", "cash_change", "net_pnl", "no_quote_exit", "price_date", "exit_deadline"])
        order_frame = pd.DataFrame(orders, columns=["signal_date", "execution_date", "symbol", "cash_budget", "status", "lot_id"])
        decision_frame = pd.DataFrame(decisions, columns=["signal_date", "execution_date", "cash", "nav", "budget_cap",
                                                        "candidate_count", "selected_count", "planned_cash", "symbols", "reason",
                                                        "candidates_before_screen", "volatility_threshold"])
        rejection_frame = pd.DataFrame(rejections, columns=["signal_date", "symbol", "prediction", "reason"])
        flags = pd.DataFrame(valuation_flags, columns=["date", "lot_id", "symbol", "price_date", "mark_price", "market_value"])
        return BacktestResult(pd.DataFrame(daily), trade_frame, order_frame, decision_frame, rejection_frame,
                              open_lots, flags, suspect_log)


def compare_performance(daily, benchmark, config):
    """Return aligned NAV curves and metrics. Benchmark buys at first close.

    Benchmark: adjusted S&P 500 total-return index, no fees. Strategy: actual
    configured fees, no cash interest. Include all open positions at the end.
    CAGR uses calendar time. Sharpe uses 252 sessions and a 2% annual risk-free
    rate by default. Drawdowns use the running maximum of daily close NAV.
    """
    if benchmark.date.duplicated().any():
        raise ValueError("Select one benchmark with unique dates.")
    curves = daily.set_index("date")[["nav"]].rename(columns={"nav": "Strategy"})
    benchmark_price = benchmark.set_index("date").adj_close.reindex(curves.index)
    if not (np.isfinite(benchmark_price) & benchmark_price.gt(0)).all():
        raise ValueError("Benchmark needs a valid observed price on every evaluation session.")
    curves["S&P 500 TR"] = config.initial_cash * benchmark_price / benchmark_price.iloc[0]
    years = (curves.index[-1] - curves.index[0]).days / 365.25
    rows = []
    for name, nav in curves.items():
        returns = nav.pct_change(fill_method=None).dropna()
        excess = returns - ((1 + config.risk_free_rate) ** (1 / 252) - 1)
        std = returns.std(ddof=1)
        rows.append({"portfolio": name, "final_value": nav.iloc[-1],
                     "total_return": nav.iloc[-1] / config.initial_cash - 1,
                     "CAGR": (nav.iloc[-1] / config.initial_cash) ** (1 / years) - 1,
                     "max_drawdown": (nav / nav.cummax() - 1).min(),
                     "daily_Sharpe": excess.mean() / std * np.sqrt(252) if std > 1e-14 else np.nan})
    return curves, pd.DataFrame(rows).set_index("portfolio")


def detailed_performance(result, benchmark, config):
    """Add annualized daily volatility, average cash weight, and paid fees.

    Preserve compare_performance's original output for earlier notebooks.
    Benchmark holds the total-return index without cash or transaction fees.
    Cash weight is the mean of daily closing cash/NAV, including the first day.
    """
    curves, metrics = compare_performance(result.daily, benchmark, config)
    metrics["annualized_volatility"] = curves.pct_change(fill_method=None).std(ddof=1) * np.sqrt(252)
    metrics["average_cash_weight"] = [(result.daily.cash / result.daily.nav).mean(), 0.0]
    metrics["fees_paid"] = [result.daily.fees.sum(), 0.0]
    return curves, metrics


def portfolio_chart_data(curves):
    """Daily returns/drawdown, last-session monthly NAV, within-month vol.

    Monthly volatility = sample std of daily portfolio returns within each
    calendar month * sqrt(252). It is ANNUALIZED, not monthly-scaled. Require
    at least two observed returns. The first evaluation day's return is unknown
    and stays missing; never add an invented initial zero return.
    """
    daily_returns = curves.pct_change(fill_method=None)
    drawdown = curves / curves.cummax() - 1
    monthly_nav = curves.groupby(curves.index.to_period("M")).tail(1)
    monthly_volatility = daily_returns.groupby(daily_returns.index.to_period("M")).std(ddof=1) * np.sqrt(252)
    monthly_volatility.index = monthly_nav.index
    return {"daily_returns": daily_returns, "daily_drawdown": drawdown,
            "monthly_nav": monthly_nav, "monthly_volatility": monthly_volatility}


def reconcile_account(result, calendar, config):
    """Check cash flows, fees, positions, unique lot sales, and exact maturities."""
    daily, trades = result.daily, result.trades
    if trades.empty:
        flows = pd.Series(0.0, index=daily.date)
    else:
        flows = trades.groupby("date").cash_change.sum().reindex(daily.date, fill_value=0.0)
        np.testing.assert_allclose(trades.fee, trades.notional * config.fee_rate, atol=1e-7)
        buys = trades.loc[trades.side.eq("BUY")].set_index("lot_id")
        sells = trades.loc[trades.side.eq("SELL")]
        assert not buys.index.has_duplicates and not sells.lot_id.duplicated().any()
        for row in sells.itertuples():
            assert row.date >= calendar[calendar.searchsorted(row.due_date)]
            waiting = calendar[(calendar >= row.due_date) & (calendar < row.date)]
            if len(waiting):
                flags = result.valuation_flags
                marked = flags.loc[flags.lot_id.eq(row.lot_id), "date"]
                assert waiting.isin(marked).all(), "An available maturity quote was skipped."
            if getattr(row, "no_quote_exit", False):
                assert row.date == row.exit_deadline and row.price_date < row.date
            else:
                assert getattr(row, "price_date", row.date) == row.date
            assert row.signal_date < row.entry_date <= row.date
            assert row.due_date == row.entry_date + pd.Timedelta(days=config.holding_days)
            assert np.isclose(row.units, buys.loc[row.lot_id, "units"])
        expected_open = set(buys.index) - set(sells.lot_id)
        assert expected_open == set(result.open_lots.lot_id)
    np.testing.assert_allclose(daily.cash, config.initial_cash + flows.cumsum(), atol=1e-6)
    np.testing.assert_allclose(daily.nav, daily.cash + daily.holdings_value, atol=1e-6)
    assert daily.cash.min() >= -1e-7
    assert np.isclose(daily.nav.iloc[-1], daily.cash.iloc[-1] + result.open_lots.market_value.sum())
    if not result.decisions.empty:
        assert result.decisions.planned_cash.le(result.decisions.budget_cap + 1e-7).all()
        np.testing.assert_allclose(result.decisions.budget_cap,
                                   np.minimum(result.decisions.cash, result.decisions.nav * config.budget_fraction), atol=1e-7)
    return {"cash_and_nav": "PASS", "fees_and_lots": "PASS", "maturities": "PASS", "budget": "PASS"}


def closed_trade_errors(result, forecasts, feature_panel, benchmark, excess_target=False):
    """One row per closed lot. Compare predictions and outcomes in SAME units.

    A/B/baseline: forecast and realized_return are gross stock returns.
    C: both are stock return minus index return on the lot's exact dates.
    Also report gross stock/index returns so the distinction stays visible.
    volatility_63_at_signal is the raw allocation input. volatility_63_at_entry
    uses the entry session's close and is retrospective ANALYSIS ONLY. It never
    enters a purchase decision made at the previous signal close.
    """
    closed = result.trades.loc[result.trades.side.eq("SELL")].copy()
    columns = ["lot_id", "symbol", "signal_date", "entry_date", "exit_date", "forecast",
               "realized_return", "forecast_minus_realized", "absolute_error", "volatility_63_at_entry",
               "volatility_63_at_signal", "gross_stock_return", "index_return", "target_type"]
    if closed.empty:
        return pd.DataFrame(columns=columns)
    predictions = forecasts[["date", "symbol", "prediction"]].rename(columns={"date": "signal_date", "prediction": "forecast"})
    closed = closed.merge(predictions, on=["signal_date", "symbol"], how="left", validate="many_to_one")
    vol = feature_panel.set_index(["date", "symbol"]).volatility_63
    closed["volatility_63_at_signal"] = vol.reindex(pd.MultiIndex.from_frame(closed[["signal_date", "symbol"]])).to_numpy()
    closed["volatility_63_at_entry"] = vol.reindex(pd.MultiIndex.from_frame(closed[["entry_date", "symbol"]])).to_numpy()
    index = benchmark.set_index("date").adj_close
    closed["gross_stock_return"] = closed.price / closed.entry_price - 1
    closed["index_return"] = closed.date.map(index) / closed.entry_date.map(index) - 1
    closed["realized_return"] = closed.gross_stock_return - closed.index_return if excess_target else closed.gross_stock_return
    closed["forecast_minus_realized"] = closed.forecast - closed.realized_return
    closed["absolute_error"] = closed.forecast_minus_realized.abs()
    closed["target_type"] = "excess over index" if excess_target else "gross stock return"
    closed = closed.rename(columns={"date": "exit_date"})
    assert not closed.lot_id.duplicated().any()
    assert closed[["forecast", "realized_return"]].notna().all().all()
    return closed[columns].sort_values(["absolute_error", "lot_id"], ascending=[False, True]).reset_index(drop=True)
