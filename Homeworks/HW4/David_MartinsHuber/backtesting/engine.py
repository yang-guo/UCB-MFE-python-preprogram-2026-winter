"""Sequential next-close simulation, with restricted refit inputs."""

from copy import deepcopy
from dataclasses import asdict, dataclass, field, fields
import warnings

import numpy as np
import pandas as pd

from .contracts import DecisionContext, FitData, Phase, Strategy, plain_amount
from .market_data import MarketData
from .oms import OMS, Lot, Order
from .risk import AUDIT_COLUMNS, RiskContext, RiskManager


class HeldPriceGapError(ValueError):
    pass


def _dates(frame, columns):
    result = frame.copy()
    for column in columns:
        result[column] = pd.to_datetime(result[column])
        dates = result[column]
        if dates.dt.tz is not None or not dates.dropna().eq(dates.dropna().dt.normalize()).all():
            raise ValueError("Use timezone-naive session dates at midnight")
    return result


@dataclass
class BacktestResult:
    history: pd.DataFrame
    orders: pd.DataFrame
    fills: pd.DataFrame
    open_lots: pd.DataFrame
    diagnostics: pd.DataFrame
    refits: pd.DataFrame
    phases: pd.DataFrame
    pnl: pd.Series
    coverage: pd.Series = field(default_factory=lambda: pd.Series(dtype=object))
    risk: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=AUDIT_COLUMNS))

    def performance(self, annual_rf=0.02):
        from .analytics import BacktestAnalytics

        return BacktestAnalytics(self, annual_rf=annual_rf).summary()


@dataclass(frozen=True)
class MarketPanel:
    """Validated, pre-pivoted market data shared by coordinated engines."""

    benchmark: pd.Series
    calendar: pd.DatetimeIndex
    prices: pd.DataFrame
    membership: pd.DataFrame
    fundamentals: pd.DataFrame | None
    volume: pd.DataFrame | None
    first_valid: pd.Series
    market_data: MarketData | None = None

    @classmethod
    def from_inputs(cls, prices=None, benchmark=None, membership=None,
                    fundamentals=None, market_data=None):
        if market_data is not None:
            if not isinstance(market_data, MarketData):
                raise TypeError("market_data must be a MarketData")
            if any(frame is not None for frame in (prices, benchmark, membership)):
                raise ValueError("Pass either market_data or explicit frames, not both")
            prices, benchmark, membership = (market_data.prices, market_data.benchmark,
                                             market_data.membership)
            volume = market_data.volume_panel()
        else:
            volume = None
        if prices is None or benchmark is None or membership is None:
            raise ValueError("Provide prices, benchmark, membership, or market_data")

        prices = _dates(prices[["date", "symbol", "adj_close"]], ["date"])
        benchmark = _dates(benchmark[["date", "adj_close"]], ["date"])
        for frame, keys in [(prices, ["date", "symbol"]), (benchmark, ["date"])]:
            if frame[keys].isna().any().any() or frame.duplicated(keys).any():
                raise ValueError("Quotes require unique, nonmissing keys")
            if not (np.isfinite(frame.adj_close) & frame.adj_close.gt(0)).all():
                raise ValueError("Omit missing quotes; observed prices must be finite and positive")
        benchmark_series = benchmark.set_index("date").adj_close.sort_index()
        calendar = benchmark_series.index
        if not prices.date.isin(calendar).all():
            raise ValueError("Stock quotes must use the benchmark calendar")
        wide = (prices.pivot(index="date", columns="symbol", values="adj_close")
                .reindex(calendar))
        notna = wide.notna()
        first_valid = notna.idxmax()
        first_valid = first_valid.where(notna.any(), pd.NaT)

        membership = _dates(membership[["symbol", "quarter_start", "selection_date"]],
                            ["quarter_start", "selection_date"])
        if (membership.isna().any().any()
                or membership.duplicated(["symbol", "quarter_start"]).any()):
            raise ValueError("Invalid membership keys or dates")
        symbols = []
        for value in membership.symbol:
            if type(value) is np.str_:
                symbols.append(str(value))
            elif type(value) is str and value:
                symbols.append(value)
            else:
                raise ValueError("Membership symbols must be exact nonempty strings")
        membership = membership.assign(symbol=symbols)
        fundamentals = (None if fundamentals is None else
                        _dates(fundamentals, ["fiscal_date", "filing_date"]))
        return cls(benchmark_series, calendar, wide, membership, fundamentals,
                   None if volume is None else volume.reindex(calendar),
                   first_valid, market_data)


class Backtester:
    """One strategy, one fresh account, one run. Prices never reach strategy ahead of time.

    Optional training_panel schema: date, symbol, available_date, label_end,
    target, and explicit feature_columns. Feature rows must be point-in-time
    correct at their own date; this cannot be inferred from values alone.
    """

    def __init__(self, prices=None, benchmark=None, membership=None, strategy=None, *,
                 market_data=None, panel=None, fundamentals=None,
                 training_panel=None, feature_columns=(), initial_cash=1_000_000,
                 fee_bps=5, max_stale_sessions=5, risk_manager=None,
                 delisting="error"):
        if not isinstance(strategy, Strategy):
            raise TypeError("strategy must implement Strategy")
        if not isinstance(max_stale_sessions, int) or max_stale_sessions < 0:
            raise ValueError("Invalid stale-price limit")
        if delisting != "error":
            raise ValueError("Only delisting='error' is supported: a stale mark is not an executable quote")
        if panel is None:
            panel = MarketPanel.from_inputs(
                prices, benchmark, membership, fundamentals=fundamentals,
                market_data=market_data)
        else:
            if not isinstance(panel, MarketPanel):
                raise TypeError("panel must be a MarketPanel")
            if (any(frame is not None for frame in (prices, benchmark, membership))
                    or market_data is not None or fundamentals is not None):
                raise ValueError("Pass either panel or data inputs, not both")
        self._benchmark = panel.benchmark
        self._calendar = panel.calendar
        self._prices = panel.prices
        self._membership = panel.membership
        self._fundamentals = panel.fundamentals
        self._volume = panel.volume
        self._first_valid = panel.first_valid
        self._market_data = panel.market_data
        self._required_financials = tuple(getattr(strategy, "required_financials", ()))
        if self._required_financials:
            if self._market_data is None:
                raise ValueError("Strategy requires financial columns; supply market_data")
            self._market_data.validate_fields(self._required_financials)
        self._features = list(feature_columns)
        self._training = None
        if training_panel is not None:
            audit = ["date", "symbol", "available_date", "label_end", "target"]
            if not self._features or len(set(self._features)) != len(self._features):
                raise ValueError("Provide unique feature_columns for fitting")
            if set(self._features) & set(audit + ["entry_price", "exit_price", "scheduled_exit", "entry_date"]):
                raise ValueError("Target and label-audit columns cannot be features")
            panel = _dates(training_panel[audit + self._features],
                           ["date", "available_date", "label_end"])
            if (panel[["date", "symbol"]].isna().any().any()
                    or panel.duplicated(["date", "symbol"]).any()
                    or (panel.available_date.notna() & panel.available_date.gt(panel.date)).any()
                    or (panel.label_end.notna() & panel.label_end.lt(panel.date)).any()):
                # A missing available_date means no filing is known yet; such a
                # row is excluded by the refit cutoff rather than rejected here.
                raise ValueError("Training rows need unique keys and point-in-time availability")
            self._training = panel
        self.strategy = strategy
        self._strategy_id = strategy.strategy_id
        self.oms = OMS(initial_cash, fee_bps)
        if risk_manager is not None and not isinstance(risk_manager, RiskManager):
            raise TypeError("risk_manager must be a RiskManager")
        self.risk_manager = deepcopy(risk_manager)
        self.max_stale_sessions = max_stale_sessions
        self.delisting = delisting
        self.phase = Phase.READY
        self.model_version = 0
        self.refits, self.phases = [], []
        self._started = False

    def _phase(self, phase, signal_date):
        self.phase = phase
        self.phases.append(dict(signal_date=signal_date, phase=phase.value))

    def _context(self, signal_date, execution_date, marks):
        members = self._membership.loc[
            self._membership.quarter_start.eq(execution_date.to_period("Q").start_time)
            & self._membership.selection_date.le(signal_date)].copy()
        if self._market_data is not None:
            members = self._market_data.membership_at(signal_date, execution_date)
        fundamentals = pd.DataFrame()
        if self._fundamentals is not None:
            f = self._fundamentals
            # Keep only trustworthy filing lags; data becomes usable the next day.
            fundamentals = f.loc[f.filing_date.gt(f.fiscal_date)
                                 & (f.filing_date + pd.Timedelta(days=1)).le(signal_date)].copy()
        position = self._calendar.get_loc(signal_date)
        # Copy-on-write makes a prefix slice a cheap read-only view; filtering
        # by first-valid date preserves the old all-NaN column behavior without
        # scanning and copying the full history every session.
        prices = self._prices.iloc[:position + 1]
        active = self._first_valid.le(signal_date)
        if not active.all():
            prices = prices.loc[:, active.to_numpy()]
        volume = (self._volume.iloc[:position + 1]
                  if self._volume is not None else pd.DataFrame())
        financials = (self._market_data.financials_at(
                          signal_date, fields=self._required_financials)
                      if self._market_data is not None and self._required_financials
                      else pd.DataFrame())
        return DecisionContext(signal_date, execution_date,
                               prices,
                               members, fundamentals, self.oms.snapshot(marks), self.model_version,
                               volume, financials)

    def _refit(self, context):
        self_fitting = bool(getattr(self.strategy, "self_fitting", False))
        if self._training is None and not self_fitting:
            raise ValueError("Refit requested without a training_panel")
        if self_fitting:
            # A self-fitting strategy gets its data from DecisionContext. An
            # unused training_panel must not change the run or its audit.
            rows, fit_data = None, None
        else:
            rows = self._training.loc[
                self._training.date.le(context.signal_date)
                & self._training.available_date.le(context.signal_date)
                & self._training.label_end.lt(context.signal_date)
                & self._training.target.notna()].copy()
            if rows.empty or not np.isfinite(rows.target).all():
                raise ValueError("Refit requires finite matured targets")
            fit_data = FitData(rows[self._features].copy(), rows.target.copy(),
                               rows[["date", "symbol", "available_date", "label_end"]].copy())
        self._phase(Phase.REFITTING, context.signal_date)
        audit = dict(
            signal_date=context.signal_date, model_version=self.model_version + 1,
            rows=0 if rows is None else len(rows),
            first_date=pd.NaT if rows is None else rows.date.min(),
            last_date=pd.NaT if rows is None else rows.date.max(),
            latest_label_end=pd.NaT if rows is None else rows.label_end.max(),
            latest_available=pd.NaT if rows is None else rows.available_date.max(),
            settings=deepcopy(dict(self.strategy.refit_settings())))
        try:
            # A failed fit cannot partially mutate the active model. Successful fits
            # replace engine.strategy; callers should use that object after a run.
            candidate = deepcopy(self.strategy)
            candidate.refit(fit_data, context)
            if candidate.strategy_id != self.strategy.strategy_id:
                raise ValueError("Refit cannot change strategy identity")
        except BaseException:
            self.refits.append({**audit, "status": "failed"})
            raise
        self.strategy = candidate
        self.model_version += 1
        self.refits.append({**audit, "status": "success"})

    def begin(self, start, end, *, allow_partial=False):
        """Initialize a run; prepare and advance expose a shared-clock barrier."""
        if self.phase != Phase.READY or getattr(self, "_started", False):
            raise ValueError("Create a new backtester for each run")
        start, end = pd.Timestamp(start), pd.Timestamp(end)
        if any(pd.isna(d) or d.tz is not None or d != d.normalize() for d in (start, end)) or start > end:
            raise ValueError("Use increasing, timezone-naive calendar dates")
        if self._calendar.empty:
            raise ValueError("Benchmark calendar is empty")
        clipped = start < self._calendar[0] or end > self._calendar[-1]
        if clipped and not allow_partial:
            raise ValueError("Requested interval exceeds benchmark data coverage; supply sufficient data "
                             "or explicitly set allow_partial=True")
        days = self._calendar[(self._calendar >= start) & (self._calendar <= end)]
        if not len(days) or self._calendar.get_loc(days[0]) == 0:
            raise ValueError("Need test sessions and a preceding signal session")
        previous = self._calendar[self._calendar.get_loc(days[0]) - 1]
        coverage = pd.Series(dict(requested_start=start, requested_end=end,
                                  effective_start=days[0], effective_end=days[-1],
                                  initial_capital_date=previous, range_clipped=bool(clipped)))
        if clipped:
            warnings.warn(f"Partial backtest: requested {start.date()}–{end.date()}, "
                          f"executing {days[0].date()}–{days[-1].date()}", stacklevel=2)
        # A warnings-as-errors policy must not leave a half-initialized run.
        self._started = True
        self._days, self._cursor, self._prepared = days, 0, False
        self._marks, self._last_seen, self._diagnostics = {}, {}, []
        self._coverage = coverage
        self._history = [dict(date=previous, cash=self.oms.cash, holdings_value=0.0,
                              nav=self.oms.cash, fees=0.0, stale_marks=0, open_lots=0)]
        self._history_cache = None
        self._history_cache_len = 0

    def _require_started(self):
        if not self._started:
            raise ValueError("Call begin() first to initialize the backtest")

    def _history_frame(self):
        if self._history_cache_len != len(self._history):
            self._history_cache = pd.DataFrame(self._history).set_index("date")
            self._history_cache_len = len(self._history)
        return self._history_cache

    @property
    def history_so_far(self):
        """Detached completed history only; pending orders never create returns."""
        self._require_started()
        # A shallow copy shares blocks under copy-on-write, so callers cannot
        # mutate the engine's cache but do not pay for a full deep copy.
        return self._history_frame().copy(deep=False)

    @property
    def next_session(self):
        self._require_started()
        if self._cursor >= len(self._days):
            return None
        return self._days[self._cursor]

    def prepare(self):
        """Refit and decide at the prior close without reading today's quotes."""
        self._require_started()
        if self.phase in (Phase.FAILED, Phase.COMPLETE) or self._prepared or self.next_session is None:
            raise ValueError("No decision can be prepared in the current state")
        day = self.next_session
        signal = self._calendar[self._calendar.get_loc(day) - 1]
        orders_before = len(self.oms.orders)
        try:
            context = self._context(signal, day, self._marks)
            # Freeze the allowed universe before any strategy code can run. The
            # strategy receives its own copies of the frames, but copying alone
            # does not stop it from editing the context it was handed.
            universe = frozenset(context.membership.symbol)
            if self.strategy.needs_refit(context):
                self._refit(context)
                context = self._context(signal, day, self._marks)
                universe = frozenset(context.membership.symbol)
            self._phase(Phase.DECIDING, signal)
            proposals = dict(self.strategy.propose_buys(context))
            if self.strategy.strategy_id != self._strategy_id:
                raise ValueError("Strategy identity must remain fixed throughout a run")
            # Require exact strings. A str subclass can compare equal to a real
            # ticker while remaining a distinct dict key; that would split one
            # name into several cap budgets and mislabel orders and fills.
            normalized = {}
            for symbol, budget in proposals.items():
                if type(symbol) is np.str_:
                    symbol = str(symbol)
                elif type(symbol) is not str:
                    raise ValueError("Buy proposals need valid symbols and finite nonnegative budgets")
                if not symbol:
                    raise ValueError("Buy proposals need valid symbols and finite nonnegative budgets")
                try:
                    amount = plain_amount(budget)
                except ValueError:
                    raise ValueError("Buy proposals need valid symbols and finite nonnegative budgets") from None
                if symbol in normalized:
                    raise ValueError("Duplicate buy proposal for one symbol")
                normalized[symbol] = amount
            proposals = normalized
            if set(proposals) - universe:
                raise ValueError("Purchase candidate is outside the known execution-quarter universe")
            self.raw_proposals = proposals.copy()
            if self.risk_manager is not None:
                # Risk review must see the engine's own account, not the planning
                # object the strategy was given. A strategy could otherwise
                # relabel or inflate holdings and have its tampered view trusted.
                decision = self.risk_manager.review(
                    RiskContext(signal, day, self.oms.snapshot(self._marks), self.history_so_far,
                                self.oms.fee_rate, self._strategy_id, self.model_version), proposals)
                proposals = dict(decision.approved_budgets)
            # Do not chase a vanishing cash residue into the denormal range.
            floor = 1e-12 * max(1.0, self.oms.available_cash)
            proposals = {symbol: budget for symbol, budget in proposals.items()
                         if budget > floor}
            tolerance = 1e-12 * max(1.0, self.oms.initial_cash)
            if sum(proposals.values()) > self.oms.available_cash + tolerance:
                raise ValueError("Proposals exceed signal-date available cash")
            self._buy_ids = [self.oms.submit_buy(self.strategy.strategy_id, symbol, budget, signal, day,
                                                model_version=self.model_version)
                             for symbol, budget in proposals.items() if budget > 0]
            self.pending_proposals = proposals.copy()
            self._prepared = True
            return context
        except BaseException:
            for order_id in [i for i in self.oms.orders if i >= orders_before]:
                self.oms.discard(order_id)
            self._phase(Phase.FAILED, signal)
            raise

    def advance(self):
        """Execute only after all coordinated accounts have prepared decisions."""
        self._require_started()
        if not self._prepared or self.phase != Phase.DECIDING:
            raise ValueError("Prepare orders before advancing the clock")
        day = self.next_session
        position = self._calendar.get_loc(day)
        signal = self._calendar[position - 1]
        buy_ids = self._buy_ids
        # Snapshot only what one session can change, so a failed execution is
        # atomic without copying the full order history every session.
        orders_before = len(self.oms.orders)
        order_status = {order_id: self.oms.orders[order_id].status for order_id in buy_ids}
        lots_before = dict(self.oms.lots)
        units_before = dict(self.oms._units)
        reserved_before = self.oms._reserved
        pending_buys_before = self.oms._pending_buys
        maturity_before = list(self.oms._maturity)
        fills_before, cash_before = len(self.oms.fills), self.oms.cash
        marks_before, last_seen_before = dict(self._marks), dict(self._last_seen)
        diagnostics_before = len(self._diagnostics)
        try:
            self._phase(Phase.EXECUTING, signal)
            quotes = self._prices.loc[day].dropna()
            for symbol, price in quotes.items():
                self._marks[symbol], self._last_seen[symbol] = float(price), position
            stale = 0
            first_fill = len(self.oms.fills)
            for symbol in self.oms.held_symbols:
                if position - self._last_seen[symbol] > self.max_stale_sessions:
                    raise HeldPriceGapError(f"{symbol}: unresolved held quote on {day.date()}")
                if symbol not in quotes:
                    stale += 1
                    self._diagnostics.append(dict(date=day, symbol=symbol, event="stale_mark"))
            for lot_id in self.oms.due_lots(day):
                lot = self.oms.lots[lot_id]
                if lot.symbol in quotes:
                    order_id = self.oms.submit_maturity_sale(lot.lot_id, day)
                    self.oms.fill(order_id, symbol=lot.symbol, quote_date=day,
                                  price=float(quotes[lot.symbol]), session=day)
                else:
                    self._diagnostics.append(dict(date=day, symbol=lot.symbol, event="maturity_exit_delayed"))
            for order_id in buy_ids:
                order = self.oms.orders[order_id]
                if self._calendar[self._calendar.get_loc(order.signal_date) + 1] != day:
                    raise ValueError("Purchase must execute exactly next session")
                if order.symbol not in quotes:
                    self.oms.reject(order_id)
                    self._diagnostics.append(dict(date=day, symbol=order.symbol, event="entry_rejected"))
                else:
                    self.oms.fill(order_id, symbol=order.symbol, quote_date=day,
                                  price=float(quotes[order.symbol]), session=day)
            portfolio = self.oms.snapshot(self._marks)
            self._history.append(dict(date=day, cash=portfolio.cash,
                                holdings_value=sum(portfolio.holdings.values()), nav=portfolio.nav,
                                fees=sum(f["fee"] for f in self.oms.fills[first_fill:]),
                                stale_marks=stale, open_lots=len(self.oms.lots)))
        except BaseException:
            for order_id in [i for i in self.oms.orders if i >= orders_before]:
                self.oms.discard(order_id)
            for order_id, status in order_status.items():
                if order_id in self.oms.orders:
                    self.oms.orders[order_id].status = status
            self.oms.lots = lots_before
            self.oms._units = units_before
            self.oms._reserved = reserved_before
            self.oms._pending_buys = pending_buys_before
            self.oms._maturity = maturity_before
            del self.oms.fills[fills_before:]
            self.oms.cash = cash_before
            self._marks = marks_before
            self._last_seen = last_seen_before
            del self._diagnostics[diagnostics_before:]
            self._phase(Phase.FAILED, signal)
            raise
        self._cursor += 1
        self._prepared = False

    def finish(self):
        self._require_started()
        if self.phase in (Phase.FAILED, Phase.COMPLETE) or self.next_session is not None or self._prepared:
            raise ValueError("Finish requires all sessions to have completed successfully")
        signal = self._calendar[self._calendar.get_loc(self._days[-1]) - 1]
        try:
            history = self._history_frame().copy(deep=True)
            history["benchmark_nav"] = self.oms.initial_cash
            with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
                benchmark_nav = (self._benchmark.loc[self._days] / self._benchmark.loc[self._days[0]]) * self.oms.initial_cash
            if not (np.isfinite(benchmark_nav).all() and benchmark_nav.gt(0).all()):
                raise ValueError("Benchmark normalization is not finite and positive")
            history.loc[self._days, "benchmark_nav"] = benchmark_nav
            fills = pd.DataFrame(self.oms.fills, columns=[
                "order_id", "lot_id", "strategy_id", "trigger", "model_version", "symbol", "side", "signal_date",
                "date", "due_date", "price", "units", "notional", "fee", "cash_change", "realized_pnl"])
            realized = fills.realized_pnl.sum()
            unrealized = sum(l.units * (self._marks[l.symbol] - l.entry_price) - l.entry_fee
                             for l in self.oms.lots.values())
            pnl = pd.Series(dict(realized=realized, unrealized=unrealized,
                                 total=history.nav.iloc[-1] - self.oms.initial_cash))
            if not np.isfinite(pnl.to_numpy()).all():
                raise ValueError("Final PnL arithmetic is not finite")
            result = BacktestResult(history,
                                    pd.DataFrame(self.oms.order_records(), columns=[f.name for f in fields(Order)]),
                                    fills, pd.DataFrame([asdict(l) for l in self.oms.lots.values()],
                                                        columns=[f.name for f in fields(Lot)]),
                                    pd.DataFrame(self._diagnostics, columns=["date", "symbol", "event"]),
                                    pd.DataFrame(self.refits, columns=[
                                        "signal_date", "model_version", "rows", "first_date",
                                        "last_date", "latest_label_end", "latest_available",
                                        "settings", "status"]),
                                    pd.DataFrame(self.phases), pnl, self._coverage.copy(),
                                    self.risk_manager.audit if self.risk_manager is not None else pd.DataFrame(columns=AUDIT_COLUMNS))
            # Publish COMPLETE only after the result exists. If the audit append
            # itself fails, the outer handler demotes the run to FAILED.
            self._phase(Phase.COMPLETE, signal)
            # Refresh the published audit so the terminal row is visible.
            result.phases = pd.DataFrame(self.phases)
        except BaseException:
            self._phase(Phase.FAILED, signal)
            raise
        return result


    def run(self, start, end, *, allow_partial=False):
        self.begin(start, end, allow_partial=allow_partial)
        while self.next_session is not None:
            self.prepare()
            self.advance()
        return self.finish()
