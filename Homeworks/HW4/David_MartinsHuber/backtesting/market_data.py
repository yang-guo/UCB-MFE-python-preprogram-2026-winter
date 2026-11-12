"""Point-in-time market data for research and the backtesting engine.

The two database readers are reused verbatim from
``Lectures/modeling/course_data.py`` via the submission's ``data_loading.py``
(attribution: course data helpers, UCB MFE Python Pre-Program).  This module
adds only the panel construction needed by HW4:

* prices and volume as wide, calendar-aligned panels;
* financial statements as an availability-dated, forward-filled panel;
* a training panel with the schema the engine's refit path expects.

Timing rules
------------
A filing has no intraday timestamp, so it becomes usable at the close of the
following calendar day: ``available_date = filing_date + 1 day``.  The
financial panel is a backward as-of join from price dates to
``available_date``.  The left side is always the price calendar, so price
history is never dropped.  Values persist until the next filing (no staleness
cap); rows before the first filing are left missing, never back-filled.

Currency
--------
All remaining names are US-listed and prices are USD.  Tickers whose
``reported_currency`` is present and not ``USD`` are removed at load time, so
the surviving raw financial levels are treated as USD.  A missing
``reported_currency`` is treated as USD for these US listings.
"""

from dataclasses import dataclass, field
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from data_loading import load_data, load_membership


LOADER_FUNDAMENTAL_COLUMNS = (
    "revenue", "net_income", "operating_income", "total_assets", "total_debt",
    "cash_and_equivalents", "free_cash_flow", "operating_cash_flow",
)

FINANCIAL_RATIOS = (
    "net_margin", "operating_roa", "debt_to_assets", "cash_to_assets",
    "fcf_to_assets", "accruals_to_assets",
)

PRICE_FEATURES = (
    "momentum_21", "momentum_63", "momentum_252_21", "volatility_63", "drawdown_252",
)

AUDIT_COLUMNS = ("available_date", "fiscal_date", "filing_date", "filing_age_days")

MARKET_CAP_COLUMN = "market_cap_usd"


def _as_datetime(frame, columns):
    result = frame.copy()
    for column in columns:
        result[column] = pd.to_datetime(result[column]).astype("datetime64[ns]")
    return result


def _database_path(database):
    """Mirror the loader's default without changing its behavior."""
    if database is not None:
        return Path(database)
    return Path(__file__).resolve().parents[1] / "data/equities_class.duckdb"


def _with_extra_fundamentals(fundamentals, database, end, symbols):
    """Add the remaining numeric fundamental columns from the same table.

    ``load_data`` intentionally selects a compact set of columns.  Research
    needs the wider vendor schema (market cap, shares, EPS, ...), so this is a
    separate read-only query rather than a change to the verbatim loader.
    """
    path = _database_path(database)
    if not path.is_file():
        raise FileNotFoundError(f"Download the modeling database from bCourses to {path}")
    symbol_filter = " AND symbol IN (SELECT unnest(?))" if symbols is not None else ""
    params = [end, end] + ([list(symbols)] if symbols is not None else [])
    with duckdb.connect(str(path), read_only=True) as con:
        schema = con.execute("DESCRIBE fundamentals").fetchdf()
        numeric = set(schema.loc[schema.column_type.eq("DOUBLE"), "column_name"])
        extra = sorted(numeric - set(LOADER_FUNDAMENTAL_COLUMNS))
        if not extra:
            return fundamentals
        columns = ", ".join(extra)
        right = con.execute(
            f"""SELECT symbol,fiscal_date,period,filing_date,{columns}
                FROM fundamentals
                WHERE fiscal_date<=? AND (filing_date<=? OR filing_date IS NULL)
                {symbol_filter}
                ORDER BY symbol,fiscal_date""",
            params,
        ).fetchdf()
    right = _as_datetime(right, ["fiscal_date", "filing_date"])
    keys = ["symbol", "fiscal_date", "period", "filing_date"]
    left = fundamentals.drop_duplicates(keys, keep="last")
    right = right.drop_duplicates(keys, keep="last")
    merged = left.merge(right, on=keys, how="left", validate="one_to_one")
    if len(merged) != len(left):
        raise ValueError("Extra fundamental merge changed the row count")
    return merged


@dataclass(frozen=True)
class MarketView:
    """Detached data window at one signal close."""

    signal_date: pd.Timestamp
    execution_date: pd.Timestamp
    prices: pd.DataFrame
    volume: pd.DataFrame
    membership: pd.DataFrame
    financials: pd.DataFrame


@dataclass(frozen=True)
class MarketData:
    """Validated market data plus point-in-time panel builders."""

    prices: pd.DataFrame
    fundamentals: pd.DataFrame
    universe: pd.DataFrame
    benchmark: pd.DataFrame
    membership: pd.DataFrame
    calendar: pd.DatetimeIndex
    non_usd_symbols: tuple[str, ...] = ()
    _cache: dict = field(default_factory=dict, compare=False, repr=False)

    def __post_init__(self):
        self._validate()

    # ------------------------------------------------------------------ load
    @classmethod
    def load(cls, database=None, end="2015-12-31", symbols=None,
             extra_fundamental_columns=True):
        prices, fundamentals, universe, benchmark = load_data(database, end=end, symbols=symbols)
        membership = load_membership(database, end=end)
        if extra_fundamental_columns:
            fundamentals = _with_extra_fundamentals(fundamentals, database, end, symbols)
        return cls.from_frames(prices, fundamentals, universe, benchmark, membership,
                               drop_non_usd=True)

    @classmethod
    def from_frames(cls, prices, fundamentals, universe, benchmark, membership,
                    calendar=None, drop_non_usd=True, non_usd_symbols=()):
        prices = _as_datetime(prices.copy(), ["date"]).sort_values(["date", "symbol"])
        fundamentals = _as_datetime(fundamentals.copy(), ["fiscal_date", "filing_date"])
        benchmark = _as_datetime(benchmark.copy(), ["date"]).sort_values("date")
        membership = _as_datetime(membership.copy(), ["quarter_start", "selection_date"])
        universe = universe.copy()

        dropped = tuple(non_usd_symbols)
        if drop_non_usd and "reported_currency" in fundamentals:
            mask = fundamentals.reported_currency.notna() & fundamentals.reported_currency.ne("USD")
            dropped = tuple(sorted(set(dropped) | set(fundamentals.loc[mask, "symbol"])))
        if dropped:
            keep = ~prices.symbol.isin(dropped)
            prices = prices.loc[keep].reset_index(drop=True)
            fundamentals = fundamentals.loc[~fundamentals.symbol.isin(dropped)].reset_index(drop=True)
            universe = universe.loc[~universe.symbol.isin(dropped)].reset_index(drop=True)
            membership = membership.loc[~membership.symbol.isin(dropped)].reset_index(drop=True)

        if calendar is None:
            calendar = pd.DatetimeIndex(benchmark.date.drop_duplicates().sort_values())
        else:
            calendar = pd.DatetimeIndex(pd.to_datetime(calendar)).sort_values()
        return cls(prices.reset_index(drop=True), fundamentals.reset_index(drop=True),
                   universe.reset_index(drop=True), benchmark.reset_index(drop=True),
                   membership.reset_index(drop=True), calendar, dropped)

    # -------------------------------------------------------------- validate
    def _validate(self):
        price_columns = {"date", "symbol", "adj_close", "close", "volume"}
        missing = price_columns - set(self.prices.columns)
        if missing:
            raise ValueError(f"prices is missing columns: {sorted(missing)}")
        if self.prices.empty or self.prices[["date", "symbol"]].isna().any().any():
            raise ValueError("prices requires nonmissing keys and at least one row")
        if self.prices.duplicated(["date", "symbol"]).any():
            raise ValueError("prices requires unique (date, symbol) keys")
        if not self.prices.date.isin(self.calendar).all():
            raise ValueError("prices must use the supplied market calendar")
        if not (np.isfinite(self.prices.adj_close) & self.prices.adj_close.gt(0)).all():
            raise ValueError("observed adjusted prices must be finite and positive")
        if not (np.isfinite(self.prices.close) & self.prices.close.gt(0)).all():
            raise ValueError("observed close prices must be finite and positive")

        fundamental_columns = {"symbol", "fiscal_date", "filing_date", "period"}
        missing = fundamental_columns - set(self.fundamentals.columns)
        if missing:
            raise ValueError(f"fundamentals is missing columns: {sorted(missing)}")
        needed = set(LOADER_FUNDAMENTAL_COLUMNS)
        missing = needed - set(self.fundamentals.columns)
        if missing:
            raise ValueError(f"fundamentals is missing loader columns: {sorted(missing)}")

        membership_columns = {"symbol", "quarter_start", "selection_date"}
        missing = membership_columns - set(self.membership.columns)
        if missing:
            raise ValueError(f"membership is missing columns: {sorted(missing)}")
        if self.membership.duplicated(["symbol", "quarter_start"]).any():
            raise ValueError("membership requires unique (symbol, quarter_start) keys")
        if {"date", "adj_close"} - set(self.benchmark.columns):
            raise ValueError("benchmark requires date and adj_close")
        if self.benchmark.date.duplicated().any():
            raise ValueError("benchmark requires unique dates")
        if not (np.isfinite(self.benchmark.adj_close) & self.benchmark.adj_close.gt(0)).all():
            raise ValueError("benchmark prices must be finite and positive")
        if not self.calendar.is_monotonic_increasing or self.calendar.has_duplicates:
            raise ValueError("calendar must be sorted with unique sessions")

    # ---------------------------------------------------------------- panels
    def price_panel(self, field="adj_close", start=None, end=None):
        """Wide date-by-symbol panel for one price field."""
        if field not in {"adj_close", "close", "volume"}:
            raise ValueError("field must be adj_close, close, or volume")
        prices = self.prices
        calendar = self.calendar
        if start is not None:
            start = pd.Timestamp(start)
            prices = prices.loc[prices.date.ge(start)]
            calendar = calendar[calendar >= start]
        if end is not None:
            end = pd.Timestamp(end)
            prices = prices.loc[prices.date.le(end)]
            calendar = calendar[calendar <= end]
        return (prices.pivot(index="date", columns="symbol", values=field)
                .reindex(calendar).sort_index())

    def volume_panel(self, start=None, end=None):
        """Wide raw-share volume panel; zero-volume rows are preserved."""
        return self.price_panel("volume", start=start, end=end)

    def financial_events(self, extra_lag_days=0):
        """Availability-dated filings after the last-statement policy.

        Filings are never discarded merely because a later restatement exists:
        the original is what was known before the restatement's
        ``available_date``.  Instead, for each ``(symbol, available_date)`` the
        latest fiscal period wins, so an old period filed late cannot replace a
        newer known period.  Identical keys keep the latest filing.
        """
        key = ("events", extra_lag_days)
        if key in self._cache:
            return self._cache[key]
        f = self.fundamentals.copy()
        f = f.loc[f.period.eq("quarter") & f.filing_date.gt(f.fiscal_date)].copy()
        f["available_date"] = f.filing_date + pd.Timedelta(days=1 + extra_lag_days)
        f = f.sort_values(["symbol", "available_date", "fiscal_date", "filing_date"])
        f = f.drop_duplicates(["symbol", "available_date", "fiscal_date"], keep="last")
        # Keep only the newest fiscal period known so far for each symbol.
        latest_fiscal = f.groupby("symbol").fiscal_date.cummax()
        f = f.loc[f.fiscal_date.eq(latest_fiscal)]
        f = f.drop_duplicates(["symbol", "available_date"], keep="last")

        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            assets = f.total_assets.where(f.total_assets.gt(0))
            revenue = f.revenue.where(f.revenue.gt(0))
            f["net_margin"] = f.net_income / revenue
            f["operating_roa"] = f.operating_income / assets
            f["debt_to_assets"] = f.total_debt / assets
            f["cash_to_assets"] = f.cash_and_equivalents / assets
            f["fcf_to_assets"] = f.free_cash_flow / assets
            f["accruals_to_assets"] = (f.net_income - f.operating_cash_flow) / assets
        f[list(FINANCIAL_RATIOS)] = f[list(FINANCIAL_RATIOS)].replace([np.inf, -np.inf], np.nan)
        f = f.sort_values(["symbol", "available_date", "fiscal_date"]).reset_index(drop=True)
        self._cache[key] = f
        return f

    @property
    def available_financials(self):
        """Numeric financial columns a strategy may request."""
        events = self.financial_events()
        numeric = events.select_dtypes(include=[np.number]).columns.tolist()
        numeric += self.fundamentals.select_dtypes(include=[np.number]).columns.tolist()
        # Keep the documented loader/ratio schema requestable even when a
        # synthetic frame has no filings yet.
        for column in (*LOADER_FUNDAMENTAL_COLUMNS, *FINANCIAL_RATIOS):
            if column not in numeric:
                numeric.append(column)
        return tuple(dict.fromkeys([*numeric, MARKET_CAP_COLUMN]))

    def validate_fields(self, fields):
        if isinstance(fields, str):
            fields = (fields,)
        fields = tuple(fields or ())
        unknown = set(fields) - set(self.available_financials)
        if unknown:
            raise ValueError(f"Unknown financial fields: {sorted(unknown)}")
        return fields

    def _resolve_fields(self, fields):
        if fields is None:
            return self.available_financials
        return self.validate_fields(fields)

    def financial_panel(self, fields=None, extra_lag_days=0, max_age_days=None,
                        start=None, end=None, symbols=None, include_market_cap=True):
        """Long forward-filled financial panel aligned to observed price rows.

        The left side is the price history, so no price row is lost.  A value
        becomes visible on its ``available_date`` and persists until the next
        filing.  ``max_age_days=None`` keeps the last value indefinitely.
        """
        fields = self._resolve_fields(fields)
        events = self.financial_events(extra_lag_days)
        left = self.prices[["date", "symbol", "close"]].copy()
        if start is not None:
            left = left.loc[left.date.ge(pd.Timestamp(start))]
        if end is not None:
            left = left.loc[left.date.le(pd.Timestamp(end))]
        if symbols is not None:
            left = left.loc[left.symbol.isin(set(symbols))]
        left = left.sort_values(["date", "symbol"])
        right = events.sort_values(["available_date", "symbol"])
        left["symbol"] = left.symbol.astype(object)
        right["symbol"] = right.symbol.astype(object)
        right["available_date"] = pd.to_datetime(right.available_date).astype("datetime64[ns]")
        panel = pd.merge_asof(left, right, left_on="date", right_on="available_date",
                              by="symbol", direction="backward", allow_exact_matches=True)
        panel["filing_age_days"] = (panel.date - panel.available_date).dt.days
        for field in fields:
            if field not in panel:
                panel[field] = np.nan
        if include_market_cap and MARKET_CAP_COLUMN in fields:
            if "shares_outstanding" not in panel:
                panel[MARKET_CAP_COLUMN] = np.nan
            else:
                panel[MARKET_CAP_COLUMN] = panel.shares_outstanding * panel.close
        if max_age_days is not None:
            stale = panel.filing_age_days.gt(max_age_days)
            existing = [c for c in fields if c in panel.columns]
            panel.loc[stale, existing] = np.nan
        columns = list(dict.fromkeys([*AUDIT_COLUMNS, "date", "symbol", *fields]))
        columns = [c for c in columns if c in panel.columns]
        return panel[columns].sort_values(["date", "symbol"]).reset_index(drop=True)

    def _close_on(self, date):
        """Raw USD close for one session, indexed by symbol."""
        date = pd.Timestamp(date)
        key = ("close", date)
        if key not in self._cache:
            dates = self.prices.date.to_numpy()
            i = np.searchsorted(dates, np.datetime64(date), side="left")
            j = np.searchsorted(dates, np.datetime64(date), side="right")
            self._cache[key] = self.prices.iloc[i:j].set_index("symbol").close
        return self._cache[key]

    def _event_index(self, events, extra_lag_days=0):
        """Per-symbol available dates and event row positions."""
        key = ("event_index", extra_lag_days)
        if key not in self._cache:
            index = {}
            for symbol, group in events.groupby("symbol", sort=False):
                index[symbol] = (group.available_date.to_numpy(), group.index.to_numpy())
            self._cache[key] = index
        return self._cache[key]

    def financials_at(self, date, fields=None, symbols=None):
        """One forward-filled financial row per symbol at ``date``."""
        date = pd.Timestamp(date)
        fields = self._resolve_fields(fields)
        if not fields:
            return pd.DataFrame(index=pd.Index([], name="symbol"))
        close = self._close_on(date)
        if symbols is not None:
            close = close.loc[close.index.intersection(pd.Index(sorted(set(symbols))))]
        internal = list(fields)
        if MARKET_CAP_COLUMN in fields and "shares_outstanding" not in internal:
            internal.append("shares_outstanding")
        columns = list(dict.fromkeys([*internal, *AUDIT_COLUMNS]))
        if close.empty:
            return pd.DataFrame(columns=columns, index=pd.Index([], name="symbol"))

        events = self.financial_events()
        index = self._event_index(events)
        positions, chosen = [], []
        point = np.datetime64(date)
        for symbol in close.index:
            entry = index.get(symbol)
            if entry is None:
                continue
            dates, rows = entry
            i = np.searchsorted(dates, point, side="right") - 1
            if i >= 0:
                positions.append(rows[i])
                chosen.append(symbol)

        out = pd.DataFrame(np.nan, index=close.index, columns=columns)
        for column in ("available_date", "fiscal_date", "filing_date"):
            out[column] = pd.NaT
        if positions:
            selected = events.iloc[positions]
            for column in columns:
                if column in selected.columns:
                    out.loc[chosen, column] = selected[column].to_numpy()
        out.index.name = "symbol"
        for column in ("available_date", "fiscal_date", "filing_date"):
            out[column] = pd.to_datetime(out[column])
        out["filing_age_days"] = (date - out.available_date).dt.days
        if MARKET_CAP_COLUMN in fields:
            if "shares_outstanding" in out:
                out[MARKET_CAP_COLUMN] = out.shares_outstanding * close.reindex(out.index)
            else:
                out[MARKET_CAP_COLUMN] = np.nan
            if "shares_outstanding" not in fields:
                out = out.drop(columns=["shares_outstanding"])
        future = out.available_date.notna() & out.available_date.gt(date)
        if future.any():
            raise ValueError("As-of lookup produced future financial data")
        return out

    def membership_at(self, signal_date, execution_date):
        """Execution-quarter members known at the signal close."""
        quarter = pd.Timestamp(execution_date).to_period("Q").start_time
        return self.membership.loc[
            self.membership.quarter_start.eq(quarter)
            & self.membership.selection_date.le(pd.Timestamp(signal_date))].copy()

    def view(self, signal_date, execution_date, financial_fields=()):
        """Detached window for one decision; never contains future sessions."""
        signal = pd.Timestamp(signal_date)
        return MarketView(
            signal_date=signal,
            execution_date=pd.Timestamp(execution_date),
            prices=self.price_panel(end=signal),
            volume=self.volume_panel(end=signal),
            membership=self.membership_at(signal, execution_date),
            financials=self.financials_at(signal, fields=financial_fields),
        )

    # -------------------------------------------------------------- research
    def _price_features(self, prices):
        frames = []
        calendar = self.calendar
        for symbol, group in prices.groupby("symbol", sort=True):
            p = group.set_index("date").reindex(calendar)
            px = p.adj_close
            daily = px.pct_change(fill_method=None)
            p["momentum_21"] = px.pct_change(21, fill_method=None)
            p["momentum_63"] = px.pct_change(63, fill_method=None)
            p["momentum_252_21"] = px.shift(21) / px.shift(252) - 1
            p["volatility_63"] = daily.rolling(63, min_periods=63).std() * np.sqrt(252)
            p["drawdown_252"] = px / px.rolling(252, min_periods=252).max() - 1
            p["symbol"] = symbol
            p.index.name = "date"
            p["eligible"] = px.notna() & p.volatility_63.notna()
            frames.append(p.loc[px.notna()].reset_index())
        return (pd.concat(frames, ignore_index=True)
                .sort_values(["date", "symbol"]).reset_index(drop=True))

    def _apply_membership(self, panel):
        calendar = self.calendar
        positions = calendar.get_indexer(panel.date)
        next_dates = pd.Series(pd.NaT, index=panel.index, dtype="datetime64[ns]")
        valid = (positions >= 0) & (positions + 1 < len(calendar))
        next_dates.loc[valid] = calendar[positions[valid] + 1].to_numpy()
        panel = panel.copy()
        panel["membership_quarter"] = next_dates.dt.to_period("Q").dt.start_time
        right = self.membership.rename(columns={"quarter_start": "membership_quarter",
                                                "selection_date": "universe_selection_date"})
        panel = panel.merge(right, on=["symbol", "membership_quarter"], how="left",
                            validate="many_to_one")
        if "universe_rank" in panel:
            panel["is_universe_member"] = panel.universe_rank.notna()
        else:
            panel["is_universe_member"] = panel.membership_quarter.notna()
        panel["eligible"] = panel.eligible & panel.is_universe_member
        return panel

    def _make_labels(self, panel, prices, holding_days, max_exit_delay):
        calendar = self.calendar
        pieces = []
        price_groups = prices.groupby("symbol", sort=False)
        for symbol, group in panel.groupby("symbol", sort=True):
            g = group.copy()
            quotes = price_groups.get_group(symbol).set_index("date").adj_close.sort_index()
            quotes = quotes.loc[quotes.index.isin(calendar)]
            positions = calendar.get_indexer(g.date) + 1
            entry = pd.Series(pd.NaT, index=g.index, dtype="datetime64[ns]")
            valid_entry = (positions > 0) & (positions < len(calendar))
            entry.loc[valid_entry] = calendar[positions[valid_entry]].to_numpy()
            g["entry_date"] = entry
            g["entry_price"] = entry.map(quotes)
            g["scheduled_exit"] = entry + pd.Timedelta(days=holding_days)
            g["label_end"] = pd.NaT
            g["exit_price"] = np.nan
            valid = g.entry_price.notna() & g.scheduled_exit.le(calendar[-1])
            due = g.loc[valid, "scheduled_exit"]
            ix = quotes.index.searchsorted(due)
            has_quote = ix < len(quotes)
            rows = due.index[has_quote]
            exits = quotes.index[ix[has_quote]]
            delay = calendar.get_indexer(exits) - calendar.searchsorted(due.loc[rows])
            rows = rows[delay <= max_exit_delay]
            exits = exits[delay <= max_exit_delay]
            g.loc[rows, "label_end"] = exits.to_numpy()
            g.loc[rows, "exit_price"] = quotes.reindex(exits).to_numpy()
            g["target"] = g.exit_price / g.entry_price - 1
            pieces.append(g)
        return (pd.concat(pieces, ignore_index=True)
                .sort_values(["date", "symbol"]).reset_index(drop=True))

    def training_panel(self, feature_columns=None, holding_days=90, max_exit_delay=5,
                       start=None, end=None, symbols=None, include_sector=True):
        """Research panel with the engine's refit schema.

        Returns ``date, symbol, available_date, label_end, target`` plus price
        features, forward-filled financial columns, ``filing_age_days`` and
        optionally ``sector``.  The engine selects the declared
        ``feature_columns`` from this frame.
        """
        prices = self.prices
        if start is not None:
            prices = prices.loc[prices.date.ge(pd.Timestamp(start))]
        if end is not None:
            prices = prices.loc[prices.date.le(pd.Timestamp(end))]
        if symbols is not None:
            prices = prices.loc[prices.symbol.isin(set(symbols))]

        panel = self._price_features(prices)
        financials = self.financial_panel(start=start, end=end, symbols=symbols)
        panel = panel.merge(financials, on=["date", "symbol"], how="left")
        panel = self._apply_membership(panel)
        if include_sector:
            if "sector" in self.universe:
                panel = panel.merge(self.universe[["symbol", "sector"]], on="symbol", how="left")
                panel["sector"] = panel.sector.fillna("Unknown")
            else:
                panel["sector"] = "Unknown"
        panel = self._make_labels(panel, prices, holding_days, max_exit_delay)

        if feature_columns is not None:
            feature_columns = tuple(feature_columns)
            unknown = set(feature_columns) - set(panel.columns)
            if unknown:
                raise ValueError(f"Unknown training columns: {sorted(unknown)}")
            audit = {"date", "symbol", "available_date", "label_end", "target",
                     "entry_date", "entry_price", "scheduled_exit", "exit_price"}
            leaked = set(feature_columns) & audit
            if leaked:
                raise ValueError(f"Target/label columns cannot be features: {sorted(leaked)}")
        return panel

    @property
    def coverage(self):
        return pd.Series(dict(
            price_start=self.prices.date.min(),
            price_end=self.prices.date.max(),
            sessions=len(self.calendar),
            price_symbols=self.prices.symbol.nunique(),
            fundamental_symbols=self.fundamentals.symbol.nunique(),
            non_usd_dropped=", ".join(self.non_usd_symbols) or "none",
        ))
