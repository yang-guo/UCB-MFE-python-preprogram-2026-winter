"""HW4 data preparation, sourced accounting inputs and source-data diagnosis.

The three sections retain the existing causal feature pipeline, the separate
accounting-only overlay, and the post-run source-data audit. All scientific
functions and constants are unchanged. Source records are restored by the
submission entry points before an experiment or replay is run.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any
import warnings
import numpy as np
import pandas as pd
import json
import argparse
from datetime import date, datetime


# 1. Causal feature construction and the original course database
# ---------------------------------------------------------------
# Causal, monthly HW4 observations and unfilled daily adjusted-price history.
#
# The price formulas and conservative financial-event rules are adapted from
# Lectures/modeling/course_data.py in the course repository, commit
# 5212932d7d2e9de49785ee2c7e0395579b42d694.  The batching, monthly sampling,
# strict market-session exit labels, and audits below are this submission's
# implementation.  No lecture backtester is imported.

WARMUP_START = pd.Timestamp("2008-01-01")
TEST_START = pd.Timestamp("2016-04-01")
TEST_END = pd.Timestamp("2026-09-30")
EXPECTED_SIZE = 330_051_584
EXPECTED_SHA256 = "a028641d96c12e6f29d5dd34e3a6ebc059074dd38351648bf0d970b1dd4095b9"
PRICE_FEATURES = [
    "momentum_21", "momentum_63", "momentum_252_21",
    "volatility_63", "drawdown_252",
]
FINANCIAL_FEATURES = [
    "net_margin", "operating_roa", "debt_to_assets", "cash_to_assets",
    "fcf_to_assets", "accruals_to_assets",
]
FEATURE_COLUMNS = PRICE_FEATURES + FINANCIAL_FEATURES + ["filing_age_days"]
FINANCIAL_INPUTS = [
    "symbol", "fiscal_date", "period", "filing_date", "reported_currency",
    "revenue", "net_income", "operating_income", "total_assets",
    "total_debt", "cash_and_equivalents", "free_cash_flow",
    "operating_cash_flow",
]
SOURCE_LIMITATIONS = [
    "Quarterly membership is recorded-date reconstructed, not certified point-in-time data.",
    "Financial statements are the latest vendor vintages; publication-date joins cannot undo restatements.",
    "Country, exchange, sector, security type, and ticker mappings lack complete dated histories.",
    "Capitalization proxies may mix split adjustments, units, or share classes.",
    "No reliable delisting-proceeds table is available; final quotes do not establish exit cash.",
    "Positive finite prices can still contain economic errors or unverified corporate actions.",
]


@dataclass
class PreparedData:
    panel: pd.DataFrame
    prices: pd.DataFrame
    benchmark: pd.Series
    calendar: pd.DatetimeIndex
    audit: dict[str, Any]


def _dates(values) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(values)).astype("datetime64[ns]")


def monthly_schedule(calendar) -> pd.DataFrame:
    """Use the last close before each month's first entry session.

    The buffer's January--March 2016 entries are excluded.  A December 2009
    signal may support January 2010's entry; the model training code separately
    applies its stated development start.
    """
    calendar = _dates(calendar)
    if calendar.has_duplicates or not calendar.is_monotonic_increasing:
        raise ValueError("The market calendar must be unique and increasing.")
    first = ~calendar.to_period("M").duplicated()
    positions = np.flatnonzero(first)
    positions = positions[positions > 0]
    entries = calendar[positions]
    keep = ((entries >= "2010-01-01") & (entries <= "2015-12-31")) | (
        (entries >= TEST_START) & (entries <= TEST_END)
    )
    positions = positions[keep]
    return pd.DataFrame({
        "signal_date": calendar[positions - 1],
        "entry_date": calendar[positions],
    })


def make_price_features(quotes: pd.Series, calendar) -> pd.DataFrame:
    """Compute features on market sessions, without filling any price gap."""
    calendar = _dates(calendar)
    quotes = quotes.copy()
    quotes.index = _dates(quotes.index)
    if quotes.index.has_duplicates:
        raise ValueError("Duplicate price dates.")
    px = pd.to_numeric(quotes, errors="raise").reindex(calendar).astype(float)
    px = px.where(np.isfinite(px) & px.gt(0))
    returns = px.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    frame = pd.DataFrame(index=calendar)
    frame["momentum_21"] = px.pct_change(21, fill_method=None)
    frame["momentum_63"] = px.pct_change(63, fill_method=None)
    frame["momentum_252_21"] = px.shift(21) / px.shift(252) - 1
    frame["volatility_63"] = returns.rolling(63, min_periods=63).std() * np.sqrt(252)
    frame["drawdown_252"] = px / px.rolling(252, min_periods=252).max() - 1
    frame[PRICE_FEATURES] = frame[PRICE_FEATURES].replace([np.inf, -np.inf], np.nan)
    frame["eligible"] = px.notna() & frame.volatility_63.notna()
    frame.index.name = "signal_date"
    return frame


def make_financial_events(fundamentals: pd.DataFrame) -> pd.DataFrame:
    """Quarterly reports become usable on filing date plus one calendar day."""
    f = fundamentals.copy()
    for column in ("fiscal_date", "filing_date"):
        f[column] = pd.to_datetime(f[column]).astype("datetime64[ns]")
    f = f.loc[f.period.eq("quarter") & f.filing_date.gt(f.fiscal_date)].copy()
    f["available_date"] = f.filing_date + pd.Timedelta(days=1)
    assets = pd.to_numeric(f.total_assets, errors="raise").where(f.total_assets.gt(0))
    revenue = pd.to_numeric(f.revenue, errors="raise").where(f.revenue.gt(0))
    f["net_margin"] = f.net_income / revenue
    f["operating_roa"] = f.operating_income / assets
    f["debt_to_assets"] = f.total_debt / assets
    f["cash_to_assets"] = f.cash_and_equivalents / assets
    f["fcf_to_assets"] = f.free_cash_flow / assets
    f["accruals_to_assets"] = (f.net_income - f.operating_cash_flow) / assets
    f[FINANCIAL_FEATURES] = f[FINANCIAL_FEATURES].replace([np.inf, -np.inf], np.nan)
    f = f.sort_values(["symbol", "available_date", "fiscal_date"])
    latest_known_period = f.groupby("symbol", sort=False).fiscal_date.cummax()
    f = f.loc[f.fiscal_date.eq(latest_known_period)].drop_duplicates(
        ["symbol", "available_date"], keep="last"
    )
    return f[["symbol", "fiscal_date", "filing_date", "available_date"] + FINANCIAL_FEATURES]


def join_financial_events(signals: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Join each symbol backward; never backfill from a later publication."""
    pieces = []
    for symbol, group in signals.groupby("symbol", sort=False):
        right = events.loc[events.symbol.eq(symbol)].drop(columns="symbol")
        joined = pd.merge_asof(
            group.sort_values("signal_date"), right.sort_values("available_date"),
            left_on="signal_date", right_on="available_date", direction="backward",
        )
        joined["filing_age_days"] = (joined.signal_date - joined.available_date).dt.days
        stale = (joined.signal_date - joined.fiscal_date).dt.days.gt(550)
        joined.loc[stale, FINANCIAL_FEATURES] = np.nan
        if not (joined.available_date.isna() | joined.available_date.le(joined.signal_date)).all():
            raise ValueError("A financial feature uses an unavailable filing.")
        pieces.append(joined)
    return pd.concat(pieces, ignore_index=True)


def apply_dated_membership(signals: pd.DataFrame, membership: pd.DataFrame) -> pd.DataFrame:
    """Membership belongs to the entry quarter; it does not remove history."""
    m = membership.copy()
    for column in ("quarter_start", "selection_date"):
        m[column] = pd.to_datetime(m[column]).astype("datetime64[ns]")
    if m.duplicated(["symbol", "quarter_start"]).any():
        raise ValueError("Duplicate universe-membership key.")
    frame = signals.copy()
    frame["quarter_start"] = frame.entry_date.dt.to_period("Q").dt.start_time
    frame = frame.merge(
        m[["symbol", "quarter_start", "selection_date", "universe_rank"]],
        on=["symbol", "quarter_start"], how="left", validate="many_to_one",
    )
    member = frame.universe_rank.notna()
    if not frame.loc[member, "selection_date"].le(frame.loc[member, "signal_date"]).all():
        raise ValueError("Quarterly membership was not available at the signal close.")
    frame["is_universe_member"] = member
    frame["eligible"] = frame.eligible & member
    return frame


def make_holding_labels(signals: pd.DataFrame, quotes: pd.Series, calendar) -> pd.DataFrame:
    """Label the exact 90-calendar-day trade, not the next available stock quote.

    ``scheduled_exit`` is entry plus 90 calendar days. ``exit_session`` is the
    first market session at or after that date.  A missing entry or exit quote
    leaves target and label_end unknown. Eligibility never depends on targets.
    """
    calendar = _dates(calendar)
    frame = signals.copy()
    frame["signal_date"] = pd.to_datetime(frame.signal_date).astype("datetime64[ns]")
    frame["entry_date"] = pd.to_datetime(frame.entry_date).astype("datetime64[ns]")
    positions = calendar.get_indexer(frame.signal_date)
    valid_signal = (positions >= 0) & (positions + 1 < len(calendar))
    if not valid_signal.all():
        raise ValueError("Each signal requires a next market session.")
    if not np.array_equal(calendar[positions + 1].to_numpy(), frame.entry_date.to_numpy()):
        raise ValueError("Entry must be the next market session after the signal.")
    px = quotes.copy()
    px.index = _dates(px.index)
    if px.index.has_duplicates:
        raise ValueError("Duplicate price dates.")
    px = px.where(np.isfinite(px) & px.gt(0))
    frame["entry_price"] = frame.entry_date.map(px)
    frame["scheduled_exit"] = frame.entry_date + pd.Timedelta(days=90)
    exit_indices = calendar.searchsorted(frame.scheduled_exit)
    frame["exit_session"] = pd.NaT
    in_window = exit_indices < len(calendar)
    frame.loc[in_window, "exit_session"] = calendar[exit_indices[in_window]].to_numpy()
    frame["exit_price"] = frame.exit_session.map(px)
    valid = frame.entry_price.notna() & frame.exit_price.notna() & frame.exit_session.notna()
    frame["label_end"] = frame.exit_session.where(valid)
    frame["target"] = (frame.exit_price / frame.entry_price - 1).where(valid)
    bad_target = ~np.isfinite(frame.target) & frame.target.notna()
    frame.loc[bad_target, "target"] = np.nan
    frame.loc[bad_target, "label_end"] = pd.NaT
    return frame


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _schema(con, table: str) -> set[str]:
    # Table names are constants defined in this module, not user input.
    return set(con.execute(f"PRAGMA table_info('{table}')").fetchdf()["name"])


def _ranking_audit(membership: pd.DataFrame) -> dict[str, Any]:
    """Flag extreme ranking inputs; do not silently repair or exclude them."""
    close_candidates = [c for c in membership if any(token in c.lower() for token in ("close", "price")) and "date" not in c.lower()]
    share_candidates = [c for c in membership if "shares" in c.lower() and "date" not in c.lower()]
    numeric_cols = [c for c in close_candidates + share_candidates if pd.api.types.is_numeric_dtype(membership[c])]
    flag = pd.Series(False, index=membership.index)
    for column in numeric_cols:
        # Investigation thresholds, not declarations that every flagged row is wrong.
        threshold = 1_000_000 if column in close_candidates else 10_000_000_000
        flag |= membership[column].abs().gt(threshold)
    cols = [c for c in ["symbol", "quarter_start", "universe_rank"] + numeric_cols if c in membership]
    sample = membership.loc[flag, cols].head(25).copy()
    for c in sample.select_dtypes(include=["datetime", "datetimetz"]):
        sample[c] = sample[c].dt.strftime("%Y-%m-%d")
    return {
        "inspected_numeric_columns": numeric_cols,
        "flagged_rows": int(flag.sum()),
        "examples": sample.replace({np.nan: None}).to_dict("records"),
        "action": "Audit only; no security or date is removed using this report.",
        "caveat": "Thresholds identify review candidates, not verified valuation errors.",
    }


def prepare_data(database: Path, batch_size: int = 128) -> PreparedData:
    """Read the full bCourses delivery in batches and prepare the HW4 sample.

    All daily prices through TEST_END are retained for marks and exits, even
    outside membership dates. SQL reads are read-only and date-capped. The
    returned panel has monthly rows, including rows with unknown future labels.
    """
    database = Path(database).expanduser().resolve()
    if not database.is_file():
        raise FileNotFoundError(
            f"Missing {database}. Download equities_top2000.duckdb from bCourses. "
            "The final experiment cannot be produced from synthetic data or the class extract."
        )
    if database.name.lower() == "equities_class.duckdb":
        raise ValueError("equities_class.duckdb ends in 2015; use equities_top2000.duckdb for HW4.")
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")
    try:
        import duckdb
    except ImportError as error:
        raise ImportError("Install the homework requirements first: python -m pip install -r requirements.txt") from error

    size = database.stat().st_size
    sha256 = _file_digest(database)
    audit: dict[str, Any] = {
        "database_name": database.name,
        "size_bytes": size,
        "sha256": sha256,
        "matches_published_size": size == EXPECTED_SIZE,
        "matches_published_sha256": sha256 == EXPECTED_SHA256,
        "published_size_bytes": EXPECTED_SIZE,
        "published_sha256": EXPECTED_SHA256,
        "warmup_start": str(WARMUP_START.date()),
        "maximum_read_date": str(TEST_END.date()),
        "source_limitations": SOURCE_LIMITATIONS,
        "numeric_features": FEATURE_COLUMNS,
        "quarantined_invalid_prices_loaded_window": 0,
        "quarantined_off_calendar_prices_loaded_window": 0,
        "loaded_price_rows": 0,
        "invalid_quarterly_filing_rows_loaded": 0,
        "unknown_currency_rows_loaded": 0,
    }
    if sha256 != EXPECTED_SHA256:
        warnings.warn("Database checksum differs from the published course manifest; inspect data_audit.json before interpreting results.", stacklevel=2)

    with duckdb.connect(str(database), read_only=True) as con:
        tables = set(con.execute("SHOW TABLES").fetchdf().iloc[:, 0])
        required = {"prices", "fundamentals", "universe", "universe_membership", "benchmark", "calendar"}
        missing = required - tables
        if missing:
            raise ValueError(f"Database is missing required tables: {sorted(missing)}")
        requirements = {
            "prices": {"symbol", "date", "adj_close"},
            "fundamentals": set(FINANCIAL_INPUTS),
            "universe": {"symbol"},
            "universe_membership": {"symbol", "quarter_start", "selection_date", "universe_rank"},
            "benchmark": {"date", "adj_close"},
            "calendar": {"date"},
        }
        schemas = {table: _schema(con, table) for table in sorted(required)}
        for table, columns in requirements.items():
            if columns - schemas[table]:
                raise ValueError(f"Table {table} is missing columns: {sorted(columns - schemas[table])}")
        audit["schema_columns"] = {k: sorted(v) for k, v in schemas.items()}
        audit["table_rows"] = {
            table: int(con.execute(f'SELECT count(*) FROM "{table}"').fetchone()[0])
            for table in sorted(required | ({"universe_rebalances", "universe_candidates", "dividends"} & tables))
        }
        symbols = con.execute("SELECT symbol FROM universe ORDER BY symbol").fetchdf().symbol
        if symbols.isna().any() or symbols.duplicated().any() or symbols.empty:
            raise ValueError("The universe must contain one nonmissing symbol per security.")
        symbols = symbols.astype(str).tolist()
        audit["securities_in_union"] = len(symbols)
        if "is_active_snapshot" in schemas["universe"]:
            audit["currently_inactive_audit_only"] = int(con.execute("SELECT count(*) FROM universe WHERE is_active_snapshot = false").fetchone()[0])
        full_calendar = _dates(con.execute("SELECT date FROM calendar WHERE date<=? ORDER BY date", [TEST_END.date()]).fetchdf().date)
        if full_calendar.has_duplicates or full_calendar.isna().any():
            raise ValueError("The supplied market calendar has missing or duplicate dates.")
        calendar = full_calendar[full_calendar >= WARMUP_START]
        if len(calendar) == 0 or TEST_START not in calendar or TEST_END not in calendar or calendar.min() > pd.Timestamp("2009-12-01"):
            raise ValueError("Database lacks the full 2010 development / April 2016--September 2026 test calendar. Use the full homework file.")
        benchmark_filter = " AND symbol='^SP500TR'" if "symbol" in schemas["benchmark"] else ""
        benchmark_frame = con.execute(
            "SELECT date,adj_close FROM benchmark WHERE date>=? AND date<=?" + benchmark_filter + " ORDER BY date",
            [WARMUP_START.date(), TEST_END.date()],
        ).fetchdf()
        benchmark_frame["date"] = _dates(benchmark_frame.date)
        if benchmark_frame.date.duplicated().any():
            raise ValueError("Duplicate benchmark sessions.")
        benchmark = benchmark_frame.set_index("date").adj_close.reindex(calendar).astype(float)
        if not (np.isfinite(benchmark) & benchmark.gt(0)).all():
            raise ValueError("The ^SP500TR benchmark has a missing or invalid calendar-session quote.")
        benchmark.name = "SP500TR"
        audit["market_sessions_loaded"] = len(calendar)
        audit["first_market_session_loaded"] = str(calendar[0].date())
        audit["last_market_session_loaded"] = str(calendar[-1].date())

        membership = con.execute("SELECT * FROM universe_membership WHERE quarter_start<=? ORDER BY quarter_start,symbol", [TEST_END.date()]).fetchdf()
        for col in membership:
            if "date" in col or col in ("quarter_start", "quarter_end"):
                if pd.api.types.is_datetime64_any_dtype(membership[col]):
                    membership[col] = pd.to_datetime(membership[col]).astype("datetime64[ns]")
        audit["ranking_inputs"] = _ranking_audit(membership)
        required_quarters = pd.date_range("2010-01-01", "2026-07-01", freq="QS")
        absent = required_quarters.difference(_dates(membership.quarter_start.unique()))
        if len(absent):
            raise ValueError(f"Missing membership quarters: {[str(x.date()) for x in absent]}")
        audit["membership_count_by_quarter"] = {
            str(pd.Timestamp(q).date()): int(n)
            for q, n in membership.groupby("quarter_start").size().items()
        }
        audit["invalid_adjusted_price_rows_all_history"] = int(con.execute(
            "SELECT count(*) FROM prices WHERE date<=? AND (adj_close IS NULL OR NOT isfinite(adj_close) OR adj_close<=0)", [TEST_END.date()]
        ).fetchone()[0])
        audit["off_calendar_price_rows_all_history"] = int(con.execute(
            "SELECT count(*) FROM prices p WHERE p.date<=? AND NOT EXISTS (SELECT 1 FROM calendar c WHERE c.date=p.date)", [TEST_END.date()]
        ).fetchone()[0])
        audit["invalid_filing_rows_all_history"] = int(con.execute(
            "SELECT count(*) FROM fundamentals WHERE fiscal_date<=? AND (filing_date IS NULL OR filing_date<=fiscal_date)", [TEST_END.date()]
        ).fetchone()[0])

        schedule = monthly_schedule(calendar)
        price_blocks = []
        panel_blocks = []
        for offset in range(0, len(symbols), batch_size):
            batch = symbols[offset: offset + batch_size]
            prices = con.execute(
                "SELECT symbol,date,adj_close FROM prices WHERE date>=? AND date<=? AND symbol IN (SELECT unnest(?)) ORDER BY symbol,date",
                [WARMUP_START.date(), TEST_END.date(), batch],
            ).fetchdf()
            prices["date"] = _dates(prices.date)
            if prices.duplicated(["symbol", "date"]).any():
                raise ValueError("Duplicate stock/session price keys; resolve the database before running.")
            invalid = ~np.isfinite(prices.adj_close) | prices.adj_close.le(0)
            off_calendar = ~prices.date.isin(calendar)
            audit["loaded_price_rows"] += len(prices)
            audit["quarantined_invalid_prices_loaded_window"] += int(invalid.sum())
            audit["quarantined_off_calendar_prices_loaded_window"] += int(off_calendar.sum())
            prices = prices.loc[~invalid & ~off_calendar]
            wide = prices.pivot(index="date", columns="symbol", values="adj_close").reindex(index=calendar, columns=batch).astype(float)
            wide.index.name = "date"
            price_blocks.append(wide)
            financials = con.execute(
                "SELECT " + ",".join(FINANCIAL_INPUTS) + " FROM fundamentals WHERE fiscal_date<=? AND (filing_date<=? OR filing_date IS NULL) AND symbol IN (SELECT unnest(?)) ORDER BY symbol,fiscal_date",
                [TEST_END.date(), TEST_END.date(), batch],
            ).fetchdf()
            for col in ("fiscal_date", "filing_date"):
                financials[col] = pd.to_datetime(financials[col]).astype("datetime64[ns]")
            audit["invalid_quarterly_filing_rows_loaded"] += int((financials.period.eq("quarter") & ~financials.filing_date.gt(financials.fiscal_date)).sum())
            audit["unknown_currency_rows_loaded"] += int((financials.reported_currency.isna() | financials.reported_currency.fillna("").astype(str).str.strip().eq("")).sum())
            events = make_financial_events(financials)
            monthly_rows = []
            for symbol in batch:
                features = make_price_features(wide[symbol], calendar).reindex(schedule.signal_date).reset_index(drop=True)
                sampled = pd.concat([schedule.reset_index(drop=True), features], axis=1)
                sampled["symbol"] = symbol
                sampled = make_holding_labels(sampled, wide[symbol], calendar)
                monthly_rows.append(sampled)
            block = pd.concat(monthly_rows, ignore_index=True)
            block = join_financial_events(block, events)
            block = apply_dated_membership(block, membership.loc[membership.symbol.isin(batch)])
            panel_blocks.append(block)

    panel = pd.concat(panel_blocks, ignore_index=True).sort_values(["signal_date", "symbol"]).reset_index(drop=True)
    price_matrix = pd.concat(price_blocks, axis=1)
    audit["monthly_rows_all_union"] = len(panel)
    audit["eligible_rows"] = int(panel.eligible.sum())
    audit["eligible_rows_with_unknown_targets"] = int((panel.eligible & panel.target.isna()).sum())
    audit["completed_labels"] = int((panel.eligible & panel.target.notna()).sum())
    audit["eligible_rows_without_any_known_filing"] = int((panel.eligible & panel.available_date.isna()).sum())
    audit["eligible_count_by_entry_month"] = {
        str(pd.Timestamp(entry).date()): int(group.eligible.sum())
        for entry, group in panel.groupby("entry_date")
    }
    if not panel.loc[panel.eligible, FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan).equals(panel.loc[panel.eligible, FEATURE_COLUMNS]):
        raise ValueError("Feature construction produced infinite values.")
    return PreparedData(panel=panel, prices=price_matrix, benchmark=benchmark, calendar=calendar, audit=audit)


# 2. Sourced accounting inputs, separate from model observations
# --------------------------------------------------------------
# Source-verified accounting repairs, kept separate from model inputs.
#
# The original features, training outcomes and frozen scores are never edited.
# Mandatory consideration and security identifiers are reconciled only inside
# the portfolio accounting stage. Every external input has a source record.

HERE = Path(__file__).resolve().parent


def load_registry(path=None):
    path = Path(path) if path else HERE / "corporate_actions.json"
    registry = json.loads(path.read_text(encoding="utf-8"))
    registry["file_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return registry


def action_frames(registry):
    """Only explicitly supported events become engine inputs."""
    cash, securities = [], []
    for event in registry["events"]:
        if "cash_per_adjusted_unit" in event:
            cash.append({key: event[key] for key in [
                "action_id", "symbol", "effective_date",
                "cash_per_adjusted_unit", "source_url",
            ]})
        elif event.get("event_type") == "security_conversion":
            securities.append({key: event[key] for key in [
                "action_id", "symbol", "effective_date", "successor_symbol",
                "units_per_adjusted_unit", "source_url",
            ]})
    return (pd.DataFrame(cash) if cash else None,
            pd.DataFrame(securities) if securities else None)


def overlay_prices(prices, registry, supplement=None):
    """Fill only documented missing accounting quotes; return full lineage.

    Alias continuations use a verified same-security mapping and an explicit
    adjusted-unit scale. They are actual successor observations in the same
    dollar-return unit as the legacy research identifier, never a stale carry.
    Original finite quotes are retained; no feature-building call uses this.
    """
    corrected = prices.copy()
    lineage = []
    for event in registry.get("events", []):
        if event.get("event_type") != "ticker_alias":
            continue
        symbol, successor = event["symbol"], event["successor_symbol"]
        factor = float(event["price_multiplier"])
        if not np.isfinite(factor) or factor <= 0:
            raise ValueError("Alias multiplier must be finite and positive")
        if successor not in corrected:
            continue  # A required unavailable successor will still stop the engine.
        if symbol not in corrected:
            corrected[symbol] = np.nan
        start = pd.Timestamp(event.get("overlay_start", event["effective_date"]))
        end = pd.Timestamp(event.get("overlay_end", "2026-09-30"))
        mask = (corrected.index >= start) & (corrected.index <= end)
        mask &= corrected[symbol].isna() & corrected[successor].notna()
        for date in corrected.index[mask]:
            observed = float(corrected.at[date, successor])
            price = observed * factor
            corrected.at[date, symbol] = price
            lineage.append({
                "symbol": symbol, "date": date, "adj_close": price,
                "source_symbol": successor, "source_adj_close": observed,
                "price_multiplier": factor, "action_id": event["action_id"],
                "source_url": event["source_url"], "type": "verified_ticker_alias",
            })
    if supplement is not None and Path(supplement).is_file():
        data = pd.read_csv(supplement, parse_dates=["date"])
        if data.duplicated(["symbol", "date"]).any():
            raise ValueError("Duplicate supplemental quote")
        for row in data.to_dict("records"):
            symbol, date, price = row["symbol"], row["date"], float(row["adj_close"])
            if not np.isfinite(price) or price <= 0 or not row.get("source_url"):
                raise ValueError("Supplemental quotes need positive values and sources")
            if date not in corrected.index:
                raise ValueError("Supplemental quote is outside the study calendar")
            if symbol not in corrected:
                corrected[symbol] = np.nan
            if pd.notna(corrected.at[date, symbol]):
                if not np.isclose(corrected.at[date, symbol], price, rtol=1e-8):
                    raise ValueError("Supplement must not overwrite an original observed quote")
                continue
            corrected.at[date, symbol] = price
            lineage.append(row | {"type": "sourced_supplemental_observation"})
    return corrected, pd.DataFrame(lineage)


# 3. Post-run diagnosis of the original source data
# -------------------------------------------------
# Post-run diagnosis of supplied data; never a trading input or repair.
#
# This script independently reconstructs monthly development labels, quantifies
# extreme source returns, and records the BID price gap that stopped the frozen
# experiment. It reads DuckDB in read-only mode and writes small audit artifacts.
# It does not exclude securities, correct prices, infer payouts, or rerun a new
# strategy. Later records are inspected only to document mapping uncertainty.

DEVELOPMENT_LABEL_QUERY = """
WITH cal AS (
    SELECT date,
           row_number() OVER (ORDER BY date) AS session_id,
           lag(date) OVER (ORDER BY date) AS signal_date
    FROM calendar WHERE date <= '2016-03-31'
), monthly AS (
    SELECT date AS entry_date, signal_date,
           row_number() OVER (
               PARTITION BY date_trunc('month', date) ORDER BY date
           ) AS month_position
    FROM cal WHERE date >= '2010-01-01' AND date <= '2015-12-31'
), schedule AS (
    SELECT entry_date, signal_date,
           (SELECT min(date) FROM cal
            WHERE date >= m.entry_date + INTERVAL '90 days') AS label_end
    FROM monthly m WHERE month_position = 1
), px AS (
    SELECT p.symbol, p.date, p.adj_close, c.session_id,
           lag(c.session_id, 63) OVER (
               PARTITION BY p.symbol ORDER BY p.date
           ) AS old_session
    FROM prices p JOIN cal c USING (date)
    WHERE p.date >= '2009-01-01'
      AND isfinite(p.adj_close) AND p.adj_close > 0
)
SELECT s.signal_date, s.entry_date, s.label_end, p.symbol,
       ep.adj_close AS entry_price, xp.adj_close AS exit_price,
       xp.adj_close / ep.adj_close - 1 AS target
FROM schedule s
JOIN px p ON p.date = s.signal_date
         AND p.session_id - p.old_session = 63
JOIN universe_membership u
  ON u.symbol = p.symbol
 AND u.quarter_start = date_trunc('quarter', s.entry_date)
 AND u.selection_date <= s.signal_date
LEFT JOIN px ep ON ep.symbol = p.symbol AND ep.date = s.entry_date
LEFT JOIN px xp ON xp.symbol = p.symbol AND xp.date = s.label_end
ORDER BY s.signal_date, p.symbol
"""


def _safe(value):
    """Make strict JSON values without NaN, NaT or NumPy scalar leakage."""
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if value is None or value is pd.NaT or value is pd.NA:
        return None
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return pd.Timestamp(value).strftime("%Y-%m-%d")
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, Path):
        return str(value)
    return value


def summarize_development(labels: pd.DataFrame) -> tuple[dict, list[dict], pd.DataFrame]:
    """Summarize observed labels without changing them or fitting a new model."""
    labels = labels.copy()
    for column in ("signal_date", "entry_date", "label_end"):
        labels[column] = pd.to_datetime(labels[column])
    finite = np.isfinite(labels.target)
    finished = labels.loc[finite & labels.label_end.le("2015-12-31")]
    if finished.empty:
        raise ValueError("No completed development labels are available for diagnosis")
    quantile_levels = [0, 0.001, 0.01, 0.5, 0.95, 0.99, 0.999, 1]
    extremes = finished.nlargest(15, "target").copy()
    extremes["target_percent"] = 100 * extremes.target
    development = {
        "eligible_monthly_rows": len(labels),
        "completed_by_development_end": len(finished),
        "unknown_target_rows": int((~finite).sum()),
        "known_targets_finishing_after_development": int(
            (finite & labels.label_end.gt("2015-12-31")).sum()
        ),
        "unknown_or_unfinished_by_development_end": len(labels) - len(finished),
        "rows_with_missing_entry_price": int(labels.entry_price.isna().sum()),
        "rows_with_missing_exit_price": int(labels.exit_price.isna().sum()),
        "target_quantiles_decimal": {
            str(q): float(v) for q, v in finished.target.quantile(quantile_levels).items()
        },
        "target_counts_above_decimal_threshold": {
            str(t): int(finished.target.gt(t).sum()) for t in [1, 10, 100, 1000]
        },
        "extreme_label_examples": extremes.to_dict("records"),
        "eligibility_reconstruction": (
            "Entry-quarter membership known at signal close plus 64 consecutive "
            "positive finite adjusted-price observations through the signal "
            "(63 complete trailing returns). Missing financial ratios do not "
            "exclude an otherwise eligible name. This independently reproduces "
            "the actual experiment's development counts."
        ),
        "outcome_window": (
            "2010-2015 entries; late-2015 outcomes may finish in the buffer. "
            "Quantiles and extreme examples use only labels ended by 2015-12-31."
        ),
    }
    folds = []
    for year in (2013, 2014, 2015):
        candidates = labels.loc[labels.entry_date.dt.year.eq(year)]
        if candidates.empty:
            continue
        asof = candidates.signal_date.min()
        train = labels.loc[
            labels.signal_date.lt(asof) & labels.label_end.lt(asof) & finite
        ]
        valid = candidates.loc[
            candidates.label_end.le("2015-12-31") & np.isfinite(candidates.target)
        ]
        if train.empty or valid.empty:
            continue
        mean = float(train.target.mean())
        squared_deviations = (train.target - mean) ** 2
        total_sse = float(squared_deviations.sum())
        folds.append({
            "validation_year": year,
            "fit_date": asof,
            "training_rows": len(train),
            "validation_rows": len(valid),
            "last_training_label_end": train.label_end.max(),
            "training_target_mean_decimal": mean,
            "training_target_median_decimal": float(train.target.median()),
            "training_target_max_decimal": float(train.target.max()),
            "dummy_rmse_decimal": float(np.sqrt(np.mean((valid.target - mean) ** 2))),
            "top_target_centered_sse_share": {
                str(n): float(squared_deviations.nlargest(n).sum() / total_sse)
                if total_sse else None for n in [1, 10, 100]
            },
        })
    return _safe(development), _safe(folds), extremes


def audit_data_issues(database, output_dir="outputs") -> dict:
    """Write data_issues.json and extreme_development_labels.csv; return audit.

    Only call this as a post-run diagnostic. Its fixed BID inspection dates
    describe the observed failed experiment and are never a historical screen.
    """
    import duckdb

    database = Path(database).expanduser().resolve()
    if not database.is_file():
        raise FileNotFoundError(database)
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(str(database), read_only=True) as con:
        labels = con.execute(DEVELOPMENT_LABEL_QUERY).fetchdf()
        development, folds, extremes = summarize_development(labels)
        bid_quotes = con.execute("""
            SELECT date,close,adj_close,volume,source FROM prices
            WHERE symbol='BID' AND date BETWEEN '2019-09-23' AND '2019-10-31'
            ORDER BY date
        """).fetchdf()
        before_gap = con.execute("""
            SELECT date,close,adj_close,volume,source FROM prices
            WHERE symbol='BID' AND date<='2019-10-02'
              AND isfinite(adj_close) AND adj_close>0
            ORDER BY date DESC LIMIT 1
        """).fetchdf()
        after_gap = con.execute("""
            SELECT date,close,adj_close,volume,source FROM prices
            WHERE symbol='BID' AND date>'2019-10-02' AND date<='2026-09-30'
              AND isfinite(adj_close) AND adj_close>0
            ORDER BY date LIMIT 1
        """).fetchdf()
        missing = con.execute("""
            SELECT c.date FROM calendar c
            LEFT JOIN prices p ON p.symbol='BID' AND p.date=c.date
            WHERE c.date BETWEEN '2019-10-03' AND '2019-10-10'
              AND (p.adj_close IS NULL OR NOT isfinite(p.adj_close) OR p.adj_close<=0)
            ORDER BY c.date
        """).fetchdf()
        snapshot = con.execute("SELECT * FROM universe WHERE symbol='BID'").fetchdf()
        dividends = con.execute("""
            SELECT * FROM dividends
            WHERE symbol='BID' AND ex_date BETWEEN '2019-01-01' AND '2019-12-31'
            ORDER BY ex_date
        """).fetchdf()
        tables = set(con.execute("SHOW TABLES").fetchdf().iloc[:, 0])
    bid = {
        "last_positive_quote_before_gap": before_gap.to_dict("records"),
        "first_positive_quote_after_gap": after_gap.to_dict("records"),
        "missing_sessions_through_stop": missing.date.tolist(),
        "missing_session_count_through_stop": len(missing),
        "price_rows_near_gap": bid_quotes.to_dict("records"),
        "source_snapshot": snapshot.to_dict("records"),
        "dividend_rows_in_2019": dividends.to_dict("records"),
        "supplied_tables": sorted(tables),
        "mapping_caveat": (
            "The static security name/status is not a dated issuer mapping. "
            "A later quote under the same ticker cannot establish continuity "
            "with the 2019 lot. No cash payout, zero recovery, or tradeable "
            "exit is inferred from the gap or from snapshot metadata."
        ),
    }
    audit = _safe({
        "diagnostic_scope": (
            "Post-run diagnosis only. Development anomalies and the observed "
            "BID gap are documented without changing features, membership, "
            "training targets, orders, or the frozen strategy."
        ),
        "database_name": database.name,
        "development": development,
        "development_folds": folds,
        "bid": bid,
        "interpretation": [
            "Very large adjusted-price ratios are unverified source anomalies, not confirmed investment gains.",
            "A few extreme outcomes dominate squared-error training. Feature clipping does not clip the target.",
            "The training-mean comparison also inherits the extreme target mean; its lower RMSE does not validate the data.",
            "The original Ridge experiment is retained as an unsuccessful modeling result, with no test-period retuning.",
            "The unaugmented BID history stopped the original run under its five-session stale-mark limit; this SQL diagnosis preserves that original incident.",
            "The separate corporate_actions.json now documents externally verified entitlements and identifier mappings. These accounting repairs never modify this diagnostic's original model inputs or targets.",
        ],
    })
    (output_dir / "data_issues.json").write_text(
        json.dumps(audit, indent=2, allow_nan=False), encoding="utf-8"
    )
    extremes.to_csv(output_dir / "extreme_development_labels.csv", index=False)
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", required=True, type=Path)
    parser.add_argument("--output", default=Path("outputs"), type=Path)
    args = parser.parse_args()
    audit = audit_data_issues(args.database, args.output)
    print(json.dumps({
        "development": {
            key: audit["development"][key] for key in [
                "eligible_monthly_rows", "completed_by_development_end", "unknown_target_rows"
            ]
        },
        "bid_missing_sessions": audit["bid"]["missing_session_count_through_stop"],
        "output": str(args.output.resolve()),
    }, indent=2))
