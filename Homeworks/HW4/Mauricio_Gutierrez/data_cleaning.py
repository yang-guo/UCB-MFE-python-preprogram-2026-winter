"""HW4 cleaning rules, extracted unchanged from notebook 01.

Inputs: source tables, market calendar, and known symbols.
Outputs: retained rows, quarantine, duplicates, and audit records.
Never fill prices or volume. Never remove a stock for a short history.
Daily changes exist only for the notebook's price-quality checks.
Both notebook 01 and clean_full_database.py import this module.
"""
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd


RETURN_WINDOW = 63       # preceding market sessions


MIN_RETURN_HISTORY = 20 # preceding observed daily returns


OUTLIER_STD = 3.0


LARGE_MOVE = 0.30


FACTOR_CHANGE = 2.0


def file_sha256(path):
    """Hash the file in small blocks without loading it all into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_symbol(values):
    """Adapt HW2's label normalization. Keep punctuation inside identifiers."""
    return (values.astype("string")
            .str.replace(r"<[^>]+>", "", regex=True)
            .str.strip().str.strip("'\"").str.strip().str.upper())


ISSUE_COLUMNS = ["source_table", "source_row", "symbol", "issue_date", "issue_type", "severity", "action", "description"]


OHLC_COLUMNS = ["open", "high", "low", "close"]


PRICE_COLUMNS = OHLC_COLUMNS + ["adj_close"]


def make_issues(rows, table, date_column, issue_type, description, severity, action):
    """Adapt the HW2 issue format. Preserve a link to each loaded source row."""
    return pd.DataFrame({
        "source_table": table,
        "source_row": rows["source_row"].to_numpy(),
        "symbol": rows["symbol"].to_numpy(),
        "issue_date": rows[date_column].to_numpy(),
        "issue_type": issue_type,
        "severity": severity,
        "action": action,
        "description": description.to_numpy() if isinstance(description, pd.Series) else description,
    }, columns=ISSUE_COLUMNS)


def concat_issues(parts):
    """Combine nonempty reports without changing their observations."""
    populated = [part for part in parts if not part.empty]
    return pd.concat(populated, ignore_index=True) if populated else pd.DataFrame(columns=ISSUE_COLUMNS)


def prepare_rows(raw_frame, table, date_columns):
    """Copy data, preserve source keys, and parse dates. Never mutate the input."""
    frame = raw_frame.copy(deep=True).reset_index(drop=True)
    frame["source_row"] = np.arange(len(frame), dtype=np.int64)
    frame["raw_symbol"] = frame["symbol"]
    frame["symbol"] = normalize_symbol(frame["symbol"])
    for column in date_columns:
        frame[f"raw_{column}"] = frame[column]
        frame[column] = pd.to_datetime(frame[column], errors="coerce")
    changed = frame["symbol"].ne(frame["raw_symbol"].astype("string")).fillna(False)
    frame["flag_symbol_normalized"] = changed
    issues = make_issues(frame.loc[changed], table, date_columns[0], "symbol_format", "Normalized the label; retained raw_symbol.", "low", "normalize")
    return frame, [issues]


def split_duplicates(frame, keys, value_columns, table, date_column):
    """Remove identical copies, then quarantine every remaining conflicting key."""
    exact = frame.duplicated(subset=value_columns, keep="first")
    duplicates = frame.loc[exact].copy()
    duplicates["removal_reason"] = "exact duplicate after key normalization"
    unique = frame.loc[~exact].copy()
    conflict = unique.duplicated(keys, keep=False)
    conflicts = unique.loc[conflict].copy()
    conflicts["quarantine_reason"] = "conflicting key"
    issues = [
        make_issues(duplicates, table, date_column, "exact_duplicate", "Kept one identical copy.", "low", "remove_duplicate"),
        make_issues(conflicts, table, date_column, "conflicting_key", "No verified way to choose between different records with this key.", "high", "quarantine"),
    ]
    return unique.loc[~conflict].copy(), duplicates, conflicts, issues


def split_invalid_rows(frame, rules, table, date_column):
    """Apply named invalid-row masks. Record every reason; count each row once."""
    reasons = pd.Series("", index=frame.index, dtype="string")
    issues = []
    for name, mask in rules.items():
        mask = mask.fillna(False)
        reasons.loc[mask] += name + "; "
        issues.append(make_issues(frame.loc[mask], table, date_column, name, name.replace("_", " "), "high", "quarantine"))
    rejected = reasons.ne("")
    quarantined = frame.loc[rejected].copy()
    quarantined["quarantine_reason"] = reasons.loc[rejected].str.rstrip("; ")
    return frame.loc[~rejected].copy(), quarantined, issues


def join_quarantine(*parts):
    """Preserve the schema even when no rows need quarantine."""
    nonempty = [part for part in parts if not part.empty]
    return pd.concat(nonempty, ignore_index=True) if nonempty else parts[0].iloc[:0].copy()


def row_accounting(raw_frame, cleaned, duplicates, quarantined, table):
    """Check a disjoint partition of all input row identifiers."""
    all_ids = pd.concat([cleaned["source_row"], duplicates["source_row"], quarantined["source_row"]])
    assert len(all_ids) == len(raw_frame) and all_ids.is_unique
    assert set(all_ids) == set(range(len(raw_frame)))
    return {"table": table, "raw_rows": len(raw_frame), "kept_rows": len(cleaned),
            "duplicate_rows": len(duplicates), "quarantined_rows": len(quarantined)}


def clean_prices(raw_frame, market_calendar, symbols):
    """Adapt HW2 price cleaning without price filling or future-based repairs."""
    frame, issues = prepare_rows(raw_frame, "prices", ["date"])
    frame, duplicates, conflicts, duplicate_issues = split_duplicates(
        frame, ["symbol", "date"], list(raw_frame.columns), "prices", "date"
    )
    issues.extend(duplicate_issues)
    for column in PRICE_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    price_values = frame[PRICE_COLUMNS]
    invalid_price = (~np.isfinite(price_values) | price_values.le(0)).any(axis=1)
    broken_bounds = (frame["low"].gt(frame["high"]) | frame["low"].gt(frame["open"])
                     | frame["low"].gt(frame["close"]) | frame["high"].lt(frame["open"])
                     | frame["high"].lt(frame["close"]))
    rules = {
        "missing_symbol": frame["symbol"].isna() | frame["symbol"].eq(""),
        "unknown_symbol": frame["symbol"].notna() & ~frame["symbol"].isin(symbols),
        "invalid_date": frame["date"].isna(),
        "off_calendar_date": frame["date"].notna() & ~frame["date"].isin(market_calendar),
        "unusable_price": invalid_price,
        "ohlc_bounds": broken_bounds,
    }
    kept, rejected, invalid_issues = split_invalid_rows(frame, rules, "prices", "date")
    issues.extend(invalid_issues)
    kept["raw_volume"] = kept["volume"]
    numeric_volume = pd.to_numeric(kept["volume"], errors="coerce").astype(float)
    invalid_volume = kept["volume"].notna() & (~np.isfinite(numeric_volume) | numeric_volume.lt(0))
    kept["flag_volume_invalid"] = invalid_volume
    kept["flag_volume_missing"] = kept["volume"].isna()
    kept["volume"] = numeric_volume.mask(invalid_volume)
    kept["flag_zero_volume"] = kept["volume"].eq(0)
    for flag, issue_type, action, explanation in [
        ("flag_volume_invalid", "invalid_volume", "set_missing", "Invalid volume replaced with missing; raw_volume retained."),
        ("flag_volume_missing", "missing_volume", "keep_missing", "Unknown volume remains missing."),
        ("flag_zero_volume", "zero_volume", "keep_flagged", "Zero volume is a review item, not proof of an invalid price."),
    ]:
        issues.append(make_issues(kept.loc[kept[flag]], "prices", "date", issue_type, explanation, "low" if issue_type == "zero_volume" else "medium", action))
    quarantined = join_quarantine(conflicts, rejected)
    kept = kept.sort_values(["symbol", "date"]).reset_index(drop=True)
    accounting = row_accounting(raw_frame, kept, duplicates, quarantined, "prices")
    return kept, quarantined, duplicates, concat_issues(issues), accounting


def add_past_only_price_flags(prices, market_calendar):
    """Use only current/prior prices. Missing sessions stay missing."""
    parts, gaps = [], []
    for symbol, group in prices.groupby("symbol", sort=True):
        sessions = market_calendar[(market_calendar >= group["date"].min()) & (market_calendar <= group["date"].max())]
        daily = group.set_index("date").reindex(sessions)
        observed = daily["source_row"].notna()
        returns = daily["adj_close"].pct_change(fill_method=None)
        prior_returns = returns.shift(1)
        prior_mean = prior_returns.rolling(RETURN_WINDOW, min_periods=MIN_RETURN_HISTORY).mean()
        prior_std = prior_returns.rolling(RETURN_WINDOW, min_periods=MIN_RETURN_HISTORY).std(ddof=1)
        usable_std = prior_std.where(prior_std > 0)
        daily["daily_return"] = returns
        daily["prior_return_zscore"] = (returns - prior_mean) / usable_std
        daily["flag_return_outlier"] = daily["prior_return_zscore"].abs().gt(OUTLIER_STD)
        daily["flag_large_move"] = returns.abs().gt(LARGE_MOVE)
        factor = daily["adj_close"] / daily["close"]
        factor_change = factor / factor.shift(1)
        daily["flag_factor_change"] = factor_change.gt(FACTOR_CHANGE) | factor_change.lt(1 / FACTOR_CHANGE)
        daily["flag_gap_before_quote"] = observed & daily["adj_close"].shift(1).isna()
        if len(daily):
            daily.iloc[0, daily.columns.get_loc("flag_gap_before_quote")] = False
        gaps.extend({"symbol": symbol, "date": date, "issue_type": "missing_usable_quote"} for date in sessions[~observed])
        # Return only real observations; a missing calendar row is never a quote.
        selected = daily.loc[observed].copy()
        selected["symbol"] = symbol
        selected["source_row"] = selected["source_row"].astype("int64")
        selected.index.name = "date"
        parts.append(selected.reset_index())
    if not parts:
        raise ValueError("No usable prices remain. Review quarantine before continuing.")
    result = pd.concat(parts, ignore_index=True).sort_values(["symbol", "date"]).reset_index(drop=True)
    gap_report = pd.DataFrame(gaps, columns=["symbol", "date", "issue_type"])
    return result, gap_report


def clean_fundamentals(raw_frame, symbols, numeric_columns):
    """Keep usable quarterly events; preserve missingness and negative earnings."""
    frame, issues = prepare_rows(raw_frame, "fundamentals", ["fiscal_date", "filing_date"])
    frame, duplicates, conflicts, duplicate_issues = split_duplicates(
        frame, ["symbol", "fiscal_date", "period"], list(raw_frame.columns), "fundamentals", "fiscal_date"
    )
    issues.extend(duplicate_issues)
    rules = {
        "missing_symbol": frame["symbol"].isna() | frame["symbol"].eq(""),
        "unknown_symbol": frame["symbol"].notna() & ~frame["symbol"].isin(symbols),
        "invalid_fiscal_date": frame["fiscal_date"].isna(),
        "missing_filing_date": frame["filing_date"].isna(),
        "invalid_filing_lag": frame["filing_date"].le(frame["fiscal_date"]),
        "unsupported_period": ~frame["period"].eq("quarter"),
    }
    kept, rejected, invalid_issues = split_invalid_rows(frame, rules, "fundamentals", "fiscal_date")
    issues.extend(invalid_issues)
    kept["invalid_numeric_cells"] = 0
    for column in numeric_columns:
        if column not in kept.columns:
            continue
        numeric = pd.to_numeric(kept[column], errors="coerce")
        invalid = kept[column].notna() & ~np.isfinite(numeric)
        description = column + "=" + kept.loc[invalid, column].astype(str) + "; replaced with missing."
        issues.append(make_issues(kept.loc[invalid], "fundamentals", "fiscal_date", "invalid_numeric_cell", description, "medium", "set_missing"))
        kept["invalid_numeric_cells"] += invalid.astype(int)
        kept[column] = numeric.where(np.isfinite(numeric))
    kept["available_date"] = kept["filing_date"] + pd.Timedelta(days=1)
    currency = kept["reported_currency"].astype("string").str.strip()
    kept["flag_currency_unknown"] = currency.isna() | currency.eq("").fillna(False)
    kept["flag_market_cap_missing"] = kept["market_cap"].isna()
    for flag, explanation in {
        "flag_currency_unknown": "Reported currency is unknown; do not assume USD or compare raw amounts across currencies.",
        "flag_market_cap_missing": "Market capitalization is missing; no value has been invented.",
    }.items():
        issues.append(make_issues(kept.loc[kept[flag]], "fundamentals", "fiscal_date", flag.removeprefix("flag_"), explanation, "medium", "keep_missing"))
    kept = kept.sort_values(["symbol", "available_date", "fiscal_date"]).reset_index(drop=True)
    quarantined = join_quarantine(conflicts, rejected)
    accounting = row_accounting(raw_frame, kept, duplicates, quarantined, "fundamentals")
    return kept, quarantined, duplicates, concat_issues(issues), accounting


def latest_available_report(reports, symbol, signal_date, max_age_days=550):
    """Select the newest known fiscal period using only available reports."""
    date = pd.Timestamp(signal_date)
    known = reports.loc[reports["symbol"].eq(symbol) & reports["available_date"].le(date)]
    if known.empty:
        return known
    newest = known.sort_values(["fiscal_date", "available_date"]).tail(1)
    if (date - newest["fiscal_date"].iloc[0]).days > max_age_days:
        return newest.iloc[:0]
    return newest


def missing_value_summary(raw_frame, cleaned, table):
    """Compare source columns only; cleaning flags are not source fields."""
    return pd.DataFrame({
        "table": table,
        "column": raw_frame.columns,
        "missing_before": raw_frame.isna().sum().to_numpy(),
        "missing_after": cleaned.reindex(columns=raw_frame.columns).isna().sum().to_numpy(),
    })


def price_validation(frame, market_calendar, symbols):
    """Validate usable observations without requiring invented volume or prices."""
    values = frame[PRICE_COLUMNS]
    volume = frame["volume"]
    return {
        "prices: one row per symbol/date": not frame.duplicated(["symbol", "date"]).any(),
        "prices: valid known keys": frame["date"].notna().all() and frame["symbol"].isin(symbols).all(),
        "prices: positive finite prices": (np.isfinite(values) & values.gt(0)).all().all(),
        "prices: valid OHLC bounds": (frame["low"].le(frame[["open", "close"]].min(axis=1)) & frame["high"].ge(frame[["open", "close"]].max(axis=1))).all(),
        "prices: dates are market sessions": frame["date"].isin(market_calendar).all(),
        "prices: volume is missing or nonnegative and finite": (volume.isna() | (np.isfinite(volume) & volume.ge(0))).all(),
        "prices: no daily returns across a missing quote": frame.loc[frame["flag_gap_before_quote"], "daily_return"].isna().all(),
        "prices: daily returns are missing or finite": (frame["daily_return"].isna() | np.isfinite(frame["daily_return"])).all(),
    }


def reference_validation(raw):
    calendar = pd.DatetimeIndex(raw["calendar"]["date"])
    universe = raw["universe"]
    membership = raw["universe_membership"]
    benchmark = raw["benchmark"]
    known_symbols = set(universe["symbol"])
    reference_checks = pd.Series({
        "calendar is nonempty, unique, and ordered": len(calendar) > 0 and calendar.is_unique and calendar.is_monotonic_increasing,
        "calendar has no missing dates": not calendar.hasnans,
        "universe symbols are unique and present": universe["symbol"].notna().all() and not universe["symbol"].duplicated().any(),
        "universe symbols already use canonical labels": normalize_symbol(universe["symbol"]).eq(universe["symbol"]).all(),
        "benchmark has one row per session": pd.DatetimeIndex(benchmark["date"]).equals(calendar),
        "benchmark is SP500 total return": benchmark["symbol"].eq("^SP500TR").all(),
        "benchmark adjusted closes are positive and finite": (np.isfinite(benchmark["adj_close"]) & benchmark["adj_close"].gt(0)).all(),
        "membership keys are unique": not membership.duplicated(["symbol", "quarter_start"]).any(),
        "membership uses known stocks": membership["symbol"].isin(known_symbols).all(),
        "membership dates are present": membership[["quarter_start", "quarter_end", "selection_date", "effective_start"]].notna().all().all(),
        "selection precedes the effective entry date": membership["selection_date"].lt(membership["effective_start"]).all(),
        "effective entry belongs to its quarter": (membership["effective_start"].ge(membership["quarter_start"]) & membership["effective_start"].le(membership["quarter_end"])).all(),
        "selection uses already available shares": membership["shares_available_date"].le(membership["selection_date"]).all(),
        "supplied ranks are valid": membership["universe_rank"].between(1, 2000).all(),
    }, name="passed")
    return reference_checks
