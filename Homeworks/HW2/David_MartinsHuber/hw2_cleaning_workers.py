"""Importable process-pool worker for Homework 2 ticker cleaning."""

import numpy as np
import pandas as pd

RAW_PRICE_COLUMNS = ["open", "high", "low", "close"]
PRICE_COLUMNS = RAW_PRICE_COLUMNS + ["adj_close"]
REPAIR_FLAG_COLUMNS = [f"{column}_repaired" for column in PRICE_COLUMNS + ["volume"]]

def repair_ticker_group(group: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Repair one sorted ticker without reading or changing other ticker groups.
    work = group.sort_values("ts").copy()

    # For an observed invalid open only, use the midpoint of the valid low-close
    # interval when close is inside the day's positive [low, high] range.
    invalid_observed_open = work["open"].notna() & (~np.isfinite(work["open"]) | work["open"].le(0))
    valid_bracket = (
        np.isfinite(work["low"]) & work["low"].gt(0)
        & np.isfinite(work["close"]) & work["close"].gt(0)
        & np.isfinite(work["high"]) & work["high"].gt(0)
        & work["low"].le(work["close"]) & work["close"].le(work["high"])
    )
    midpoint_open = invalid_observed_open & valid_bracket
    work.loc[midpoint_open, "open"] = (work.loc[midpoint_open, "low"] + work.loc[midpoint_open, "close"]) / 2
    work.loc[midpoint_open, "open_repaired"] = True

    for column in RAW_PRICE_COLUMNS:
        invalid = ~np.isfinite(work[column]) | work[column].le(0)
        work.loc[invalid, column] = np.nan
        work.loc[invalid, f"{column}_repaired"] = True

    # An inverted high/low range does not identify which bound is wrong.
    # Treat both as suspect, then use earlier values from this ticker only.
    inverted_range = (
        np.isfinite(work["high"]) & np.isfinite(work["low"])
        & work["high"].lt(work["low"])
    )
    work.loc[inverted_range, ["high", "low"]] = np.nan
    work.loc[inverted_range, ["high_repaired", "low_repaired"]] = True

    bad_high = work["high"].notna() & ((work["open"].notna() & work["high"].lt(work["open"])) | (work["close"].notna() & work["high"].lt(work["close"])))
    bad_low = work["low"].notna() & ((work["open"].notna() & work["low"].gt(work["open"])) | (work["close"].notna() & work["low"].gt(work["close"])))
    work.loc[bad_high, "high"] = np.nan
    work.loc[bad_high, "high_repaired"] = True
    work.loc[bad_low, "low"] = np.nan
    work.loc[bad_low, "low_repaired"] = True

    for field in ["open", "close"]:
        outside = work[field].notna() & work["low"].notna() & work["high"].notna() & (work[field].lt(work["low"]) | work[field].gt(work["high"]))
        work.loc[outside, field] = np.nan
        work.loc[outside, f"{field}_repaired"] = True

    for column in RAW_PRICE_COLUMNS:
        missing = work[column].isna()
        work[column] = work[column].ffill()
        work.loc[missing & work[column].notna(), f"{column}_repaired"] = True

    lower = work[["open", "close"]].min(axis=1)
    upper = work[["open", "close"]].max(axis=1)
    expand_high = work["high"].notna() & upper.notna() & work["high"].lt(upper)
    expand_low = work["low"].notna() & lower.notna() & work["low"].gt(lower)
    work.loc[expand_high, "high"] = upper[expand_high]
    work.loc[expand_high, "high_repaired"] = True
    work.loc[expand_low, "low"] = lower[expand_low]
    work.loc[expand_low, "low_repaired"] = True

    reference_factor = work["_reference_factor"]
    has_reference = np.isfinite(reference_factor) & reference_factor.gt(0)
    expected_adjusted = work["close"] * reference_factor
    adjusted_valid = np.isfinite(work["adj_close"]) & work["adj_close"].gt(0)
    adjusted_matches = pd.Series(
        np.isclose(work["adj_close"].to_numpy(dtype=float), expected_adjusted.to_numpy(dtype=float), rtol=1e-4, atol=0.02, equal_nan=False),
        index=work.index,
    )
    reference_repair = has_reference & (~adjusted_valid | ~adjusted_matches)
    work.loc[reference_repair, "adj_close"] = expected_adjusted[reference_repair]
    work.loc[reference_repair, "adj_close_repaired"] = True

    original_close = work["_original_close"]
    original_adjusted = work["_original_adj_close"]
    row_factor_valid = np.isfinite(original_close) & np.isfinite(original_adjusted) & original_close.gt(0) & original_adjusted.gt(0)
    can_use_row_factor = ~has_reference & ~adjusted_valid & row_factor_valid & ~work["close_repaired"]
    work.loc[can_use_row_factor, "adj_close"] = work.loc[can_use_row_factor, "close"] * (original_adjusted[can_use_row_factor] / original_close[can_use_row_factor])
    work.loc[can_use_row_factor, "adj_close_repaired"] = True
    invalid_adjusted = ~np.isfinite(work["adj_close"]) | work["adj_close"].le(0)
    work.loc[invalid_adjusted, "adj_close_repaired"] = True

    invalid_volume = ~np.isfinite(work["volume"]) | work["volume"].lt(0)
    work.loc[invalid_volume, "volume"] = 0
    work.loc[invalid_volume, "volume_repaired"] = True

    prices_valid = np.isfinite(work[PRICE_COLUMNS]).all(axis=1) & work[PRICE_COLUMNS].gt(0).all(axis=1)
    bounds_valid = work["low"].le(work["high"]) & work["low"].le(work["open"]) & work["open"].le(work["high"]) & work["low"].le(work["close"]) & work["close"].le(work["high"])
    volume_valid = np.isfinite(work["volume"]) & work["volume"].ge(0)
    unresolved = ~(prices_valid & bounds_valid & volume_valid)
    quarantined = work.loc[unresolved].copy()
    if not quarantined.empty:
        quarantined["quarantine_reason"] = "A required price remained missing/invalid, or OHLC bounds remained unresolved after ticker-local repair."
    cleaned = work.loc[~unresolved].copy()
    cleaned["ts"] = cleaned["ts"].dt.strftime("%Y-%m-%d")
    internal = ["_reference_factor", "_original_close", "_original_adj_close", "_row_id"]
    cleaned = cleaned.drop(columns=internal)
    quarantined = quarantined.drop(columns=internal)
    return cleaned, quarantined


