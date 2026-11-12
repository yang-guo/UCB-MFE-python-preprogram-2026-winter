"""Cleaning and validation helpers for the HW2 ``ohlc_hw`` table.

``01_data_cleaning.ipynb`` walks through these steps in order:

1. Normalize ticker labels (before any grouping or deduplication).
2. Separate exact duplicate records from conflicting ``(ticker, ts)`` keys.
3. Diagnose: missing cells, missing sessions, 3-sigma return outliers, OHLC
   bounds, nonpositive / non-finite prices, zero volume -> data quality report.
4. Repair: drop all-missing OHLC rows, rebuild ``adj_close`` from trusted
   reference factors, forward-fill invalid raw rows as a block, fill unknown
   volume with 0 (flagged).
5. Validate the retained rows; quarantine anything not defensibly repaired.

Every function is pure: it returns new DataFrames and never touches the database.
Dates are pandas Timestamps inside this module; ``to_sql_frame`` converts them to
ISO ``YYYY-MM-DD`` strings for storage.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

KEY = ["ticker", "ts"]
RAW_PRICES = ["open", "high", "low", "close"]
PRICE_COLS = RAW_PRICES + ["adj_close"]
VALUE_COLS = PRICE_COLS + ["volume"]
REPORT_COLUMNS = ["ticker", "issue_type", "issue_date", "description", "severity"]

# Relative tolerance when comparing prices with the reference table (float noise only).
PRICE_RTOL = 1e-6

# Severity guide:
#   high   - the value cannot be used in any calculation; it must be repaired or quarantined
#   medium - repairable from a trusted source, or an absent observation
#   low    - informational; kept and flagged for human review
MISSING_SEVERITY = {"open": "high", "high": "high", "low": "high", "close": "high",
                    "adj_close": "medium", "volume": "medium"}

CLEAN_COLUMNS = KEY + VALUE_COLS + [
    "ticker_original", "adj_close_vendor", "repair_flags",
    "zero_volume_flag", "return_outlier", "return_outlier_class", "raw_rowid",
]

_TAG_RE = re.compile(r"<[^>]*>")


# ---------------------------------------------------------------------------
# 1. Ticker labels and loading
# ---------------------------------------------------------------------------

def normalize_ticker_label(label):
    """Strip HTML markup/entities and all whitespace, then upper-case one label.

    >>> normalize_ticker_label('<a><span style="color: red"> aapl </span></a>')
    'AAPL'
    """
    if pd.isna(label):
        return label
    text = html.unescape(_TAG_RE.sub("", str(label)))
    return re.sub(r"\s+", "", text).upper()


def normalize_tickers(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with a normalized ``ticker`` and the raw label in ``ticker_original``."""
    out = df.copy()
    out["ticker_original"] = out["ticker"]
    # Normalize each distinct label once; the table has few labels but many rows.
    mapping = {label: normalize_ticker_label(label) for label in out["ticker"].dropna().unique()}
    out["ticker"] = out["ticker"].map(mapping)
    return out


def ticker_audit(df: pd.DataFrame) -> pd.DataFrame:
    """Audit table of raw label -> normalized ticker with row counts."""
    return (df.groupby(["ticker", "ticker_original"], dropna=False).size()
              .rename("rows").reset_index()
              .sort_values(["ticker", "rows"], ascending=[True, False], ignore_index=True))


def prepare_raw(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalize tickers, parse ``ts`` and coerce numeric columns; adds empty ``repair_flags``.

    Unparseable dates become NaT and unparseable numbers become NaN, so later
    checks report them instead of the load crashing.
    """
    out = raw.copy()
    if "raw_rowid" not in out:
        out.insert(0, "raw_rowid", np.arange(len(out)))
    out = normalize_tickers(out)
    out["ts"] = pd.to_datetime(out["ts"], format="%Y-%m-%d", errors="coerce")
    for col in VALUE_COLS:
        out[col] = pd.to_numeric(out[col], errors="coerce").astype(float)
    out["repair_flags"] = ""
    return out


# ---------------------------------------------------------------------------
# 2. Duplicates
# ---------------------------------------------------------------------------

def split_duplicates(df: pd.DataFrame):
    """Split rows into (unique, exact_duplicates, conflicts).

    * exact duplicates repeat the key **and** every value (NaN counts as equal);
      the first copy is kept, the rest are returned for the audit.
    * conflicts share a ``(ticker, ts)`` key but disagree on some value. All
      copies are returned (none kept) because there is no basis to pick one.
    """
    exact = df.duplicated(subset=KEY + VALUE_COLS, keep="first")
    deduped = df.loc[~exact]
    conflict = deduped.duplicated(subset=KEY, keep=False)
    return deduped.loc[~conflict].copy(), df.loc[exact].copy(), deduped.loc[conflict].copy()


# ---------------------------------------------------------------------------
# 3. Diagnosis
# ---------------------------------------------------------------------------

def missing_value_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Count and percentage of missing cells per column."""
    counts = df.isna().sum()
    return pd.DataFrame({"missing": counts, "pct": counts / max(len(df), 1)})


def find_missing_sessions(df: pd.DataFrame, reference_keys: pd.DataFrame) -> pd.DataFrame:
    """Reference ``(ticker, ts)`` keys absent from ``df``.

    ``reference_keys`` is each ticker's expected trading calendar (the supplied
    ``ohlc`` table's dates). Only tickers present in ``df`` are checked. Using a
    reference calendar avoids flagging exchange holidays as gaps; no prices are
    invented for the absent dates.
    """
    expected = reference_keys[KEY].drop_duplicates()
    expected = expected[expected["ticker"].isin(df["ticker"].unique())]
    merged = expected.merge(df[KEY].drop_duplicates(), on=KEY, how="left", indicator=True)
    missing = merged.loc[merged["_merge"] == "left_only", KEY]
    return missing.sort_values(KEY).reset_index(drop=True)


def gap_summary(df: pd.DataFrame, missing_sessions: pd.DataFrame) -> pd.DataFrame:
    """One row per ticker: "Ticker X is missing Y trading days between [start] and [end]"."""
    span = df.groupby("ticker")["ts"].agg(start="min", end="max")
    counts = missing_sessions.groupby("ticker").size().reindex(span.index, fill_value=0)
    out = span.assign(missing_days=counts.astype(int)).reset_index()
    out["message"] = [
        f"Ticker {t} is missing {n} trading days between {s:%Y-%m-%d} and {e:%Y-%m-%d}"
        for t, n, s, e in zip(out["ticker"], out["missing_days"], out["start"], out["end"])
    ]
    return out


def simple_returns(df: pd.DataFrame, price: str = "close") -> pd.Series:
    """``(price - prev_price) / prev_price`` within each ticker.

    ``df`` must already be sorted by ticker then date. The first row of each
    ticker and any return involving a zero/missing price is NaN (undefined).
    """
    if not df.sort_values(KEY).index.equals(df.index):
        raise ValueError("Sort by ticker and ts before calculating returns.")
    prev = df.groupby("ticker")[price].shift(1)
    returns = (df[price] - prev) / prev
    return returns.where(np.isfinite(returns))


def three_sigma_flags(returns: pd.Series, tickers: pd.Series, k: float = 3.0) -> pd.DataFrame:
    """Flag ``abs(r - mean) > k * std`` per ticker (sample std, ddof=1, both tails).

    Undefined returns are excluded from the mean/std and never flagged. A ticker
    whose std is undefined (< 2 returns) or zero is not screened at all; its rows
    get ``screen_status`` 'undefined_or_zero_std' instead of a silent False.
    """
    r = returns.where(np.isfinite(returns))
    grouped = r.groupby(tickers)
    mean = grouped.transform("mean")
    std = grouped.transform("std")  # pandas default ddof=1
    usable = r.notna() & std.notna() & (std > 0)
    flag = usable & ((r - mean).abs() > k * std)
    status = np.select([r.isna(), ~usable], ["undefined_return", "undefined_or_zero_std"],
                       default="screened")
    return pd.DataFrame({"zscore": ((r - mean) / std).where(usable),
                         "outlier": flag, "screen_status": status}, index=returns.index)


def bounds_violation(df: pd.DataFrame) -> pd.Series:
    """True where raw OHLC breaks ``low <= open, close <= high``."""
    return ((df["high"] < df["low"])
            | (df["open"] < df["low"]) | (df["open"] > df["high"])
            | (df["close"] < df["low"]) | (df["close"] > df["high"]))


def reference_adjustment_factors(reference: pd.DataFrame) -> pd.DataFrame:
    """Date-specific ``adj_close / close`` factors from the trusted ``ohlc`` table.

    Returns ``ticker, ts, ref_close, ref_factor``; rows without a usable
    (finite, positive) close and adj_close are dropped.
    """
    ref = reference[KEY + ["close", "adj_close"]].copy()
    ref["ts"] = pd.to_datetime(ref["ts"], format="%Y-%m-%d", errors="coerce")
    usable = (np.isfinite(ref["close"]) & np.isfinite(ref["adj_close"])
              & (ref["close"] > 0) & (ref["adj_close"] > 0))
    ref = ref.loc[usable]
    return pd.DataFrame({"ticker": ref["ticker"], "ts": ref["ts"], "ref_close": ref["close"],
                         "ref_factor": ref["adj_close"] / ref["close"]}).reset_index(drop=True)


def _expected_adj_close(df: pd.DataFrame, factors: pd.DataFrame) -> pd.DataFrame:
    """Merge reference factors; ``expected_adj`` is set only where raw close matches reference."""
    out = df.merge(factors, on=KEY, how="left", validate="one_to_one")
    close_matches = pd.Series(np.isclose(out["close"], out["ref_close"], rtol=PRICE_RTOL, atol=0),
                              index=out.index)
    out["expected_adj"] = (out["close"] * out["ref_factor"]).where(close_matches)
    return out


def _issue_frame(rows: pd.DataFrame, issue_type: str, severity: str, descriptions) -> pd.DataFrame:
    return pd.DataFrame({
        "ticker": rows["ticker"].to_numpy(),
        "issue_type": issue_type,
        "issue_date": rows["ts"].dt.strftime("%Y-%m-%d").to_numpy(),
        "description": list(descriptions),
        "severity": severity,
    }, columns=REPORT_COLUMNS)


def detect_issues(df: pd.DataFrame, missing_sessions: pd.DataFrame | None = None,
                  factors: pd.DataFrame | None = None, k: float = 3.0) -> pd.DataFrame:
    """Build the data quality report (``ticker | issue_type | issue_date | description | severity``).

    One row per (observation, issue), so a single observation can appear under
    several issue types. Run this on ticker-normalized, exact-deduplicated data
    *before* any repair so the original diagnosis is retained.
    """
    data = df.sort_values(KEY).reset_index(drop=True)
    parts = []

    for col in VALUE_COLS:
        rows = data[data[col].isna()]
        parts.append(_issue_frame(rows, f"missing_{col}", MISSING_SEVERITY[col],
                                  [f"{col} is missing"] * len(rows)))

    for col in PRICE_COLS:
        rows = data[np.isinf(data[col])]
        parts.append(_issue_frame(rows, "nonfinite_price", "high",
                                  [f"{col} = {v}" for v in rows[col]]))
        rows = data[data[col] <= 0]
        parts.append(_issue_frame(rows, "nonpositive_price", "high",
                                  [f"{col} = {v:g} (prices must be > 0)" for v in rows[col]]))

    bound_checks = [
        ("high_below_close", data["high"] < data["close"], ["high", "close"]),
        ("low_above_close", data["low"] > data["close"], ["low", "close"]),
        ("open_outside_range", (data["open"] < data["low"]) | (data["open"] > data["high"]),
         ["low", "open", "high"]),
        ("high_below_low", data["high"] < data["low"], ["low", "high"]),
    ]
    for issue_type, mask, cols in bound_checks:
        rows = data[mask]
        parts.append(_issue_frame(rows, issue_type, "high", [
            ", ".join(f"{c}={v:g}" for c, v in zip(cols, values))
            for values in rows[cols].itertuples(index=False)]))

    rows = data[data["volume"] == 0]
    parts.append(_issue_frame(rows, "zero_volume", "low",
                              ["volume = 0; possibly stale quote, kept for review"] * len(rows)))
    rows = data[data["volume"] < 0]
    parts.append(_issue_frame(rows, "negative_volume", "medium",
                              [f"volume = {v:g}" for v in rows["volume"]]))

    flags = three_sigma_flags(simple_returns(data, "close"), data["ticker"], k)
    rows = data[flags["outlier"]]
    parts.append(_issue_frame(rows, "return_outlier", "low", [
        f"simple return {r:+.2%} is {z:+.1f} sample std from the ticker mean; review flag, not deleted"
        for r, z in zip(simple_returns(data, "close")[flags["outlier"]], flags.loc[flags["outlier"], "zscore"])]))

    if missing_sessions is not None:
        parts.append(_issue_frame(missing_sessions, "missing_session", "medium",
                                  ["date is in the reference calendar but has no row"] * len(missing_sessions)))

    if factors is not None:
        merged = _expected_adj_close(data, factors)
        mask = (merged["adj_close"].notna() & merged["expected_adj"].notna()
                & ~np.isclose(merged["adj_close"], merged["expected_adj"], rtol=PRICE_RTOL))
        rows = merged[mask]
        parts.append(_issue_frame(rows, "adj_close_mismatch", "high", [
            f"vendor adj_close {v:g} vs close x reference factor {e:g}"
            for v, e in zip(rows["adj_close"], rows["expected_adj"])]))

    report = pd.concat(parts, ignore_index=True)
    return report.sort_values(["ticker", "issue_date", "issue_type"], ignore_index=True)


def duplicate_issues(exact_duplicates: pd.DataFrame, conflicts: pd.DataFrame) -> pd.DataFrame:
    """Quality-report rows for removed exact duplicates and quarantined key conflicts."""
    return pd.concat([
        _issue_frame(exact_duplicates, "exact_duplicate", "low",
                     ["identical copy of another record; extra copy removed"] * len(exact_duplicates)),
        _issue_frame(conflicts, "conflicting_duplicate", "high",
                     ["same (ticker, ts) with different values; all copies quarantined"] * len(conflicts)),
    ], ignore_index=True)


# ---------------------------------------------------------------------------
# 4. Repairs
# ---------------------------------------------------------------------------

def _add_flag(df: pd.DataFrame, mask, flag: str) -> None:
    """Append ``flag`` to ``repair_flags`` (semicolon separated) where ``mask`` is True. In place."""
    mask = pd.Series(mask, index=df.index).fillna(False).astype(bool)
    if not mask.any():
        return
    current = df.loc[mask, "repair_flags"]
    df.loc[mask, "repair_flags"] = np.where(current == "", flag, current + ";" + flag)


def drop_all_missing_ohlc(df: pd.DataFrame):
    """Split into (kept, removed) where removed rows have all four raw prices missing."""
    all_missing = df[RAW_PRICES].isna().all(axis=1)
    return df.loc[~all_missing].copy(), df.loc[all_missing].copy()


def repair_adj_close(df: pd.DataFrame, factors: pd.DataFrame) -> pd.DataFrame:
    """Rebuild missing or inconsistent ``adj_close`` as ``close x reference factor``.

    The factor comes from the matching ``(ticker, ts)`` row of the trusted
    ``ohlc`` table, and is used only when this row's raw close equals the
    reference close (so we know which raw price the factor applies to). Factors
    are never carried to other dates. The vendor value is kept in
    ``adj_close_vendor``. Rows that cannot be repaired keep their value and are
    caught by validation.
    """
    out = _expected_adj_close(df, factors)
    out["adj_close_vendor"] = out["adj_close"]
    fixable = out["expected_adj"].notna()
    missing = out["adj_close"].isna() & fixable
    mismatch = (out["adj_close"].notna() & fixable
                & ~np.isclose(out["adj_close"], out["expected_adj"], rtol=PRICE_RTOL))
    out.loc[missing | mismatch, "adj_close"] = out.loc[missing | mismatch, "expected_adj"]
    _add_flag(out, missing, "adj_close_filled_from_reference_factor")
    _add_flag(out, mismatch, "adj_close_replaced_with_reference_factor")
    return out.drop(columns=["ref_close", "ref_factor", "expected_adj"]).set_index(df.index)


def invalid_raw_mask(df: pd.DataFrame) -> pd.Series:
    """True where any raw price is missing, non-finite or nonpositive, or OHLC bounds fail."""
    prices = df[RAW_PRICES]
    bad_cell = (~np.isfinite(prices) | (prices <= 0)).any(axis=1)
    return bad_cell | bounds_violation(df)


def repair_invalid_raw_rows(df: pd.DataFrame):
    """Forward-fill invalid raw rows *as a block* from the previous valid row of the same ticker.

    Filling one bad cell (e.g. close) from yesterday can violate today's
    high/low, so the whole price row (OHLC and adj_close together) is carried
    forward; the result always satisfies the OHLC bounds and keeps raw and
    adjusted prices consistent. Volume is left as reported.

    Returns ``(repaired, unresolved_mask)``; unresolved rows have no earlier
    valid row (a leading gap) and must be quarantined, never back-filled.
    """
    out = df.sort_values(KEY).copy()
    bad = invalid_raw_mask(out)
    out.loc[bad, PRICE_COLS] = np.nan
    filled = out.groupby("ticker")[PRICE_COLS].ffill()
    resolved = bad & filled[RAW_PRICES].notna().all(axis=1)
    out.loc[resolved, PRICE_COLS] = filled.loc[resolved]
    _add_flag(out, resolved, "ohlc_forward_filled_from_previous_row")
    # Restore the original values on unresolved rows so the quarantine shows what was received.
    unresolved = bad & ~resolved
    out.loc[unresolved, PRICE_COLS] = df.loc[unresolved[unresolved].index, PRICE_COLS]
    return out, unresolved


def interpolate_confirmed_errors(df: pd.DataFrame, error_mask: pd.Series):
    """Linearly interpolate prices on rows confirmed to be data errors.

    Interpolation happens within each sorted ticker, only between valid
    neighbours (``limit_area='inside'``): leading/trailing errors are never
    extrapolated and are returned as unresolved. Because each interpolated
    price is a weighted average of two valid rows, OHLC bounds still hold.

    Returns ``(repaired, unresolved_mask)``.
    """
    out = df.sort_values(KEY).copy()
    mask = error_mask.reindex(out.index, fill_value=False)
    out.loc[mask, PRICE_COLS] = np.nan
    interp = out.groupby("ticker")[PRICE_COLS].transform(lambda s: s.interpolate(limit_area="inside"))
    resolved = mask & interp.notna().all(axis=1)
    out.loc[resolved, PRICE_COLS] = interp.loc[resolved]
    _add_flag(out, resolved, "prices_interpolated_confirmed_error")
    unresolved = mask & ~resolved
    out.loc[unresolved, PRICE_COLS] = df.loc[unresolved[unresolved].index, PRICE_COLS]
    return out, unresolved


def repair_volume(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing/invalid volume with 0 (flagged) and mark reported zero-volume days.

    A filled 0 means *unknown*, not "no trading", which is why it gets a repair
    flag distinct from ``zero_volume_flag`` (a reported 0, possibly stale data).
    """
    out = df.copy()
    volume = out["volume"].astype(float)
    missing = volume.isna()
    invalid = ~missing & (~np.isfinite(volume) | (volume < 0))
    out["zero_volume_flag"] = volume.eq(0)
    out["volume"] = volume.mask(missing | invalid, 0.0).astype("int64")
    _add_flag(out, missing, "volume_missing_filled_0")
    _add_flag(out, invalid, "volume_invalid_set_0")
    return out


def add_return_outlier_flags(df: pd.DataFrame, factors: pd.DataFrame | None = None,
                             k: float = 3.0) -> pd.DataFrame:
    """Add ``return_outlier`` (raw-close 3-sigma screen) and the evidence about each flag.

    A flagged return depends on two closes (today and the previous row). With
    reference ``factors`` available, ``return_outlier_class`` records whether
    both closes agree with the trusted ``ohlc`` table:

    * ``matches_reference`` - both closes match, so the vendor did not mistype
      them. This does not prove the price is real: the reference can contain
      the same bad print. Kept and flagged for review.
    * ``differs_from_reference`` - a close is missing from or disagrees with the
      reference: a candidate vendor error for manual review.
    """
    out = df.sort_values(KEY).reset_index(drop=True)
    flags = three_sigma_flags(simple_returns(out, "close"), out["ticker"], k)
    out["return_outlier"] = flags["outlier"]
    if factors is None:
        matches = pd.Series(False, index=out.index)
    else:
        ref_close = out[KEY].merge(factors, on=KEY, how="left")["ref_close"]
        matches = pd.Series(np.isclose(out["close"], ref_close, rtol=PRICE_RTOL), index=out.index)
    both_match = matches & matches.groupby(out["ticker"]).shift(1, fill_value=False)
    out["return_outlier_class"] = np.select(
        [~flags["outlier"], both_match], ["", "matches_reference"], default="differs_from_reference")
    return out


# ---------------------------------------------------------------------------
# 5. Validation and the full pipeline
# ---------------------------------------------------------------------------

def _row_checks(df: pd.DataFrame) -> dict[str, pd.Series]:
    prices = df[PRICE_COLS]
    volume = df["volume"].astype(float)
    return {
        "missing ticker": df["ticker"].isna() | df["ticker"].eq(""),
        "missing or unparseable ts": df["ts"].isna(),
        "missing price": prices.isna().any(axis=1),
        "non-finite price": np.isinf(prices).any(axis=1),
        "nonpositive price": (prices <= 0).any(axis=1),
        "OHLC bounds violated": bounds_violation(df),
        "missing, non-finite or negative volume": ~np.isfinite(volume) | (volume < 0),
    }


def row_failure_reasons(df: pd.DataFrame) -> pd.Series:
    """Semicolon-joined failed checks per row ('' when the row is valid)."""
    reasons = pd.Series("", index=df.index, dtype=object)
    for label, mask in _row_checks(df).items():
        mask = mask.fillna(False).astype(bool)
        reasons[mask] = np.where(reasons[mask] == "", label, reasons[mask] + "; " + label)
    return reasons


def validate_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Table of every required check with its failure count and pass/fail."""
    rows = [(label, int(mask.fillna(False).sum())) for label, mask in _row_checks(df).items()]
    rows.append(("duplicate (ticker, ts) keys", int(df.duplicated(KEY).sum())))
    report = pd.DataFrame(rows, columns=["check", "failures"])
    report["passed"] = report["failures"].eq(0)
    return report


def assert_valid(df: pd.DataFrame) -> None:
    """Raise ``ValueError`` naming every failed check."""
    report = validate_clean(df)
    failed = report[~report["passed"]]
    if not failed.empty:
        raise ValueError("Validation failed: " + "; ".join(
            f"{c} ({n})" for c, n in zip(failed["check"], failed["failures"])))


def _quarantine(rows: pd.DataFrame, reason) -> pd.DataFrame:
    return rows.assign(quarantine_reason=reason)


@dataclass
class CleaningResult:
    """Everything produced by :func:`clean_ohlc`, kept for the audit trail."""
    clean: pd.DataFrame
    quarantine: pd.DataFrame
    exact_duplicates: pd.DataFrame
    conflicts: pd.DataFrame
    dropped_all_missing: pd.DataFrame
    ticker_audit: pd.DataFrame
    raw_rows: int

    def row_accounting(self) -> pd.DataFrame:
        """Where every raw row went; the parts must add up to the raw count."""
        table = pd.DataFrame({"rows": {
            "raw input": self.raw_rows,
            "exact duplicates removed": len(self.exact_duplicates),
            "all-missing OHLC removed": len(self.dropped_all_missing),
            "quarantined": len(self.quarantine),
            "retained in ohlc_hw_cleaned": len(self.clean),
        }})
        accounted = table["rows"].iloc[1:].sum()
        if accounted != self.raw_rows:
            raise AssertionError(f"Row accounting mismatch: {accounted} != {self.raw_rows}")
        return table


def clean_ohlc(raw: pd.DataFrame, reference: pd.DataFrame, k: float = 3.0) -> CleaningResult:
    """Run the full cleaning pipeline on raw ``ohlc_hw`` rows.

    ``reference`` is the supplied ``ohlc`` table, used only for date-specific
    adjustment factors (never to replace the dirty data wholesale).
    """
    prepared = prepare_raw(raw)
    unique, exact, conflicts = split_duplicates(prepared)
    kept, all_missing = drop_all_missing_ohlc(unique)

    factors = reference_adjustment_factors(reference)
    kept = repair_adj_close(kept, factors)
    kept, unresolved = repair_invalid_raw_rows(kept)
    quarantine = [
        _quarantine(conflicts, "conflicting values for the same (ticker, ts); no basis to choose"),
        _quarantine(kept[unresolved], "invalid raw OHLC with no earlier valid row to carry forward"),
    ]
    kept = repair_volume(kept[~unresolved])

    reasons = row_failure_reasons(kept)
    failed = reasons != ""
    quarantine.append(_quarantine(kept[failed], reasons[failed]))
    clean = add_return_outlier_flags(kept[~failed], factors, k)
    assert_valid(clean)

    quarantine = pd.concat(quarantine, ignore_index=True).sort_values(KEY, ignore_index=True)
    return CleaningResult(clean=clean, quarantine=quarantine, exact_duplicates=exact,
                          conflicts=conflicts, dropped_all_missing=all_missing,
                          ticker_audit=ticker_audit(prepared), raw_rows=len(raw))


def to_sql_frame(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """Copy ready for ``to_sql``: ISO date strings, booleans as 0/1, chosen column order."""
    out = df[columns].copy() if columns else df.copy()
    for col in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[col]):
            out[col] = out[col].dt.strftime("%Y-%m-%d")
        elif pd.api.types.is_bool_dtype(out[col]):
            out[col] = out[col].astype(int)
    return out
