"""Detection, cleaning and validation for HW2 Part 1 (``ohlc_hw`` -> ``ohlc_hw_cleaned``).

Every function works on pandas DataFrames with the course column names
(``ts``, ``ticker``, ``open``, ``high``, ``low``, ``close``, ``adj_close``,
``volume``).  Database access is isolated in a few helpers: reads use a
read-only connection and writes refuse to touch the supplied source tables.
"""

from __future__ import annotations

import hashlib
import html
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

KEY = ["ticker", "ts"]
RAW_PRICES = ["open", "high", "low", "close"]
PRICES = RAW_PRICES + ["adj_close"]
VALUE_COLUMNS = PRICES + ["volume"]
ISSUE_COLUMNS = ["ticker", "issue_type", "issue_date", "description", "severity"]
PROTECTED_TABLES = frozenset({"ohlc_hw", "ohlc", "market_caps"})

OUTLIER_SIGMA = 3.0
SPIKE_MIN_MOVE = 0.40        # a confirmed spike moves >40% away from BOTH neighbours
SPIKE_NEIGHBOR_TOL = 0.05    # ...while the two neighbours agree within 5%
FACTOR_WINDOW = 11           # centred rolling median of adj_close / close
FACTOR_TOL = 0.10            # factor >10% away from that median is anomalous
NEIGHBOR_FACTOR_TOL = 0.005  # neighbour factors must agree within 0.5%
REFERENCE_RTOL = 1e-6

SEVERITY = {
    "nonpositive_price": "high",
    "nonfinite_price": "high",
    "missing_raw_price": "high",
    "high_below_close": "high",
    "low_above_close": "high",
    "open_outside_range": "high",
    "high_below_low": "high",
    "conflicting_duplicate": "high",
    "isolated_price_spike": "high",
    "adj_factor_anomaly": "high",
    "adj_close_reference_mismatch": "high",
    "invalid_volume": "high",
    "missing_adj_close": "medium",
    "missing_volume": "medium",
    "missing_trading_date": "medium",
    "return_outlier_3sd": "medium",
    "weekend_date": "medium",
    "exact_duplicate": "low",
    "ticker_markup": "low",
    "zero_volume": "low",
}


# --------------------------------------------------------------------------- I/O

def connect_readonly(db_path: str | Path) -> sqlite3.Connection:
    """Open a SQLite database that cannot be written through this connection."""
    path = Path(db_path).resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def load_table(db_path: str | Path, table: str, with_rowid: bool = False) -> pd.DataFrame:
    """Read a whole table; ``with_rowid`` keeps SQLite's rowid as ``source_rowid``."""
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", table):
        raise ValueError(f"unsafe table name: {table!r}")
    columns = "rowid AS source_rowid, *" if with_rowid else "*"
    with connect_readonly(db_path) as con:
        return pd.read_sql_query(f"SELECT {columns} FROM {table}", con)


def table_counts(db_path: str | Path) -> pd.Series:
    """Row count of every table in the database."""
    with connect_readonly(db_path) as con:
        names = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
        return pd.Series({n: con.execute(f'SELECT COUNT(*) FROM "{n}"').fetchone()[0] for n in names},
                         name="rows")


def table_fingerprint(db_path: str | Path, table: str) -> str:
    """SHA-256 over a table's rows in rowid order, used to prove a table was not modified."""
    frame = load_table(db_path, table, with_rowid=True)
    hashed = pd.util.hash_pandas_object(frame, index=False).to_numpy()
    return hashlib.sha256(hashed.tobytes()).hexdigest()


def write_tables(db_path: str | Path, tables: dict[str, pd.DataFrame]) -> None:
    """Replace (or create) output tables. Source tables are refused outright."""
    blocked = PROTECTED_TABLES.intersection(tables)
    if blocked:
        raise PermissionError(f"refusing to overwrite source table(s): {sorted(blocked)}")
    with sqlite3.connect(Path(db_path).resolve()) as con:
        for name, frame in tables.items():
            frame.to_sql(name, con, if_exists="replace", index=False)


# ----------------------------------------------------------------- normalization

_TAG = re.compile(r"<[^>]*>")


def normalize_ticker(label: object) -> object:
    """Strip HTML markup/entities and surrounding whitespace, then upper-case.

    Returns ``pd.NA`` for missing or empty labels so they can be quarantined.
    """
    if label is None or (isinstance(label, float) and np.isnan(label)):
        return pd.NA
    text = html.unescape(_TAG.sub("", str(label))).strip().upper()
    return text or pd.NA


def normalize_tickers(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add a normalized ``ticker`` (original kept as ``ticker_raw``) and an audit table."""
    out = frame.rename(columns={"ticker": "ticker_raw"})
    out["ticker"] = out["ticker_raw"].map(normalize_ticker).astype("string")

    audit = (out.groupby(["ticker_raw", "ticker"], dropna=False).size()
             .rename("n_rows").reset_index())
    raw_text = audit["ticker_raw"].astype("string").fillna("")
    stripped = raw_text.map(lambda s: html.unescape(_TAG.sub("", s)))
    audit["markup_removed"] = raw_text != stripped
    audit["whitespace_stripped"] = stripped != stripped.str.strip()
    audit["case_changed"] = stripped.str.strip() != stripped.str.strip().str.upper()
    audit = audit.rename(columns={"ticker": "normalized_ticker"})
    return out, audit.sort_values(["normalized_ticker", "ticker_raw"]).reset_index(drop=True)


def to_iso_date(values: pd.Series) -> pd.Series:
    """Parse dates and render ``YYYY-MM-DD``; unparseable values become missing."""
    parsed = pd.to_datetime(values, errors="coerce")
    return parsed.dt.strftime("%Y-%m-%d").astype("string")


def split_exact_duplicates(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split rows into (kept, removed) where removed rows repeat a kept row exactly.

    Rows are compared on the normalized key and every value column, so the
    same observation delivered under different ticker markup counts as a copy.
    ``kept`` gains ``duplicate_copies_removed``.

    Each copy is matched to its kept row through a group id rather than a join
    on the values, because a join never matches missing values (NaN != NaN).
    If ``source_rowid`` is present, ``removed`` gains ``duplicate_of_source_rowid``.
    """
    subset = KEY + VALUE_COLUMNS
    group = frame.groupby(subset, dropna=False, sort=False).ngroup()
    is_copy = group.duplicated(keep="first")
    group_size = group.map(group.value_counts())
    kept, removed = frame.loc[~is_copy].copy(), frame.loc[is_copy].copy()
    kept["duplicate_copies_removed"] = (group_size[~is_copy] - 1).astype(int)
    if "source_rowid" in frame:
        first_rowid = frame["source_rowid"].groupby(group).transform("first")
        removed["duplicate_of_source_rowid"] = first_rowid[is_copy]
    return kept, removed


def split_conflicting_keys(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split into (unique-key rows, rows whose ``(ticker, ts)`` key has differing values).

    Call after exact duplicates are removed; every remaining repeat is a conflict
    and none of its rows is preferred over the others.
    """
    conflict = frame.duplicated(subset=KEY, keep=False)
    return frame.loc[~conflict].copy(), frame.loc[conflict].copy()


# -------------------------------------------------------------------- detection

def _fmt(values: pd.Series, spec: str) -> pd.Series:
    """Format numbers as strings; always returns a string Series, even when empty."""
    return pd.Series([spec.format(v) for v in values], index=values.index, dtype="string")


def _issues(frame: pd.DataFrame, mask: pd.Series, issue_type: str, description) -> pd.DataFrame:
    rows = frame.loc[mask, ["ticker", "ts"]].rename(columns={"ts": "issue_date"})
    rows["issue_type"] = issue_type
    rows["description"] = description[mask] if isinstance(description, pd.Series) else description
    rows["severity"] = SEVERITY[issue_type]
    return rows[ISSUE_COLUMNS]


def _empty_issues() -> pd.DataFrame:
    return pd.DataFrame(columns=ISSUE_COLUMNS)


def _concat_issues(parts: list[pd.DataFrame]) -> pd.DataFrame:
    parts = [p for p in parts if not p.empty]
    if not parts:
        return _empty_issues()
    return pd.concat(parts, ignore_index=True)[ISSUE_COLUMNS]


def _sorted(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values(KEY, kind="mergesort")


def missing_value_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Missing-cell count and share for every value column."""
    cols = [c for c in VALUE_COLUMNS if c in frame]
    counts = frame[cols].isna().sum()
    return pd.DataFrame({"missing": counts, "share": counts / max(len(frame), 1)})


def missing_by_ticker(frame: pd.DataFrame) -> pd.DataFrame:
    """Missing-cell counts per ticker (tickers with no missing cells omitted)."""
    counts = frame[VALUE_COLUMNS].isna().groupby(frame["ticker"]).sum()
    return counts.loc[counts.sum(axis=1) > 0]


def detect_missing_values(frame: pd.DataFrame) -> pd.DataFrame:
    """One issue per missing cell; raw prices, adjusted close and volume are graded separately."""
    parts = []
    for col in RAW_PRICES:
        parts.append(_issues(frame, frame[col].isna(), "missing_raw_price", f"{col} is missing"))
    parts.append(_issues(frame, frame["adj_close"].isna(), "missing_adj_close", "adj_close is missing"))
    parts.append(_issues(frame, frame["volume"].isna(), "missing_volume", "volume is missing"))
    return _concat_issues(parts)


ABSENT_FROM_SOURCE = "absent_from_source"
WITHHELD_BY_CLEANING = "withheld_by_cleaning"


def find_missing_dates(frame: pd.DataFrame, calendar: pd.DataFrame,
                       withheld: pd.DataFrame | None = None) -> pd.DataFrame:
    """Expected ``(ticker, ts)`` keys from ``calendar`` that ``frame`` does not contain.

    ``calendar`` supplies each ticker's own expected dates (here, the reference
    ``ohlc`` table). No prices are invented for absent dates.

    ``withheld`` (columns ``ticker, ts, reason``) lists observations that were
    delivered but taken out by cleaning. A missing date found there gets
    ``cause = withheld_by_cleaning`` and its reason; every other missing date
    is ``absent_from_source``.
    """
    expected = calendar[KEY].drop_duplicates()
    expected = expected[expected["ticker"].isin(frame["ticker"].unique())]
    merged = expected.merge(frame[KEY].drop_duplicates(), on=KEY, how="left", indicator=True)
    missing = merged.loc[merged["_merge"] == "left_only", KEY]
    if withheld is not None and not withheld.empty:
        reasons = (withheld[KEY + ["reason"]].groupby(KEY)["reason"]
                   .agg(lambda r: "; ".join(sorted(set(r)))).rename("withheld_reason").reset_index())
        missing = missing.merge(reasons, on=KEY, how="left")
    else:
        missing = missing.assign(withheld_reason=pd.NA)
    missing["cause"] = np.where(missing["withheld_reason"].notna(), WITHHELD_BY_CLEANING, ABSENT_FROM_SOURCE)
    return _sorted(missing[KEY + ["cause", "withheld_reason"]]).reset_index(drop=True)


def missing_dates_summary(frame: pd.DataFrame, calendar: pd.DataFrame,
                          withheld: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per-ticker count of missing trading days over that ticker's reference history.

    Missing days are split into ``absent_from_source`` and ``withheld_by_cleaning``;
    the message names the split only when ``withheld`` is supplied.
    """
    missing = find_missing_dates(frame, calendar, withheld)
    span = calendar.groupby("ticker")["ts"].agg(start="min", end="max")
    out = pd.DataFrame(index=pd.Index(sorted(frame["ticker"].dropna().unique()), name="ticker"))
    out = out.join(span)
    by_cause = missing.groupby(["ticker", "cause"]).size().unstack(fill_value=0)
    for cause in [ABSENT_FROM_SOURCE, WITHHELD_BY_CLEANING]:
        out[cause] = by_cause.get(cause, pd.Series(dtype=int)).reindex(out.index).fillna(0).astype(int)
    out["missing_days"] = out[ABSENT_FROM_SOURCE] + out[WITHHELD_BY_CLEANING]
    messages = []
    for t, r in out.iterrows():
        if pd.isna(r.start):
            messages.append(f"Ticker {t} has no reference calendar; gaps cannot be assessed")
            continue
        text = f"Ticker {t} is missing {r.missing_days} trading days between {r.start} and {r.end}"
        if withheld is not None:
            text += (f" ({r[ABSENT_FROM_SOURCE]} absent from source, "
                     f"{r[WITHHELD_BY_CLEANING]} withheld by cleaning)")
        messages.append(text)
    out["message"] = messages
    return out.reset_index()[["ticker", "start", "end", "missing_days", ABSENT_FROM_SOURCE,
                              WITHHELD_BY_CLEANING, "message"]]


def detect_missing_dates(frame: pd.DataFrame, calendar: pd.DataFrame,
                         withheld: pd.DataFrame | None = None) -> pd.DataFrame:
    """Each absent reference date becomes its own issue row, stating why it is absent."""
    missing = find_missing_dates(frame, calendar, withheld)
    desc = pd.Series(
        np.where(missing["cause"] == WITHHELD_BY_CLEANING,
                 "delivered observation withheld by cleaning ("
                 + missing["withheld_reason"].astype("string").fillna("") + "); not a source gap",
                 "date in the ticker's reference calendar is absent from the source delivery"),
        index=missing.index)
    return _issues(missing, pd.Series(True, index=missing.index), "missing_trading_date", desc)


def simple_returns(frame: pd.DataFrame) -> pd.Series:
    """``(close - prev_close) / prev_close`` within each ticker, ordered by date.

    Returns are undefined (NaN) for a ticker's first row and wherever either
    close is missing, nonpositive or non-finite.
    """
    data = _sorted(frame)
    close = data["close"].where(np.isfinite(data["close"]) & (data["close"] > 0))
    ret = close.groupby(data["ticker"], sort=False).pct_change(fill_method=None)
    return ret.reindex(frame.index)


def ticker_return_stats(frame: pd.DataFrame) -> pd.DataFrame:
    """Mean, sample std (ddof=1) and count of defined simple returns per ticker.

    ``status`` makes the degenerate cases explicit: fewer than two returns or a
    zero standard deviation means no observation can be a 3-sigma outlier.
    """
    ret = simple_returns(frame)
    stats = ret.groupby(frame["ticker"]).agg(n_returns="count", mean="mean", std=lambda s: s.std(ddof=1))
    stats["status"] = np.select(
        [stats["n_returns"] < 2, ~np.isfinite(stats["std"]) | (stats["std"] == 0)],
        ["insufficient_returns", "zero_std"], default="ok")
    return stats


def flag_return_outliers(frame: pd.DataFrame, n_sigma: float = OUTLIER_SIGMA) -> pd.DataFrame:
    """Per-row return, ticker mean/std and the two-tailed ``|r - mean| > n * std`` flag."""
    ret = simple_returns(frame)
    stats = ticker_return_stats(frame)
    out = pd.DataFrame({"ticker": frame["ticker"], "ts": frame["ts"], "return": ret})
    out = out.join(stats[["mean", "std", "status"]], on="ticker")
    usable = out["return"].notna() & (out["status"] == "ok")
    out["is_outlier"] = usable & ((out["return"] - out["mean"]).abs() > n_sigma * out["std"])
    return out


def detect_return_outliers(frame: pd.DataFrame, n_sigma: float = OUTLIER_SIGMA) -> pd.DataFrame:
    """3-sigma review flags; flagged returns are not deleted or repaired by this check."""
    flags = flag_return_outliers(frame, n_sigma)
    z = (flags["return"] - flags["mean"]) / flags["std"]
    desc = ("simple return " + _fmt(flags["return"], "{:+.2%}")
            + " vs ticker mean " + _fmt(flags["mean"], "{:+.3%}")
            + " (" + _fmt(z, "{:+.1f}") + " sample std)")
    return _issues(flags, flags["is_outlier"], "return_outlier_3sd", desc)


REVERSAL_RETAINED = 0.25     # next close keeps at most 25% of the move -> one-day reversal
PERSISTENT_RETAINED = 0.50   # close 5 days later keeps at least half of the move
LOW_VOLUME_RATIO = 0.05      # volume at most 5% of the trailing 20-day median
NORMAL_VOLUME_RATIO = 0.50
CLUSTER_MIN_TICKERS = 5      # this many tickers flagged on the same date
LARGE_MOVE = 0.40
ADJ_DIVERGENCE = 0.01        # adjusted return differs from raw return by > 1 percentage point

OUTLIER_ASSESSMENT = {
    "low_volume_one_day_reversal": "likely_data_error",
    "reversal_of_likely_error": "likely_data_error",
    "large_level_shift": "unresolved",
    "adjustment_factor_change": "unresolved",
    "market_wide_move": "plausible_market_event",
    "persistent_move": "plausible_market_event",
    "held_next_day": "plausible_market_event",
    "one_day_reversal": "unresolved",
    "other": "unresolved",
}


def investigate_return_outliers(frame: pd.DataFrame) -> pd.DataFrame:
    """Describe and classify every 3-sigma flag using only evidence in the data.

    For each flagged return the neighbouring prices, volume, same-date flags
    across tickers and the vendor adjustment factor are measured, then one
    category is assigned (first match wins, in the order of ``OUTLIER_ASSESSMENT``):

    * ``low_volume_one_day_reversal``: the next close gives back at least 75% of the
      move and the day's volume is at most 5% of its trailing 20-day median.
    * ``reversal_of_likely_error``: the flagged return back from such a day.
    * ``large_level_shift``: a move above 40% that persists for 5 days.
    * ``adjustment_factor_change``: the adjusted return differs from the raw return
      by more than 1 percentage point, i.e. the vendor factor changed that day.
    * ``market_wide_move``: at least 5 tickers are flagged on the same date.
    * ``persistent_move``: half the move remains after 5 days on at least half of
      normal volume.
    * ``held_next_day``: half the move remains at the next close on at least half
      of normal volume, but it fades within 5 days.
    * ``one_day_reversal``: reversal on normal volume.
    * ``other``: everything else (partial retracements, short histories).

    ``flat_bar`` (open = high = low = close) is reported as a diagnostic but is
    deliberately not a rule. Before 1990 many vendors recorded only a close, so
    flat bars are the normal format of old data (about 18% of 1960s and 24% of
    1970s rows here, none after 1990) and are no more common among flagged
    returns than among all rows. A flat bar says how a day was recorded, not
    whether its price is wrong; the evidence that matters is the reversal and
    the missing volume, which the first rule already requires.

    This is a review aid; nothing is repaired here. The causes of moves are not
    inferred beyond what these measurements show.
    """
    data = _sorted(frame).copy()
    g = data.groupby("ticker", sort=False)
    flags = flag_return_outliers(data)
    data["return"], data["is_outlier"] = flags["return"], flags["is_outlier"]
    data["prev_close"] = g["close"].shift(1)
    data["next_close"] = g["close"].shift(-1)
    data["close_5d"] = g["close"].shift(-5)
    move = np.log(data["close"] / data["prev_close"])
    data["retained_1d"] = np.log(data["next_close"] / data["prev_close"]) / move
    data["retained_5d"] = np.log(data["close_5d"] / data["prev_close"]) / move
    trailing = g["volume"].transform(lambda s: s.shift(1).rolling(20, min_periods=5).median())
    data["volume_ratio"] = data["volume"] / trailing.where(trailing > 0)
    data["flat_bar"] = data[RAW_PRICES].nunique(axis=1) == 1
    data["adj_return"] = g["adj_close"].pct_change(fill_method=None)
    factor = data["adj_close"] / data["close"]
    data["factor_change"] = factor / factor.groupby(data["ticker"], sort=False).shift(1) - 1
    data["date_flag_count"] = data.groupby("ts")["is_outlier"].transform("sum").astype(int)

    reversal = data["retained_1d"].abs() <= REVERSAL_RETAINED
    low_volume = (data["volume"] == 0) | (data["volume_ratio"] <= LOW_VOLUME_RATIO)
    suspect = data["is_outlier"] & reversal & low_volume
    after_suspect = suspect.groupby(data["ticker"], sort=False).shift(1, fill_value=False)
    persistent = data["retained_5d"] >= PERSISTENT_RETAINED
    rules = [
        ("low_volume_one_day_reversal", suspect),
        ("reversal_of_likely_error", after_suspect),
        ("large_level_shift", (data["return"].abs() > LARGE_MOVE) & persistent),
        ("adjustment_factor_change", (data["adj_return"] - data["return"]).abs() > ADJ_DIVERGENCE),
        ("market_wide_move", data["date_flag_count"] >= CLUSTER_MIN_TICKERS),
        ("persistent_move", persistent & (data["volume_ratio"] >= NORMAL_VOLUME_RATIO)),
        ("held_next_day", (data["retained_1d"] >= PERSISTENT_RETAINED)
         & (data["volume_ratio"] >= NORMAL_VOLUME_RATIO)),
        ("one_day_reversal", reversal),
    ]
    data["category"] = np.select([mask.fillna(False).to_numpy(bool) for _, mask in rules],
                                 [name for name, _ in rules], default="other")
    out = data.loc[data["is_outlier"]].copy()
    out["assessment"] = out["category"].map(OUTLIER_ASSESSMENT)
    out["evidence"] = (
        "next close keeps " + _fmt(out["retained_1d"], "{:.0%}") + " of the move, 5 days later "
        + _fmt(out["retained_5d"], "{:.0%}") + "; volume " + _fmt(out["volume_ratio"], "{:.1%}")
        + " of 20-day median; " + out["date_flag_count"].astype("string") + " tickers flagged that date; "
        + "adjustment factor change " + _fmt(out["factor_change"], "{:+.1%}"))
    columns = ["ticker", "ts", "return", "prev_close", "open", "high", "low", "close", "next_close",
               "close_5d", "retained_1d", "retained_5d", "volume", "volume_ratio", "flat_bar",
               "date_flag_count", "adj_return", "factor_change", "category", "assessment", "evidence"]
    return out[columns].reset_index(drop=True)


def detect_invalid_prices(frame: pd.DataFrame) -> pd.DataFrame:
    """Nonpositive or infinite raw prices and adjusted close (missing cells are a separate check)."""
    parts = []
    for col in PRICES:
        values = frame[col]
        parts.append(_issues(frame, values.notna() & np.isinf(values), "nonfinite_price",
                             f"{col} is not finite"))
        parts.append(_issues(frame, values <= 0, "nonpositive_price",
                             col + " = " + values.astype("string") + " (must be > 0)"))
    return _concat_issues(parts)


def detect_ohlc_inconsistencies(frame: pd.DataFrame) -> pd.DataFrame:
    """Impossible raw OHLC relationships. Adjusted close is not bound by raw high/low."""
    f = frame
    checks = [
        ("high_below_close", f["high"] < f["close"], "high < close"),
        ("low_above_close", f["low"] > f["close"], "low > close"),
        ("open_outside_range", (f["low"] > f["open"]) | (f["high"] < f["open"]),
         "open outside [low, high]"),
        ("high_below_low", f["high"] < f["low"], "high < low"),
    ]
    detail = ("O=" + f["open"].astype("string") + " H=" + f["high"].astype("string")
              + " L=" + f["low"].astype("string") + " C=" + f["close"].astype("string"))
    return _concat_issues([_issues(f, mask, name, text + ": " + detail) for name, mask, text in checks])


def detect_volume_issues(frame: pd.DataFrame) -> pd.DataFrame:
    """Zero volume (possible stale data) and negative/infinite volume."""
    vol = frame["volume"]
    invalid = vol.notna() & (np.isinf(vol) | (vol < 0))
    return _concat_issues([
        _issues(frame, vol == 0, "zero_volume", "volume = 0; possibly stale or untraded"),
        _issues(frame, invalid, "invalid_volume", "volume = " + vol.astype("string")),
    ])


def detect_weekend_dates(frame: pd.DataFrame) -> pd.DataFrame:
    """Observations dated on a Saturday or Sunday, when US exchanges are closed."""
    day = pd.to_datetime(frame["ts"]).dt.day_name()
    return _issues(frame, day.isin(["Saturday", "Sunday"]), "weekend_date",
                   "observation dated on a " + day + "; date retained as delivered")


def find_isolated_price_spikes(frame: pd.DataFrame, min_move: float = SPIKE_MIN_MOVE,
                               neighbor_tol: float = SPIKE_NEIGHBOR_TOL) -> pd.DataFrame:
    """Rows whose close is a confirmed one-day error, with the evidence used.

    All of these must hold:
    * the previous and next rows of the same ticker exist, have valid OHLC
      (positive, finite, within bounds) and positive volume;
    * close moves more than ``min_move`` away from BOTH neighbours in the same
      direction, while the neighbours agree within ``neighbor_tol``;
    * the row itself reports zero volume.
    """
    data = _sorted(frame)
    g = data.groupby("ticker", sort=False)
    prices = data[RAW_PRICES]
    valid = (np.isfinite(prices) & (prices > 0)).all(axis=1) & (
        (data["low"] <= data[["open", "close"]].min(axis=1))
        & (data["high"] >= data[["open", "close"]].max(axis=1)))
    neighbor_ok = valid & (data["volume"] > 0)

    prev_close, next_close = g["close"].shift(1), g["close"].shift(-1)
    vs_prev, vs_next = data["close"] / prev_close - 1, data["close"] / next_close - 1
    neighbors_agree = (next_close / prev_close - 1).abs() <= neighbor_tol
    spike = (
        g["ts"].shift(1).notna() & g["ts"].shift(-1).notna()
        & neighbor_ok.groupby(data["ticker"], sort=False).shift(1, fill_value=False)
        & neighbor_ok.groupby(data["ticker"], sort=False).shift(-1, fill_value=False)
        & (np.sign(vs_prev) == np.sign(vs_next))
        & (vs_prev.abs() > min_move) & (vs_next.abs() > min_move)
        & neighbors_agree & (data["volume"] == 0)
    )
    out = data.loc[spike, KEY + RAW_PRICES + ["volume"]].copy()
    out["prev_ts"], out["next_ts"] = g["ts"].shift(1)[spike], g["ts"].shift(-1)[spike]
    out["prev_close"], out["next_close"] = prev_close[spike], next_close[spike]
    out["move_vs_prev"], out["move_vs_next"] = vs_prev[spike], vs_next[spike]
    return out


def adj_factor_deviation(frame: pd.DataFrame, adj_col: str = "adj_close",
                         window: int = FACTOR_WINDOW) -> tuple[pd.Series, pd.Series]:
    """Return ``(factor, deviation)``, with factor = adj_close / close.

    Deviation is the factor relative to the ticker's centred rolling median.
    Dividend adjustments move the factor by fractions of a percent; a
    multiple-x jump that reverts is a data error, not a corporate action.
    """
    data = _sorted(frame)
    factor = data[adj_col] / data["close"]
    factor = factor.where(np.isfinite(factor) & (factor > 0))
    median = factor.groupby(data["ticker"], sort=False).transform(
        lambda s: s.rolling(window, center=True, min_periods=3).median())
    return factor.reindex(frame.index), (factor / median - 1).reindex(frame.index)


def detect_adj_close_anomalies(frame: pd.DataFrame, reference: pd.DataFrame | None = None,
                               tol: float = FACTOR_TOL) -> pd.DataFrame:
    """Adjustment-factor spikes, plus disagreement with the reference when one is given."""
    factor, dev = adj_factor_deviation(frame)
    anomalous = dev.abs() > tol
    parts = [_issues(frame, anomalous, "adj_factor_anomaly",
                     "adj_close/close = " + _fmt(factor, "{:.5f}") + ", "
                     + _fmt(dev, "{:+.0%}") + " vs the ticker's rolling median factor")]
    if reference is not None:
        ref = frame[KEY].merge(reference[KEY + ["adj_close"]].rename(columns={"adj_close": "ref_adj"}),
                               on=KEY, how="left").set_index(frame.index)["ref_adj"]
        differs = (frame["adj_close"].notna() & ref.notna()
                   & ~np.isclose(frame["adj_close"], ref, rtol=REFERENCE_RTOL, atol=0))
        parts.append(_issues(frame, differs, "adj_close_reference_mismatch",
                             "adj_close " + _fmt(frame["adj_close"], "{:.6g}")
                             + " vs reference " + _fmt(ref, "{:.6g}")
                             + " (x" + (frame["adj_close"] / ref).map("{:.2f}".format) + ")"))
    return _concat_issues(parts)


def detect_price_spikes(frame: pd.DataFrame) -> pd.DataFrame:
    """Issue rows for :func:`find_isolated_price_spikes`."""
    spikes = find_isolated_price_spikes(frame)
    desc = ("close " + spikes["close"].astype("string") + " vs neighbours "
            + spikes["prev_close"].astype("string") + " / " + spikes["next_close"].astype("string")
            + " with zero volume; one-day reversal")
    return _issues(spikes, pd.Series(True, index=spikes.index), "isolated_price_spike", desc)


def label_and_duplicate_issues(normalized: pd.DataFrame, removed: pd.DataFrame,
                               conflicts: pd.DataFrame) -> pd.DataFrame:
    """Issues found before deduplication: markup in labels, exact copies, conflicting keys."""
    raw = normalized["ticker_raw"].astype("string").fillna("")
    marked = raw != normalized["ticker"].astype("string").fillna("")
    return _concat_issues([
        _issues(normalized, marked, "ticker_markup", "raw label '" + raw + "' normalized"),
        _issues(removed, pd.Series(True, index=removed.index), "exact_duplicate",
                "exact copy of another row after ticker normalization; removed"),
        _issues(conflicts, pd.Series(True, index=conflicts.index), "conflicting_duplicate",
                "same (ticker, ts) with different values; all copies quarantined"),
    ])


def build_quality_report(unique: pd.DataFrame, calendar: pd.DataFrame,
                         reference: pd.DataFrame | None = None,
                         pre_dedup_issues: pd.DataFrame | None = None,
                         withheld: pd.DataFrame | None = None) -> pd.DataFrame:
    """Combine every detector into ``ticker | issue_type | issue_date | description | severity``.

    ``unique`` should hold normalized rows with unique keys. An observation can
    appear under several issue types. Pass ``withheld`` when reporting on
    cleaned data so dates removed by cleaning are not mistaken for source gaps.
    """
    parts = [
        pre_dedup_issues if pre_dedup_issues is not None else _empty_issues(),
        detect_missing_values(unique),
        detect_missing_dates(unique, calendar, withheld),
        detect_invalid_prices(unique),
        detect_ohlc_inconsistencies(unique),
        detect_volume_issues(unique),
        detect_return_outliers(unique),
        detect_price_spikes(unique),
        detect_adj_close_anomalies(unique, reference),
        detect_weekend_dates(unique),
    ]
    report = _concat_issues(parts)
    return report.sort_values(["ticker", "issue_date", "issue_type"], kind="mergesort").reset_index(drop=True)


# --------------------------------------------------------------------- cleaning

@dataclass
class CleaningResult:
    """Outputs of :func:`clean_ohlc`; ``counts`` reconciles every input row."""

    cleaned: pd.DataFrame
    quarantine: pd.DataFrame
    removed: pd.DataFrame
    repair_audit: pd.DataFrame
    ticker_audit: pd.DataFrame
    counts: dict

    def withheld(self) -> pd.DataFrame:
        """Delivered ``(ticker, ts)`` keys that cleaning took out, with the reason.

        Exact duplicates are excluded because their kept copy is still present
        (unless that copy was itself quarantined, which the quarantine covers).
        """
        q = self.quarantine[KEY + ["quarantine_reason"]].rename(columns={"quarantine_reason": "reason"})
        r = self.removed.loc[self.removed["removal_reason"] != "exact_duplicate", KEY + ["removal_reason"]]
        out = pd.concat([q, r.rename(columns={"removal_reason": "reason"})], ignore_index=True)
        return out.dropna(subset=KEY).reset_index(drop=True)


QUARANTINE_COLUMNS = (["source_rowid", "ticker_raw", "ticker", "ts_raw", "ts"] + VALUE_COLUMNS
                      + ["duplicate_copies_removed", "quarantine_reason", "detail"])
REMOVED_COLUMNS = (["source_rowid", "ticker_raw", "ticker", "ts_raw", "ts"] + VALUE_COLUMNS
                   + ["removal_reason", "duplicate_of_source_rowid"])
AUDIT_COLUMNS = ["source_rowid", "ticker", "ts", "column", "original_value", "repaired_value",
                 "reason", "method", "evidence"]


class _Ledger:
    """Collects quarantine and repair records while cleaning proceeds.

    ``original`` holds every row exactly as delivered (indexed by
    ``source_rowid``), so quarantined rows keep their original values.
    """

    def __init__(self, original: pd.DataFrame):
        self.original = original
        self.quarantined: list[pd.DataFrame] = []
        self.repairs: list[pd.DataFrame] = []

    def quarantine(self, frame: pd.DataFrame, mask, reason: str, detail) -> pd.DataFrame:
        mask = np.asarray(mask, dtype=bool)
        if mask.any():
            rows = self.original.loc[frame.index[mask]].copy()
            copies = frame["duplicate_copies_removed"].to_numpy()[mask] \
                if "duplicate_copies_removed" in frame else 0
            rows["duplicate_copies_removed"] = copies
            rows["quarantine_reason"] = reason
            rows["detail"] = detail.to_numpy()[mask] if isinstance(detail, pd.Series) else detail
            self.quarantined.append(rows)
        return frame.loc[~mask]

    def repair(self, frame: pd.DataFrame, mask, column: str, before: pd.Series,
               after: pd.Series, reason, method: str, evidence) -> None:
        mask = np.asarray(mask, dtype=bool)
        if not mask.any():
            return
        idx = frame.index[mask]
        pick = lambda v: v.loc[idx].to_numpy() if isinstance(v, pd.Series) else v
        self.repairs.append(pd.DataFrame({
            "source_rowid": idx, "ticker": frame.loc[idx, "ticker"].to_numpy(),
            "ts": frame.loc[idx, "ts"].to_numpy(), "column": column,
            "original_value": before.loc[idx].to_numpy(), "repaired_value": after.loc[idx].to_numpy(),
            "reason": pick(reason), "method": method, "evidence": pick(evidence),
        }))


def _bounds_ok(frame: pd.DataFrame) -> pd.Series:
    return ((frame["low"] <= frame[["open", "close"]].min(axis=1))
            & (frame["high"] >= frame[["open", "close"]].max(axis=1)))


def _valid_price(values: pd.Series) -> pd.Series:
    return values.notna() & np.isfinite(values) & (values > 0)


def valid_bar(frame: pd.DataFrame) -> pd.Series:
    """True where all four raw prices are present, finite, positive and within bounds."""
    return _valid_price_frame(frame[RAW_PRICES]).all(axis=1) & _bounds_ok(frame)


def interpolate_confirmed_errors(frame: pd.DataFrame, mask) -> tuple[pd.DataFrame, pd.Series]:
    """Linearly interpolate raw OHLC on the ``mask`` rows only, then revalidate them.

    Each field is interpolated from the same field of the surrounding rows of
    the same ticker (ordered by date) and never extrapolated past the ends.
    Only masked rows change: other gaps in the frame are left for the
    forward-fill step. Returns the updated frame and a Series that is True
    for masked rows whose repaired bar is valid; the caller quarantines the rest.
    """
    mask = pd.Series(np.asarray(mask, dtype=bool), index=frame.index)
    out = frame.copy()
    if not mask.any():
        return out, mask
    blanked = out[RAW_PRICES].mask(mask, np.nan)
    interpolated = blanked.groupby(out["ticker"], sort=False).transform(
        lambda s: s.interpolate(limit_area="inside"))
    out.loc[mask, RAW_PRICES] = interpolated.loc[mask, RAW_PRICES]
    return out, mask & valid_bar(out)


def clean_ohlc(raw: pd.DataFrame, reference: pd.DataFrame) -> CleaningResult:
    """Clean a raw ``ohlc_hw`` delivery into unique, validated, flagged observations.

    ``raw`` must contain ``source_rowid`` (see :func:`load_table`). ``reference``
    is the supplied ``ohlc`` table, used only for date-matched adjustment
    factors; its prices never replace the delivery's own raw prices.

    Order of operations (each step sees the output of the previous one):

    1. normalize tickers and dates; quarantine rows missing either key
    2. remove exact duplicates; quarantine every row of a conflicting key
    3. drop rows whose four raw prices are all missing
    4. blank nonpositive / non-finite raw cells; quarantine rows that are
       entirely invalid or whose valid OHLC violate bounds (the faulty field
       cannot be identified)
    5. interpolate confirmed isolated price spikes between valid neighbours
    6. forward-fill remaining raw gaps within the ticker; quarantine leading
       gaps and fills that break today's high/low range
    7. fill missing/invalid volume with 0 and flag it
    8. rebuild missing or confirmed-corrupt adj_close from validated factors
    9. recheck bounds, then attach review flags
    """
    if "source_rowid" not in raw:
        raise ValueError("raw must include source_rowid; use load_table(..., with_rowid=True)")
    n_input = len(raw)

    frame, ticker_audit = normalize_tickers(raw)
    frame["ts_raw"] = frame["ts"]
    frame["ts"] = to_iso_date(frame["ts_raw"])
    frame = frame.set_index("source_rowid", drop=False)
    frame.index.name = None
    delivered = frame.copy()
    for col in VALUE_COLUMNS:
        frame[col] = pd.to_numeric(frame[col], errors="coerce").astype(float)
    ledger = _Ledger(delivered)

    # 1 ------------------------------------------------------------- keys
    frame = ledger.quarantine(frame, frame["ticker"].isna(), "missing_ticker", "ticker empty after normalization")
    frame = ledger.quarantine(frame, frame["ts"].isna(), "invalid_date", "ts could not be parsed as a date")

    # 2 ------------------------------------------------------- duplicates
    frame, removed_dupes = split_exact_duplicates(frame)
    removed_dupes["removal_reason"] = "exact_duplicate"
    _, conflicts = split_conflicting_keys(frame)
    frame = ledger.quarantine(frame, frame.index.isin(conflicts.index), "conflicting_duplicate_key",
                              "same (ticker, ts) delivered with different values")
    frame = _sorted(frame)

    # 3 ------------------------------------------------- all prices missing
    all_missing = frame[RAW_PRICES].isna().all(axis=1)
    removed_empty = frame.loc[all_missing].copy()
    removed_empty["removal_reason"] = "all_raw_prices_missing"
    frame = frame.loc[~all_missing].copy()
    original = frame.copy()

    # 4 ---------------------------------------------------- invalid cells
    invalid_cells = frame[RAW_PRICES].notna() & ~_valid_price_frame(frame[RAW_PRICES])
    frame[RAW_PRICES] = frame[RAW_PRICES].mask(invalid_cells)
    frame = ledger.quarantine(frame, frame[RAW_PRICES].isna().all(axis=1), "all_raw_prices_invalid",
                              "no valid raw price on this row")
    complete = frame[RAW_PRICES].notna().all(axis=1)
    frame = ledger.quarantine(frame, complete & ~_bounds_ok(frame), "ohlc_bounds_violation",
                              "valid prices violate low <= open/close <= high; faulty field unknown")

    # 5 ---------------------------------------------------------- spikes
    spikes = find_isolated_price_spikes(frame)
    spike_mask = frame.index.isin(spikes.index)
    if spike_mask.any():
        before = frame[RAW_PRICES].copy()
        frame, repaired_ok = interpolate_confirmed_errors(frame, spike_mask)
        frame = ledger.quarantine(frame, spike_mask & ~repaired_ok.to_numpy(), "interpolation_failed_validation",
                                  "interpolated OHLC is missing or violates low <= open/close <= high")
        evidence = pd.Series(
            "close " + spikes["close"].astype("string") + " vs " + spikes["prev_ts"].astype("string") + " "
            + spikes["prev_close"].astype("string") + " / " + spikes["next_ts"].astype("string") + " "
            + spikes["next_close"].astype("string") + "; volume 0; one-day reversal", index=spikes.index)
        repaired_rows = frame.index.isin(spikes.index)
        for col in RAW_PRICES:
            ledger.repair(frame, repaired_rows, col, before[col], frame[col],
                          "isolated_price_spike", "linear_interpolation_between_neighbours", evidence)

    # 6 ---------------------------------------------------- forward fill
    gaps = frame[RAW_PRICES].isna()
    filled = frame.groupby("ticker", sort=False)[RAW_PRICES].ffill()
    leading = gaps & filled.isna()
    leading_rows = leading.any(axis=1)
    detail = pd.Series("", index=frame.index)
    detail[leading_rows] = [
        "no earlier observation to forward-fill " + ", ".join(
            f"{c} (original {original.at[i, c]})" for c in RAW_PRICES if leading.at[i, c])
        for i in frame.index[leading_rows]]
    frame = ledger.quarantine(frame, leading_rows, "unresolved_leading_gap", detail)
    gaps, filled = gaps.loc[frame.index], filled.loc[frame.index]
    before = original.loc[frame.index, RAW_PRICES]
    frame[RAW_PRICES] = frame[RAW_PRICES].fillna(filled)
    fill_reason = pd.DataFrame(np.where(invalid_cells.loc[frame.index], "nonpositive_or_nonfinite", "missing"),
                               index=frame.index, columns=RAW_PRICES)
    frame = ledger.quarantine(frame, gaps.any(axis=1) & ~_bounds_ok(frame), "forward_fill_violates_bounds",
                              "carried price falls outside the day's high/low range")
    gaps = gaps.loc[frame.index]
    for col in RAW_PRICES:
        ledger.repair(frame, gaps[col], col, before[col], frame[col], fill_reason[col],
                      "forward_fill_within_ticker", "last valid price of the same ticker")
    price_method = pd.Series(pd.NA, index=frame.index, dtype="string")
    price_method[gaps.any(axis=1)] = "forward_fill_within_ticker"
    price_method[frame.index.isin(spikes.index)] = "linear_interpolation_between_neighbours"
    close_repaired = gaps["close"] | pd.Series(frame.index.isin(spikes.index), index=frame.index)

    # 7 ---------------------------------------------------------- volume
    vol = frame["volume"]
    bad_vol = vol.isna() | np.isinf(vol) | (vol < 0)
    before_vol = vol.copy()
    frame["volume"] = vol.mask(bad_vol, 0.0)
    ledger.repair(frame, bad_vol, "volume", before_vol, frame["volume"],
                  np.where(vol[bad_vol].isna(), "missing", "invalid") if bad_vol.any() else "",
                  "fill_zero_flagged", "unknown volume; 0 is a placeholder, not evidence of no trading")
    frame["volume_filled"] = bad_vol

    # 8 -------------------------------------------------- adjusted close
    frame, adj_info = _repair_adjusted_close(frame, reference, close_repaired, ledger)

    # 9 ---------------------------------------------------- final checks
    bad = ~(_bounds_ok(frame) & frame[PRICES].apply(_valid_price).all(axis=1))
    frame = ledger.quarantine(frame, bad, "post_repair_validation_failed", "row failed final validation")

    frame["price_repaired"] = price_method.loc[frame.index].notna()
    frame["price_repair_method"] = price_method.loc[frame.index]
    for col in ["adj_close_source", "adj_close_repair_reason", "adj_factor_review_flag"]:
        frame[col] = adj_info[col].loc[frame.index]
    frame["adj_close_repaired"] = frame["adj_close_source"] != "vendor"
    frame["return_outlier_flag"] = flag_return_outliers(frame)["is_outlier"]
    frame["zero_volume_flag"] = (frame["volume"] == 0) & ~frame["volume_filled"]
    frame["weekend_date_flag"] = pd.to_datetime(frame["ts"]).dt.dayofweek >= 5
    frame["volume"] = frame["volume"].round().astype("int64")

    columns = (["ts", "ticker"] + VALUE_COLUMNS
               + ["price_repaired", "price_repair_method", "adj_close_repaired", "adj_close_source",
                  "adj_close_repair_reason", "adj_factor_review_flag", "volume_filled",
                  "return_outlier_flag", "zero_volume_flag", "weekend_date_flag",
                  "duplicate_copies_removed", "ticker_raw", "source_rowid"])
    cleaned = _sorted(frame[columns]).reset_index(drop=True)

    removed = pd.concat([
        delivered.loc[removed_dupes.index].assign(
            removal_reason="exact_duplicate",
            duplicate_of_source_rowid=removed_dupes["duplicate_of_source_rowid"]),
        delivered.loc[removed_empty.index].assign(removal_reason="all_raw_prices_missing",
                                                  duplicate_of_source_rowid=pd.NA),
    ])
    removed = removed.reindex(columns=REMOVED_COLUMNS).reset_index(drop=True)
    removed["duplicate_of_source_rowid"] = removed["duplicate_of_source_rowid"].astype("Int64")
    quarantine = (pd.concat(ledger.quarantined) if ledger.quarantined
                  else pd.DataFrame(columns=QUARANTINE_COLUMNS)).reindex(columns=QUARANTINE_COLUMNS)
    repair_audit = (pd.concat(ledger.repairs, ignore_index=True) if ledger.repairs
                    else pd.DataFrame(columns=AUDIT_COLUMNS))[AUDIT_COLUMNS]

    counts = {
        "input_rows": n_input,
        "removed_exact_duplicates": len(removed_dupes),
        "removed_all_prices_missing": len(removed_empty),
        "quarantined": len(quarantine),
        "retained": len(cleaned),
        "retained_with_price_repair": int(cleaned["price_repaired"].sum()),
        "retained_with_adj_close_repair": int(cleaned["adj_close_repaired"].sum()),
        "retained_with_volume_fill": int(cleaned["volume_filled"].sum()),
    }
    accounted = counts["removed_exact_duplicates"] + counts["removed_all_prices_missing"] \
        + counts["quarantined"] + counts["retained"]
    if accounted != n_input:
        raise AssertionError(f"row accounting mismatch: {accounted} != {n_input}")
    copies_flagged = int(cleaned["duplicate_copies_removed"].sum() + quarantine["duplicate_copies_removed"].sum())
    if copies_flagged != len(removed_dupes):
        raise AssertionError(f"duplicate accounting mismatch: {copies_flagged} != {len(removed_dupes)}")
    return CleaningResult(cleaned, quarantine.reset_index(drop=True), removed, repair_audit, ticker_audit, counts)


def _valid_price_frame(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.notna() & np.isfinite(prices) & (prices > 0)


def _reference_factors(reference: pd.DataFrame) -> pd.DataFrame:
    """Reference close, adjusted close, factor and the factor's own anomaly deviation."""
    ref = reference.copy()
    ref["ticker"] = ref["ticker"].map(normalize_ticker).astype("string")
    ref["ts"] = to_iso_date(ref["ts"])
    ref = _sorted(ref.drop_duplicates(subset=KEY, keep=False)).reset_index(drop=True)
    factor, dev = adj_factor_deviation(ref)
    return pd.DataFrame({"ticker": ref["ticker"], "ts": ref["ts"], "ref_close": ref["close"],
                         "ref_adj_close": ref["adj_close"], "ref_factor": factor, "ref_factor_dev": dev})


def _repair_adjusted_close(frame: pd.DataFrame, reference: pd.DataFrame, close_repaired: pd.Series,
                           ledger: _Ledger) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rebuild adj_close only where it is missing or confirmed corrupt.

    * A factor is *confirmed corrupt* when it is anomalous against the ticker's
      own rolling median AND disagrees with the reference, or when both sources
      share the same anomalous factor (the reference itself is then unusable).
    * Pass 1 uses the date-matched reference factor, but only when the reference
      row's raw close equals ours and its factor is not itself anomalous.
    * Pass 2 handles what is left with the neighbouring rows' factors, only when
      both neighbours exist, are not under suspicion and agree within
      ``NEIGHBOR_FACTOR_TOL`` (so no corporate action falls in between).
    * Missing or corrupt values that neither pass can resolve are quarantined;
      an anomalous factor that cannot be confirmed is kept and flagged.
    """
    frame = frame.copy()
    ref = frame[KEY].merge(_reference_factors(reference), on=KEY, how="left").set_index(frame.index)
    adj = frame["adj_close"].copy()
    _, dev = adj_factor_deviation(frame)

    missing = ~_valid_price(adj)
    anomalous = (dev.abs() > FACTOR_TOL) & ~missing
    ref_present = _valid_price(ref["ref_adj_close"])
    matches_ref = ref_present & np.isclose(adj, ref["ref_adj_close"], rtol=REFERENCE_RTOL, atol=0)
    ref_valid = (ref_present & np.isclose(frame["close"], ref["ref_close"], rtol=1e-9, atol=0)
                 & (ref["ref_factor_dev"].abs() <= FACTOR_TOL))
    corrupted = anomalous & ref_present & ~matches_ref
    shared = anomalous & matches_ref & ~ref_valid

    reason = pd.Series(pd.NA, index=frame.index, dtype="string")
    reason[shared] = "anomalous_factor_shared_with_reference"
    reason[corrupted] = "corrupted_vendor_factor"
    reason[missing] = np.where(adj[missing].isna(), "missing", "nonpositive_or_nonfinite")
    reason[close_repaired] = "raw_close_repaired"
    needs = reason.notna()

    source = pd.Series("vendor", index=frame.index, dtype="string")
    evidence = pd.Series(pd.NA, index=frame.index, dtype="string")

    use_ref = needs & ref_valid
    frame.loc[use_ref, "adj_close"] = frame.loc[use_ref, "close"] * ref.loc[use_ref, "ref_factor"]
    source[use_ref] = "reference_factor"
    evidence[use_ref] = ("ohlc factor " + _fmt(ref.loc[use_ref, "ref_factor"], "{:.6f}")
                         + " on the same date; raw close matches")

    pending = needs & ~use_ref
    unconfirmed = anomalous & ~needs
    usable = (frame["adj_close"] / frame["close"]).where(~pending & ~unconfirmed)
    g = usable.groupby(frame["ticker"], sort=False)
    prev_f, next_f = g.shift(1), g.shift(-1)
    agree = prev_f.notna() & next_f.notna() & ((next_f / prev_f - 1).abs() <= NEIGHBOR_FACTOR_TOL)
    use_nb = pending & agree
    factor_nb = (prev_f + next_f) / 2
    frame.loc[use_nb, "adj_close"] = frame.loc[use_nb, "close"] * factor_nb[use_nb]
    source[use_nb] = "neighbor_factor"
    evidence[use_nb] = ("neighbour factors " + _fmt(prev_f[use_nb], "{:.6f}") + " / "
                        + _fmt(next_f[use_nb], "{:.6f}") + " agree; reference factor not validated")

    repaired = use_ref | use_nb
    ledger.repair(frame, repaired, "adj_close", adj, frame["adj_close"], reason,
                  "close_times_validated_factor", evidence)

    unresolved = needs & ~repaired
    keep_flagged = unresolved & shared & ~close_repaired
    review = unconfirmed | keep_flagged

    info = pd.DataFrame({"adj_close_source": source, "adj_close_repair_reason": reason.where(repaired),
                         "adj_factor_review_flag": review})
    detail = "adj_close " + reason.astype("string") + "; no validated reference or agreeing neighbour factor"
    frame = ledger.quarantine(frame, unresolved & ~keep_flagged, "adj_close_unresolved", detail)
    return frame, info


# ------------------------------------------------------------------- validation

ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")


def validate_cleaned(frame: pd.DataFrame) -> pd.DataFrame:
    """Run every required post-cleaning check; returns ``check | passed | failures``."""
    f = frame
    prices = f[PRICES]
    _, dev = adj_factor_deviation(f)
    flagged = f["adj_factor_review_flag"].astype(bool) if "adj_factor_review_flag" in f else False
    checks = {
        "no missing prices": prices.isna().any(axis=1),
        "all prices finite": ~np.isfinite(prices.fillna(1)).all(axis=1),
        "all prices positive": (prices <= 0).any(axis=1),
        "unique (ticker, ts)": f.duplicated(subset=KEY, keep=False),
        "volume present, finite, >= 0": f["volume"].isna() | ~np.isfinite(f["volume"].astype(float))
        | (f["volume"] < 0),
        "low <= open <= high": (f["low"] > f["open"]) | (f["open"] > f["high"]),
        "low <= close <= high": (f["low"] > f["close"]) | (f["close"] > f["high"]),
        "low <= high": f["low"] > f["high"],
        "ts stored as ISO YYYY-MM-DD": ~f["ts"].astype("string").str.fullmatch(ISO_DATE).fillna(False)
        | pd.to_datetime(f["ts"], format="%Y-%m-%d", errors="coerce").isna(),
        "tickers normalized": f["ticker"].astype("string").fillna("")
        != f["ticker"].map(normalize_ticker).astype("string").fillna("<missing>"),
        "adj factor anomalies all flagged for review": (dev.abs() > FACTOR_TOL) & ~flagged,
    }
    return pd.DataFrame([{"check": name, "passed": not bool(mask.any()), "failures": int(mask.sum())}
                         for name, mask in checks.items()])


def to_sql_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert booleans to 0/1 integers so SQLite stores flags consistently."""
    out = frame.copy()
    for col in out.columns:
        if out[col].dtype == bool or str(out[col].dtype) == "boolean":
            out[col] = out[col].astype("int64")
    return out
