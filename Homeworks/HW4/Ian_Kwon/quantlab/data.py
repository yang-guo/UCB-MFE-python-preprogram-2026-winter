"""Read-only database access with the research/final firewall applied in SQL.

Adapted from ``Lectures/modeling/course_data.py`` (``load_data``,
``load_membership``): same tables, read-only connections, quarantine of
non-positive and off-calendar prices.  Differences: prices are returned as
date x symbol matrices (15.8M rows would not fit comfortably as a long pandas
frame), the load end date comes from ``protocol.data_end(mode)``, and every
quarantined row is kept in an audit table instead of only a warning.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from . import protocol
from .protocol import Mode

COURSE_ROOT = protocol.SUBMISSION_ROOT.parents[2]
DATA_DIR = COURSE_ROOT / "Lectures" / "modeling" / "data"
DEFAULT_DB = DATA_DIR / "equities_top2000.duckdb"
CLASS_DB = DATA_DIR / "equities_class.duckdb"
WARMUP_START = pd.Timestamp("2007-01-01")  # >= 252 + 21 sessions before the first 2010 signal

FUNDAMENTAL_COLUMNS = [
    "revenue", "gross_profit", "operating_income", "net_income", "total_assets", "total_debt",
    "total_equity", "cash_and_equivalents", "free_cash_flow", "operating_cash_flow", "shares_outstanding",
]


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_database(path: Path) -> dict:
    """Compare the file's SHA-256 with the course manifest (provenance check)."""
    manifest = json.loads((DATA_DIR / "manifest.json").read_text())
    expected = manifest["files"].get(Path(path).name, {}).get("sha256")
    actual = file_sha256(path)
    if expected and actual != expected:
        raise ValueError(f"{path.name} checksum {actual[:12]} != manifest {expected[:12]}")
    return {"database": Path(path).name, "sha256": actual, "manifest_match": expected == actual}


def _connect(path: Path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Download the database from bCourses to {path}")
    return duckdb.connect(str(path), read_only=True)


def _end_for(mode: Mode, auth) -> pd.Timestamp:
    mode = Mode(mode)
    if mode is Mode.FINAL:
        protocol.require_final(auth)
    return protocol.data_end(mode)


@dataclass
class MarketData:
    """Calendar-aligned price matrices.  Rows = market sessions, columns = symbols.

    ``adj`` is the vendor adjusted close with invalid rows set to NaN (never
    filled).  ``close``/``volume`` are kept for liquidity screens only.
    A NaN means "no trustworthy quote that session", not a zero return.
    """

    calendar: pd.DatetimeIndex
    symbols: pd.Index
    adj: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    benchmark: pd.Series
    mode: Mode
    end: pd.Timestamp
    provenance: dict = field(default_factory=dict)
    load_quarantine: pd.DataFrame = field(default_factory=pd.DataFrame)

    def frame(self, which="adj") -> pd.DataFrame:
        return pd.DataFrame(getattr(self, which), index=self.calendar, columns=self.symbols)

    def loc(self, date) -> int:
        return int(self.calendar.get_loc(pd.Timestamp(date)))


def load_market(database: Path = DEFAULT_DB, mode: Mode = Mode.RESEARCH, auth=None,
                start=WARMUP_START, verify=True) -> MarketData:
    end = _end_for(mode, auth)
    start = pd.Timestamp(start)
    provenance = verify_database(database) if verify else {"database": Path(database).name}
    with _connect(database) as con:
        cal = con.execute("SELECT date FROM calendar WHERE date BETWEEN ? AND ? ORDER BY date",
                          [start, end]).fetchdf()
        calendar = pd.DatetimeIndex(pd.to_datetime(cal.date)).astype("datetime64[ns]")
        symbols = pd.Index(con.execute("SELECT symbol FROM universe ORDER BY symbol").fetchdf().symbol)
        # Integer keys only: avoids materializing 10M+ Python strings.
        arrays = con.execute("""
            WITH cal AS (SELECT date, row_number() OVER (ORDER BY date) - 1 AS di
                         FROM calendar WHERE date BETWEEN ? AND ?),
                 sym AS (SELECT symbol, row_number() OVER (ORDER BY symbol) - 1 AS si FROM universe)
            SELECT cal.di, sym.si, p.adj_close, p.close, p.volume
            FROM prices p JOIN cal USING (date) JOIN sym USING (symbol)""", [start, end]).fetchnumpy()
        off_calendar = con.execute("""
            SELECT symbol, date, adj_close, 'off_calendar' AS reason FROM prices
            WHERE date BETWEEN ? AND ? AND date NOT IN (SELECT date FROM calendar)""", [start, end]).fetchdf()
        bench = con.execute("SELECT date, adj_close FROM benchmark WHERE date BETWEEN ? AND ? ORDER BY date",
                            [start, end]).fetchdf()
    T, N = len(calendar), len(symbols)
    adj = np.full((T, N), np.nan)
    close = np.full((T, N), np.nan)
    volume = np.full((T, N), np.nan)
    di, si = arrays["di"].astype(np.int64), arrays["si"].astype(np.int64)
    a = np.asarray(arrays["adj_close"], dtype=float)
    invalid = ~np.isfinite(a) | (a <= 0)
    adj[di[~invalid], si[~invalid]] = a[~invalid]
    close[di, si] = np.asarray(arrays["close"], dtype=float)
    volume[di, si] = np.asarray(arrays["volume"], dtype=float)
    invalid_rows = pd.DataFrame({"symbol": symbols[si[invalid]], "date": calendar[di[invalid]],
                                 "adj_close": a[invalid], "reason": "nonpositive_or_nonfinite_adj_close"})
    off_calendar["date"] = pd.to_datetime(off_calendar.date).astype("datetime64[ns]")
    quarantine = pd.concat([invalid_rows, off_calendar], ignore_index=True)
    benchmark = pd.Series(bench.adj_close.to_numpy(), name="^SP500TR",
                          index=pd.DatetimeIndex(pd.to_datetime(bench.date)).astype("datetime64[ns]"))
    if not benchmark.index.equals(calendar):
        raise ValueError("Benchmark sessions must equal the supplied calendar")
    if not (np.isfinite(benchmark) & (benchmark > 0)).all():
        raise ValueError("Benchmark must be finite and positive")
    if Mode(mode) is Mode.RESEARCH:
        protocol.assert_research_safe(calendar, "research calendar")
    return MarketData(calendar, symbols, adj, close, volume, benchmark, Mode(mode), end,
                      provenance, quarantine)


@dataclass
class Reference:
    """Slowly changing inputs: static sectors, dated membership, financial statements."""

    universe: pd.DataFrame
    membership: pd.DataFrame
    fundamentals: pd.DataFrame


def load_reference(database: Path = DEFAULT_DB, mode: Mode = Mode.RESEARCH, auth=None) -> Reference:
    end = _end_for(mode, auth)
    cols = ", ".join(FUNDAMENTAL_COLUMNS)
    with _connect(database) as con:
        universe = con.execute("""SELECT symbol, name, sector, industry, exchange, is_active_snapshot
                                  FROM universe ORDER BY symbol""").fetchdf()
        membership = con.execute("""SELECT quarter_start, quarter_end, selection_date, symbol, universe_rank,
                                           market_cap_proxy FROM universe_membership
                                    WHERE selection_date <= ? ORDER BY quarter_start, universe_rank""",
                                 [end]).fetchdf()
        # Load statements whose (possibly imputed) availability could precede `end`.
        # The +120 day bound matches features.PLACEHOLDER_AVAILABILITY_DAYS.
        fundamentals = con.execute(f"""SELECT symbol, fiscal_date, period, filing_date, {cols}
            FROM fundamentals
            WHERE period = 'quarter'
              AND ((filing_date > fiscal_date AND filing_date + INTERVAL 1 DAY <= ?)
                   OR (filing_date <= fiscal_date AND fiscal_date + INTERVAL 120 DAY <= ?))
            ORDER BY symbol, fiscal_date""", [end, end]).fetchdf()
    for frame, columns in [(membership, ["quarter_start", "quarter_end", "selection_date"]),
                           (fundamentals, ["fiscal_date", "filing_date"])]:
        for c in columns:
            frame[c] = pd.to_datetime(frame[c]).astype("datetime64[ns]")
    if membership.duplicated(["symbol", "quarter_start"]).any():
        raise ValueError("Duplicate quarterly membership key")
    return Reference(universe, membership, fundamentals)
