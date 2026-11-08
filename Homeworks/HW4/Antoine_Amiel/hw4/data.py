"""Read the course DuckDB file into wide, date-indexed matrices.

Adapted from ``Lectures/modeling/course_data.py`` (``load_data``,
``financial_events``): the same quarantine rules for invalid / off-calendar
prices and the same availability rule for filings (filing date + 1 day, only
when the filing date is after the fiscal date, newest fiscal period wins).
The wide layout (one column per stock) is new and lets all features be
computed with vectorized rolling windows instead of per-stock loops.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[4]
DEFAULT_DB = REPO / "Lectures/modeling/data/equities_top2000.duckdb"


@dataclass
class MarketData:
    calendar: pd.DatetimeIndex     # benchmark sessions
    benchmark: pd.Series           # ^SP500TR adjusted close
    adj: pd.DataFrame              # adjusted close (dates x symbols), NaN = no quote
    dollar_volume: pd.DataFrame    # split-adjusted close x volume
    membership: pd.DataFrame       # quarterly top-2000 lists
    fundamentals: pd.DataFrame     # quarterly statements with availability dates
    sectors: pd.Series             # snapshot sector label by symbol
    quarantine: dict               # counts of rows set aside


def load_market_data(database: Path | str = DEFAULT_DB, end: str = "2026-09-30") -> MarketData:
    database = Path(database)
    if not database.is_file():
        raise FileNotFoundError(f"Download the database from bCourses to {database}")
    with duckdb.connect(str(database), read_only=True) as con:
        bench = con.execute("SELECT date, adj_close FROM benchmark WHERE date <= ? ORDER BY date", [end]).fetchdf()
        symbols = [r[0] for r in con.execute("SELECT DISTINCT symbol FROM prices ORDER BY symbol").fetchall()]
        # Map (date, symbol) to (row, column) inside SQL so that 15.8M rows
        # arrive as small numeric arrays instead of Python strings.
        q = con.execute("""
            WITH cal AS (SELECT date, row_number() OVER (ORDER BY date) - 1 AS r FROM benchmark WHERE date <= $1),
                 sym AS (SELECT symbol, row_number() OVER (ORDER BY symbol) - 1 AS c
                         FROM (SELECT DISTINCT symbol FROM prices))
            SELECT cal.r, sym.c, p.adj_close, p.close * p.volume AS dv, p.volume, cal.r IS NOT NULL AS on_cal
            FROM prices p JOIN sym USING (symbol) LEFT JOIN cal USING (date)
            WHERE p.date <= $1""", [end]).fetchnumpy()
        membership = con.execute("""SELECT quarter_start, selection_date, symbol, market_cap_proxy, universe_rank
                                    FROM universe_membership WHERE quarter_start <= ?""", [end]).fetchdf()
        fundamentals = con.execute("""SELECT * EXCLUDE (source, period_start, accounting_basis)
                                      FROM fundamentals WHERE period = 'quarter' AND fiscal_date <= ?""",
                                   [end]).fetchdf()
        sectors = con.execute("SELECT symbol, sector FROM universe").fetchdf()

    bench["date"] = pd.to_datetime(bench.date).astype("datetime64[ns]")
    benchmark = bench.set_index("date").adj_close.astype(float)
    calendar = pd.DatetimeIndex(benchmark.index)

    q = {k: np.ma.filled(np.ma.asarray(v).astype("float64"), np.nan) for k, v in q.items()}
    adj_close = q["adj_close"]
    bad = ~np.isfinite(adj_close) | (adj_close <= 0)
    off_cal = ~np.isfinite(q["r"])
    quarantine = {"invalid_adjusted_price_rows": int(bad.sum()), "off_calendar_rows": int((off_cal & ~bad).sum())}
    if bad.any() or off_cal.any():
        warnings.warn(f"Quarantining price rows: {quarantine}", stacklevel=2)
    keep = ~bad & ~off_cal
    rows = np.asarray(q["r"])[keep].astype(np.int64)
    cols = np.asarray(q["c"])[keep].astype(np.int64)
    if len(np.unique(rows * len(symbols) + cols)) != len(rows):
        raise ValueError("duplicate (date, symbol) price rows")
    adj_m = np.full((len(calendar), len(symbols)), np.nan, dtype="float32")
    adj_m[rows, cols] = adj_close[keep]
    dv = np.asarray(q["dv"], dtype="float64")[keep]
    vol = np.asarray(q["volume"], dtype="float64")[keep]
    dv_m = np.full_like(adj_m, np.nan)
    dv_m[rows, cols] = np.where(vol > 0, dv, np.nan)
    del q, adj_close, dv, vol, rows, cols
    adj = pd.DataFrame(adj_m, index=calendar, columns=pd.Index(symbols, name="symbol"))
    dollar_volume = pd.DataFrame(dv_m, index=calendar, columns=adj.columns)

    for c in ["quarter_start", "selection_date"]:
        membership[c] = pd.to_datetime(membership[c]).astype("datetime64[ns]")
    for c in ["fiscal_date", "filing_date"]:
        fundamentals[c] = pd.to_datetime(fundamentals[c]).astype("datetime64[ns]")
    sectors = sectors.set_index("symbol").sector.fillna("Unknown")
    return MarketData(calendar, benchmark, adj, dollar_volume, membership, fundamentals, sectors, quarantine)
