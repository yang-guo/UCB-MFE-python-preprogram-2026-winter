"""Read the course DuckDB file into dates x symbols tables.

Every query is cut at ``end``. The development script passes
``end="2015-12-31"``, so nothing from the test period is even loaded there.

The quarantine rules (non-positive adjusted prices, rows outside the supplied
market calendar) follow ``Lectures/modeling/course_data.py::load_data``.
"""

from dataclasses import dataclass
from pathlib import Path
import warnings

import duckdb
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "Lectures" / "modeling" / "data"
FULL_DB = DATA_DIR / "equities_top2000.duckdb"
CLASS_DB = DATA_DIR / "equities_class.duckdb"

FUNDAMENTAL_COLUMNS = [
    "revenue", "gross_profit", "operating_income", "net_income", "operating_cash_flow",
    "free_cash_flow", "total_assets", "total_debt", "total_equity",
    "cash_and_equivalents", "shares_outstanding",
]


@dataclass
class MarketData:
    sessions: pd.DatetimeIndex  # market calendar (benchmark sessions)
    adj_close: pd.DataFrame  # sessions x symbols, NaN = no quote
    close: pd.DataFrame  # vendor close (used only with volume / shares)
    dollar_volume: pd.DataFrame  # close * volume
    benchmark: pd.Series  # ^SP500TR adjusted close
    membership: pd.DataFrame  # symbol, quarter_start, universe_rank
    fundamentals: pd.DataFrame  # quarterly statements with filing dates
    sectors: pd.Series  # snapshot sector label per symbol


def load_market_data(database=FULL_DB, start="2008-01-01", end="2026-09-30"):
    """Load prices, benchmark, membership and fundamentals up to ``end``."""
    database = Path(database)
    if not database.is_file():
        raise FileNotFoundError(f"Download the modeling database from bCourses to {database}")
    with duckdb.connect(str(database), read_only=True) as con:
        benchmark = con.execute(
            "SELECT date, adj_close FROM benchmark WHERE date BETWEEN ? AND ? ORDER BY date",
            [start, end]).fetchdf()
        symbols = [row[0] for row in con.execute("SELECT symbol FROM universe ORDER BY symbol").fetchall()]
        # Integer codes keep the 16M-row price read small enough for 8 GB of RAM.
        raw = con.execute("""
            WITH s AS (SELECT symbol, row_number() OVER (ORDER BY symbol) - 1 AS j FROM universe),
                 c AS (SELECT date, row_number() OVER (ORDER BY date) - 1 AS i
                       FROM benchmark WHERE date BETWEEN ? AND ?)
            SELECT c.i, s.j, p.adj_close, p.close, p.volume
            FROM prices p JOIN s USING (symbol) LEFT JOIN c USING (date)
            WHERE p.date BETWEEN ? AND ?""", [start, end, start, end]).fetchnumpy()
        membership = con.execute("""SELECT symbol, quarter_start, universe_rank
            FROM universe_membership WHERE quarter_start <= ? ORDER BY quarter_start, universe_rank""",
            [end]).fetchdf()
        fundamentals = con.execute(f"""SELECT symbol, fiscal_date, filing_date, {', '.join(FUNDAMENTAL_COLUMNS)}
            FROM fundamentals WHERE period = 'quarter' AND fiscal_date <= ? AND filing_date <= ?
            ORDER BY symbol, fiscal_date""", [end, end]).fetchdf()
        sectors = con.execute("SELECT symbol, sector FROM universe ORDER BY symbol").fetchdf()

    sessions = pd.DatetimeIndex(pd.to_datetime(benchmark.date).astype("datetime64[ns]"))
    benchmark = pd.Series(benchmark.adj_close.to_numpy(), index=sessions, name="benchmark")
    assert sessions.is_unique and (benchmark > 0).all()

    i = np.ma.filled(raw["i"].astype(float), np.nan) if np.ma.isMaskedArray(raw["i"]) else raw["i"].astype(float)
    j = np.asarray(raw["j"]).astype(int)
    adj = np.ma.filled(raw["adj_close"], np.nan) if np.ma.isMaskedArray(raw["adj_close"]) else raw["adj_close"]
    close = np.ma.filled(raw["close"], np.nan) if np.ma.isMaskedArray(raw["close"]) else raw["close"]
    volume = (np.ma.filled(raw["volume"].astype(float), np.nan)
              if np.ma.isMaskedArray(raw["volume"]) else raw["volume"].astype(float))

    off_calendar = np.isnan(i)
    invalid = ~np.isfinite(adj) | (adj <= 0)
    if off_calendar.any():
        warnings.warn(f"Quarantining {int(off_calendar.sum())} price rows outside the market calendar.", stacklevel=2)
    if (invalid & ~off_calendar).any():
        warnings.warn(f"Quarantining {int((invalid & ~off_calendar).sum())} invalid adjusted-price rows.", stacklevel=2)
    keep = ~off_calendar & ~invalid
    i, j = i[keep].astype(int), j[keep]

    def table(values, dtype):
        matrix = np.full((len(sessions), len(symbols)), np.nan, dtype=dtype)
        matrix[i, j] = values[keep]
        return pd.DataFrame(matrix, index=sessions, columns=symbols)

    for column in ["quarter_start"]:
        membership[column] = pd.to_datetime(membership[column]).astype("datetime64[ns]")
    for column in ["fiscal_date", "filing_date"]:
        fundamentals[column] = pd.to_datetime(fundamentals[column]).astype("datetime64[ns]")
    return MarketData(
        sessions=sessions,
        adj_close=table(adj, np.float64),
        close=table(close, np.float32),
        dollar_volume=table(close * volume, np.float32),
        benchmark=benchmark,
        membership=membership,
        fundamentals=fundamentals,
        sectors=sectors.set_index("symbol").sector.fillna("Unknown"),
    )
