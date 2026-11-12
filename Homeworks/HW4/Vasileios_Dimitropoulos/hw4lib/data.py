"""Read-only data access for HW4.

Prices are loaded into *wide* numpy matrices (market sessions x symbols) so that
trailing features can be computed with vectorised pandas operations instead of a
Python loop over 3,849 tickers.  Gaps stay NaN: we never forward-fill prices.

Adapted from ``Lectures/modeling/course_data.py`` (``load_data``,
``load_membership``): same quarantine rules for invalid adjusted prices and
off-calendar rows, same read-only DuckDB access.  The wide-matrix layout and
integer indexing are my own.
"""

from dataclasses import dataclass
from pathlib import Path
import warnings

import duckdb
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[4]
DATA_DIR = REPO_ROOT / "Lectures" / "modeling" / "data"
FULL_DB = DATA_DIR / "equities_top2000.duckdb"
CLASS_DB = DATA_DIR / "equities_class.duckdb"


@dataclass
class MarketData:
    """Everything the feature builder and the backtester need, in one place."""

    calendar: pd.DatetimeIndex      # one entry per market session (benchmark calendar)
    symbols: pd.Index               # column labels of the wide matrices
    adj: pd.DataFrame               # adjusted close (total-return units), NaN = no quote
    close: pd.DataFrame             # raw close as supplied (used for price/size screens)
    volume: pd.DataFrame            # raw volume as supplied
    benchmark: pd.Series            # ^SP500TR adjusted close on the calendar
    membership: pd.DataFrame        # dated quarterly universe membership
    universe: pd.DataFrame          # static symbol metadata (snapshot sector etc.)
    fundamentals: pd.DataFrame      # quarterly financial records
    audit: dict                     # counts of quarantined rows etc.


def _connect(database):
    database = Path(database)
    if not database.is_file():
        raise FileNotFoundError(f"Download the modeling database from bCourses to {database}")
    return duckdb.connect(str(database), read_only=True)


def load_market_data(database=FULL_DB, start="2008-01-01", end="2026-09-30"):
    """Load prices, benchmark, membership, universe and fundamentals.

    ``start`` only limits the look-back history (we need ~1 year of warm-up
    before the first 2010 development signal).  ``end`` is a hard cut-off: no
    row dated after it is ever read, so the test cannot peek past 2026-09-30.
    """
    audit = {}
    with _connect(database) as con:
        cal = con.execute("SELECT date FROM calendar WHERE date BETWEEN ? AND ? ORDER BY date",
                          [start, end]).fetchdf()
        calendar = pd.DatetimeIndex(pd.to_datetime(cal.date).astype("datetime64[ns]"), name="date")

        bench = con.execute("SELECT date, adj_close FROM benchmark WHERE date BETWEEN ? AND ? ORDER BY date",
                            [start, end]).fetchdf()
        benchmark = pd.Series(bench.adj_close.to_numpy(),
                              index=pd.DatetimeIndex(pd.to_datetime(bench.date).astype("datetime64[ns]")),
                              name="^SP500TR")
        assert benchmark.index.equals(calendar), "benchmark and calendar must share sessions"
        assert np.isfinite(benchmark).all() and (benchmark > 0).all()

        symbols = con.execute("SELECT symbol FROM universe ORDER BY symbol").fetchdf().symbol
        symbols = pd.Index(symbols.astype(str), name="symbol")

        # Price rows as integer coordinates: (session index, symbol index).
        raw = con.execute("""
            WITH cal AS (SELECT date, row_number() OVER (ORDER BY date) - 1 AS t
                         FROM calendar WHERE date BETWEEN ? AND ?),
                 sym AS (SELECT symbol, row_number() OVER (ORDER BY symbol) - 1 AS s FROM universe)
            SELECT cal.t, sym.s, p.adj_close, p.close, p.volume
            FROM prices p JOIN cal USING (date) JOIN sym USING (symbol)
            WHERE p.date BETWEEN ? AND ?""", [start, end, start, end]).fetchnumpy()
        audit["price_rows_in_window"] = con.execute(
            "SELECT count(*) FROM prices WHERE date BETWEEN ? AND ?", [start, end]).fetchone()[0]
        audit["off_calendar_rows_quarantined"] = audit["price_rows_in_window"] - len(raw["t"])

        membership = con.execute("""SELECT quarter_start, selection_date, symbol, universe_rank,
                                           selection_close, selection_shares, market_cap_proxy
                                    FROM universe_membership WHERE quarter_start <= ?
                                    ORDER BY quarter_start, universe_rank""", [end]).fetchdf()
        universe = con.execute("""SELECT symbol, name, sector, industry, exchange,
                                         is_active_snapshot FROM universe ORDER BY symbol""").fetchdf()
        fundamentals = con.execute("""
            SELECT symbol, fiscal_date, period, filing_date, revenue, gross_profit, operating_income,
                   net_income, operating_cash_flow, free_cash_flow, total_assets, total_debt,
                   total_equity, cash_and_equivalents, common_stock_issued, common_stock_repurchased
            FROM fundamentals
            WHERE fiscal_date <= ? AND (filing_date <= ? OR filing_date IS NULL)
            ORDER BY symbol, fiscal_date""", [end, end]).fetchdf()

    for frame, cols in [(membership, ["quarter_start", "selection_date"]),
                        (fundamentals, ["fiscal_date", "filing_date"])]:
        for c in cols:
            frame[c] = pd.to_datetime(frame[c]).astype("datetime64[ns]")
    universe["sector"] = universe.sector.fillna("Unknown")

    # Quarantine invalid adjusted prices (e.g. BHRB's zero rows) -- never substitute.
    t, s = raw["t"].astype(np.int64), raw["s"].astype(np.int64)
    adj_v = np.asarray(raw["adj_close"], dtype=float)
    bad = ~np.isfinite(adj_v) | (adj_v <= 0)
    audit["invalid_adj_rows_quarantined"] = int(bad.sum())
    if bad.any():
        warnings.warn(f"Quarantining {int(bad.sum())} invalid adjusted-price rows", stacklevel=2)
    keep = ~bad
    shape = (len(calendar), len(symbols))
    mats = {}
    for name, values in [("adj", adj_v), ("close", np.asarray(raw["close"], dtype=float)),
                         ("volume", np.asarray(raw["volume"], dtype=float))]:
        m = np.full(shape, np.nan)
        m[t[keep], s[keep]] = values[keep]
        mats[name] = pd.DataFrame(m, index=calendar, columns=symbols)
    # Duplicate (date, symbol) keys would silently overwrite; the schema forbids them.
    assert np.unique(t[keep] * shape[1] + s[keep]).size == keep.sum()
    if audit["off_calendar_rows_quarantined"]:
        warnings.warn(f"Quarantined {audit['off_calendar_rows_quarantined']} off-calendar price rows", stacklevel=2)

    return MarketData(calendar, symbols, mats["adj"], mats["close"], mats["volume"], benchmark,
                      membership, universe, fundamentals, audit)
