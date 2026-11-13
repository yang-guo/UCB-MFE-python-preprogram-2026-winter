"""Read-only database access with the data guide's quarantine rules.

Adapted from `Lectures/modeling/course_data.py` (`load_data`, `load_membership`):
same read-only SQL + symbol batching, same quarantine of non-positive adjusted
prices (BHRB's 275 zero rows) and of the 28 rows whose dates are not in the
supplied market calendar. Changes: float32 columns to save RAM, a configurable
start date, and asserts that the calendar and benchmark agree.
"""

import hashlib
import json
import warnings

import duckdb
import numpy as np
import pandas as pd

from . import config


def connect(database):
    if not database.is_file():
        raise FileNotFoundError(f"Download the database from bCourses to {database}")
    return duckdb.connect(str(database), read_only=True)


def _dates(frame, columns):
    for c in columns:
        frame[c] = pd.to_datetime(frame[c]).astype("datetime64[ns]")
    return frame


def load_calendar_and_benchmark(database, end=config.TEST_END):
    """Market calendar = the `calendar` table; it must equal the benchmark dates."""
    with connect(database) as con:
        cal = con.execute("SELECT date FROM calendar WHERE date<=? ORDER BY date", [end]).fetchdf()
        bench = con.execute("SELECT date, adj_close FROM benchmark WHERE date<=? ORDER BY date", [end]).fetchdf()
    cal, bench = _dates(cal, ["date"]), _dates(bench, ["date"])
    assert cal.date.is_unique and bench.date.is_unique
    assert cal.date.equals(bench.date), "calendar and benchmark sessions differ"
    assert (np.isfinite(bench.adj_close) & bench.adj_close.gt(0)).all()
    return pd.DatetimeIndex(cal.date), bench.set_index("date").adj_close


def list_symbols(database):
    with connect(database) as con:
        return [r[0] for r in con.execute("SELECT symbol FROM universe ORDER BY symbol").fetchall()]


def load_prices(database, symbols, calendar, start=config.LOOKBACK_START, end=config.TEST_END):
    """Prices for a symbol batch, with invalid/off-calendar rows quarantined.

    `close` and `volume` are split-adjusted by the vendor (checked: NVDA 2010
    close = $0.46), so close*volume is true dollar volume. adj_close also
    includes dividends and is the only price used for returns and accounting.
    """
    with connect(database) as con:
        p = con.execute("""SELECT symbol, date, close, adj_close, volume FROM prices
            WHERE date BETWEEN ? AND ? AND symbol IN (SELECT unnest(?)) ORDER BY symbol, date""",
                        [start, end, list(symbols)]).fetchdf()
    p = _dates(p, ["date"])
    assert not p.duplicated(["symbol", "date"]).any()
    bad = ~np.isfinite(p.adj_close) | p.adj_close.le(0)
    if bad.any():
        warnings.warn(f"Quarantining {int(bad.sum())} non-positive adjusted-price rows "
                      f"({sorted(p.loc[bad, 'symbol'].unique())})", stacklevel=2)
        p = p.loc[~bad]
    off = ~p.date.isin(calendar)
    if off.any():
        warnings.warn(f"Quarantining {int(off.sum())} off-calendar price rows", stacklevel=2)
        p = p.loc[~off]
    for c in ["close", "adj_close"]:
        p[c] = p[c].astype("float64")   # keep precision for returns; downcast features later
    p["volume"] = p.volume.astype("float64")
    return p.reset_index(drop=True)


FUNDAMENTAL_COLUMNS = ["revenue", "gross_profit", "operating_income", "net_income",
                       "operating_cash_flow", "free_cash_flow", "total_assets", "total_equity",
                       "total_debt", "cash_and_equivalents", "shares_outstanding"]


def load_fundamentals(database, end=config.TEST_END):
    """All quarterly statement rows (284k rows; small enough to load at once).

    Only rows with filing_date <= end are read, so nothing filed after the study
    end can leak in. The availability rule is applied in features.py.
    """
    cols = ", ".join(FUNDAMENTAL_COLUMNS)
    with connect(database) as con:
        f = con.execute(f"""SELECT symbol, fiscal_date, period, filing_date, {cols}
            FROM fundamentals WHERE fiscal_date<=? AND filing_date<=?
            ORDER BY symbol, fiscal_date""", [end, end]).fetchdf()
    return _dates(f, ["fiscal_date", "filing_date"])


def load_membership(database, end=config.TEST_END):
    """Quarterly membership is a dated decision input (course_data.load_membership)."""
    with connect(database) as con:
        m = con.execute("""SELECT symbol, quarter_start, selection_date, universe_rank
            FROM universe_membership WHERE quarter_start<=? ORDER BY quarter_start, universe_rank""",
                        [end]).fetchdf()
    m = _dates(m, ["quarter_start", "selection_date"])
    if m.duplicated(["symbol", "quarter_start"]).any():
        raise ValueError("Duplicate quarterly membership key")
    return m


def load_sectors(database):
    """Snapshot sector labels (audit caveat: not dated history)."""
    with connect(database) as con:
        u = con.execute("SELECT symbol, sector FROM universe ORDER BY symbol").fetchdf()
    u["sector"] = u.sector.fillna("Unknown")
    return u.set_index("symbol").sector


def file_sha256(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def database_fingerprint(database):
    """sha256 of the file, checked against manifest.json, plus dataset_metadata."""
    manifest = json.loads((config.DATA_DIR / "manifest.json").read_text())
    expected = manifest["files"].get(database.name, {}).get("sha256")
    actual = file_sha256(database)
    with connect(database) as con:
        meta = dict(con.execute("SELECT key, value FROM dataset_metadata").fetchall())
    return {"file": database.name, "sha256": actual, "manifest_sha256": expected,
            "matches_manifest": actual == expected,
            "snapshot_sha256": meta.get("snapshot_sha256"), "variant": meta.get("variant")}
