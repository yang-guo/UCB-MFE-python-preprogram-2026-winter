"""Read-only loaders for the HW4 DuckDB file.

Conventions adapted from ``Lectures/modeling/course_data.py`` (load_data,
financial_events, apply_membership):

* invalid (non-positive / non-finite) adjusted prices are quarantined, never
  replaced by raw closes;
* price rows on dates missing from the supplied market calendar are dropped;
* prices are stored on the market calendar *without* forward filling, so a
  missing quote stays NaN everywhere downstream;
* a filing is usable from filing_date + 1 calendar day, nonpositive filing lags
  are excluded, and a late old fiscal period never replaces a newer one.

The implementation here is my own: it builds dense (date x symbol) numpy
matrices so that features for ~3,850 stocks can be computed with vectorised
rolling windows instead of a per-ticker Python loop.
"""

from dataclasses import dataclass
from pathlib import Path
import warnings

import duckdb
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_DB_LOCATIONS = [
    REPO_ROOT / "Lectures/modeling/data/equities_top2000.duckdb",  # location named in hw4.md
    REPO_ROOT / "equities_top2000.duckdb",
]


def _ns(values):
    """Pandas 3 defaults to microsecond datetimes; keep one unit everywhere."""
    return pd.to_datetime(values).astype("datetime64[ns]")


def find_database(path=None):
    if path is not None:
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    for candidate in DEFAULT_DB_LOCATIONS:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("Download equities_top2000.duckdb from bCourses to Lectures/modeling/data/")


@dataclass
class MarketData:
    calendar: pd.DatetimeIndex      # one entry per benchmark session
    symbols: pd.Index               # column order of every matrix
    adj_close: np.ndarray           # float64 (n_dates, n_symbols), NaN = no quote
    close: np.ndarray               # split-adjusted, not dividend-adjusted close
    volume: np.ndarray
    benchmark: pd.Series            # ^SP500TR adjusted close on the calendar
    sector: pd.Series               # snapshot sector by symbol (classification only)
    audit: dict

    def frame(self, name):
        return pd.DataFrame(getattr(self, name), index=self.calendar, columns=self.symbols)


def load_market_data(db_path, end="2026-09-30"):
    with duckdb.connect(str(db_path), read_only=True) as con:
        calendar = pd.DatetimeIndex(pd.to_datetime(
            con.execute("SELECT date FROM calendar WHERE date<=? ORDER BY date", [end]).fetchdf().date)
        ).astype("datetime64[ns]")
        symbols = pd.Index(con.execute("SELECT symbol FROM universe ORDER BY symbol").fetchdf().symbol)
        bench = con.execute("SELECT date, adj_close FROM benchmark WHERE date<=? ORDER BY date", [end]).fetchdf()
        sector = con.execute("SELECT symbol, sector FROM universe ORDER BY symbol").fetchdf()
        n_off = con.execute("""SELECT count(*) FROM prices p WHERE p.date<=?
                               AND NOT EXISTS (SELECT 1 FROM calendar c WHERE c.date=p.date)""", [end]).fetchone()[0]
        # Integer positions computed in SQL keep the 15.8M-row load cheap.
        raw = con.execute("""
            WITH cal AS (SELECT date, row_number() OVER (ORDER BY date) - 1 AS di FROM calendar WHERE date<=?),
                 sym AS (SELECT symbol, row_number() OVER (ORDER BY symbol) - 1 AS si FROM universe)
            SELECT cal.di, sym.si, p.adj_close, p.close, p.volume
            FROM prices p JOIN cal USING (date) JOIN sym USING (symbol)""", [end]).fetchnumpy()

    shape = (len(calendar), len(symbols))
    adj = np.full(shape, np.nan)
    close = np.full(shape, np.nan)
    vol = np.full(shape, np.nan)
    di, si = raw["di"].astype(np.int64), raw["si"].astype(np.int64)
    a = raw["adj_close"].astype(float)
    bad = ~np.isfinite(a) | (a <= 0)
    adj[di[~bad], si[~bad]] = a[~bad]
    close[di[~bad], si[~bad]] = raw["close"].astype(float)[~bad]
    vol[di[~bad], si[~bad]] = raw["volume"].astype(float)[~bad]
    if bad.any():
        warnings.warn(f"Quarantined {int(bad.sum())} invalid adjusted-price rows", stacklevel=2)
    if n_off:
        warnings.warn(f"Dropped {n_off} price rows on dates outside the market calendar", stacklevel=2)

    benchmark = pd.Series(bench.adj_close.to_numpy(), index=_ns(bench.date)).reindex(calendar)
    assert benchmark.notna().all(), "benchmark must quote on every calendar session"
    sector = sector.set_index("symbol").sector.fillna("Unknown").reindex(symbols)
    audit = {"invalid_adj_rows": int(bad.sum()), "off_calendar_rows": int(n_off), "price_rows_loaded": int((~bad).sum())}
    return MarketData(calendar, symbols, adj, close, vol, benchmark, sector, audit)


def load_membership(db_path, end="2026-09-30"):
    with duckdb.connect(str(db_path), read_only=True) as con:
        m = con.execute("""SELECT symbol, quarter_start, selection_date, universe_rank, market_cap_proxy
                           FROM universe_membership WHERE quarter_start<=? ORDER BY quarter_start, universe_rank""",
                        [end]).fetchdf()
    m["quarter_start"] = _ns(m.quarter_start)
    m["selection_date"] = _ns(m.selection_date)
    assert not m.duplicated(["symbol", "quarter_start"]).any()
    return m


FUND_COLUMNS = ["revenue", "gross_profit", "net_income", "operating_cash_flow", "total_assets",
                "total_equity", "total_debt", "shares_outstanding"]


def load_financial_events(db_path, end="2026-09-30"):
    """Availability-dated quarterly fundamentals with trailing-twelve-month sums.

    Every derived quantity (TTM sums, year-over-year changes) only combines
    fiscal quarters of the same company, and the event's ``available_date`` is
    the *latest* availability date of every quarter it uses, so nothing is
    known before all its inputs were published.
    """
    with duckdb.connect(str(db_path), read_only=True) as con:
        f = con.execute(f"""SELECT symbol, fiscal_date, filing_date, {", ".join(FUND_COLUMNS)}
                            FROM fundamentals WHERE period='quarter' AND filing_date<=?
                            ORDER BY symbol, fiscal_date""", [end]).fetchdf()
    for c in ["fiscal_date", "filing_date"]:
        f[c] = _ns(f[c])
    n_raw = len(f)
    f = f.loc[f.filing_date.gt(f.fiscal_date)].copy()          # untrusted lags excluded
    f = f.drop_duplicates(["symbol", "fiscal_date"], keep="last").sort_values(["symbol", "fiscal_date"])
    f["available_date"] = f.filing_date + pd.Timedelta(days=1)

    g = f.groupby("symbol", sort=False)
    span3 = (f.fiscal_date - g.fiscal_date.shift(3)).dt.days        # 4 consecutive quarters ~ 273 days
    span4 = (f.fiscal_date - g.fiscal_date.shift(4)).dt.days        # same quarter last year ~ 365 days
    ttm_ok = span3.between(250, 300)
    yoy_ok = span4.between(330, 400)
    for c in ["revenue", "gross_profit", "net_income", "operating_cash_flow"]:
        f[f"{c}_ttm"] = g[c].transform(lambda s: s.rolling(4, min_periods=4).sum()).where(ttm_ok)
    f["assets_lag4"] = g.total_assets.shift(4).where(yoy_ok)
    f["net_income_lag4"] = g.net_income.shift(4).where(yoy_ok)
    # Availability of an event = latest filing among the 5 quarters it can touch.
    avail_int = f.available_date.astype("int64")
    f["available_date"] = pd.to_datetime(
        avail_int.groupby(f.symbol).transform(lambda s: s.rolling(5, min_periods=1).max()), unit="ns"
    ).astype("datetime64[ns]")

    # A late filing of an old fiscal quarter must not replace a newer known period.
    f = f.sort_values(["symbol", "available_date", "fiscal_date"])
    f = f.loc[f.fiscal_date.eq(f.groupby("symbol").fiscal_date.cummax())]
    f = f.drop_duplicates(["symbol", "available_date"], keep="last")
    f.attrs["audit"] = {"quarter_rows": n_raw, "usable_events": len(f)}
    return f.reset_index(drop=True)
