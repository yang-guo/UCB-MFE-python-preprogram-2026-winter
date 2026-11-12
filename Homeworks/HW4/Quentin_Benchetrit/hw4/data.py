"""Read the course DuckDB file into date x symbol matrices.

The two quarantine rules for unusable price rows (non-positive adjusted closes
and dates outside the benchmark calendar) are taken from load_data in
Lectures/modeling/course_data.py. Everything else is mine.
"""

from dataclasses import dataclass, field
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

FUNDAMENTAL_COLUMNS = [
    "revenue", "gross_profit", "operating_income", "net_income", "ebitda",
    "operating_cash_flow", "free_cash_flow", "total_assets", "total_debt",
    "total_equity", "cash_and_equivalents", "shares_outstanding", "rd_expense",
]


def find_database(name="equities_top2000.duckdb"):
    """Look for Lectures/modeling/data/<name> in the folders above this file."""
    for folder in Path(__file__).resolve().parents:
        candidate = folder / "Lectures" / "modeling" / "data" / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"{name} not found; download it to Lectures/modeling/data/")


@dataclass
class Market:
    """Prices on the benchmark calendar. A missing quote is NaN, never filled.

    adj_close is the series used for returns and fills; raw_adj_close is the
    vendor series before forfeit_jumps, kept for the entry screen.
    """

    calendar: pd.DatetimeIndex
    adj_close: pd.DataFrame
    raw_adj_close: pd.DataFrame
    close: pd.DataFrame
    volume: pd.DataFrame
    benchmark: pd.Series
    sectors: pd.Series
    audit: dict = field(default_factory=dict)

    @property
    def symbols(self):
        return self.adj_close.columns


def _matrix(rows, cols, values, shape):
    out = np.full(shape, np.nan)
    out[rows, cols] = values
    return out


def isolated_spikes(prices, jump=10.0, tolerance=2.0):
    """Flag quotes that sit more than `jump` times above or below both
    neighbouring quotes while those neighbours agree with each other.

    These are vendor glitches (a stock printing 10,000 for one day between two
    closes at 1). Removing the row leaves a missing quote, so no order can be
    filled at the bogus price.
    """
    previous = prices.ffill().shift(1)
    following = prices.bfill().shift(-1)
    to_previous = np.log(prices / previous).abs()
    to_following = np.log(prices / following).abs()
    neighbours = np.log(following / previous).abs()
    return ((to_previous > np.log(jump)) & (to_following > np.log(jump))
            & (neighbours < np.log(tolerance)))


def forfeit_jumps(prices, limit=2.0):
    """Rescale each series so that no single session gains more than `limit`x.

    The vendor missed some reverse splits (LVLT prints 1.55 then 23.10 on
    2011-10-20), which would hand a holder a fake fifteen-fold gain. I give up
    every one-day gain above 100%, real or not, by dividing all later prices by
    the jump. Earlier prices are untouched, so nothing dated t changes when
    data after t arrives. Drops are left alone: a missed forward split then
    costs a fake loss, which is the conservative side to err on.
    """
    ratio = prices / prices.ffill().shift(1)
    factor = ratio.where(ratio > limit, 1.0)
    return prices / factor.cumprod(), int((factor > 1).to_numpy().sum())


def load_market(database=None, start="2008-01-01", end=None):
    """Load prices from `start` (one year of warm-up before 2010 is enough)."""
    database = Path(database) if database else find_database()
    end = end or "2100-01-01"
    with duckdb.connect(str(database), read_only=True) as con:
        bench = con.execute(
            "SELECT date, adj_close FROM benchmark WHERE date BETWEEN ? AND ? ORDER BY date",
            [start, end]).fetchdf()
        universe = con.execute("SELECT symbol, sector FROM universe ORDER BY symbol").fetchdf()
        rows = con.execute("""
            SELECT s.j, c.i, p.close, p.adj_close, p.volume
            FROM prices p
            JOIN (SELECT symbol, row_number() OVER (ORDER BY symbol) - 1 AS j FROM universe) s
              USING (symbol)
            LEFT JOIN (SELECT date, row_number() OVER (ORDER BY date) - 1 AS i
                       FROM benchmark WHERE date BETWEEN ? AND ?) c USING (date)
            WHERE p.date BETWEEN ? AND ?""", [start, end, start, end]).fetchdf()

    calendar = pd.DatetimeIndex(pd.to_datetime(bench.date)).astype("datetime64[ns]")
    benchmark = pd.Series(bench.adj_close.to_numpy(), index=calendar, name="^SP500TR")
    assert calendar.is_unique and calendar.is_monotonic_increasing
    assert (np.isfinite(benchmark) & (benchmark > 0)).all()

    off_calendar = rows.i.isna()
    bad_price = ~off_calendar & ~(np.isfinite(rows.adj_close) & (rows.adj_close > 0))
    rows = rows.loc[~off_calendar & ~bad_price]
    i, j = rows.i.to_numpy(int), rows.j.to_numpy(int)
    shape = (len(calendar), len(universe))
    symbols = pd.Index(universe.symbol, name="symbol")
    frames = {name: pd.DataFrame(_matrix(i, j, rows[name].to_numpy(float), shape),
                                 index=calendar, columns=symbols)
              for name in ["adj_close", "close", "volume"]}

    spikes = isolated_spikes(frames["adj_close"])
    for frame in frames.values():
        frame[spikes] = np.nan
    raw = frames["adj_close"]
    adjusted, forfeited = forfeit_jumps(raw)

    audit = {"off_calendar_rows": int(off_calendar.sum()),
             "non_positive_adj_close_rows": int(bad_price.sum()),
             "isolated_spike_rows": int(spikes.to_numpy().sum()),
             "symbols_with_spikes": int(spikes.any().sum()),
             "forfeited_one_day_gains": forfeited}
    sectors = universe.set_index("symbol").sector.fillna("Unknown")
    return Market(calendar, adjusted, raw, frames["close"], frames["volume"],
                  benchmark, sectors, audit)


def load_fundamentals(database=None):
    """Quarterly statements with the columns used for features."""
    database = Path(database) if database else find_database()
    columns = ", ".join(FUNDAMENTAL_COLUMNS)
    with duckdb.connect(str(database), read_only=True) as con:
        f = con.execute(f"""SELECT symbol, fiscal_date, filing_date, {columns}
            FROM fundamentals WHERE period = 'quarter' ORDER BY symbol, fiscal_date""").fetchdf()
    for column in ["fiscal_date", "filing_date"]:
        f[column] = pd.to_datetime(f[column]).astype("datetime64[ns]")
    return f


def load_membership(database=None):
    """Quarterly top-2,000 lists. `quarter_start` identifies the entry quarter."""
    database = Path(database) if database else find_database()
    with duckdb.connect(str(database), read_only=True) as con:
        m = con.execute("""SELECT symbol, quarter_start, selection_date, universe_rank
            FROM universe_membership ORDER BY quarter_start, universe_rank""").fetchdf()
    for column in ["quarter_start", "selection_date"]:
        m[column] = pd.to_datetime(m[column]).astype("datetime64[ns]")
    assert not m.duplicated(["symbol", "quarter_start"]).any()
    return m
