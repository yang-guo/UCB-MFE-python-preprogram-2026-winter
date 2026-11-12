"""Loading functions copied verbatim from Lectures/modeling/course_data.py.

Only the two loading functions and their required imports are included.
The original defaults are preserved: end="2015-12-31" and a data directory
relative to this file. Pass database and end explicitly for HW4 experiments.
"""

from pathlib import Path
import warnings

import duckdb
import numpy as np
import pandas as pd


def load_data(database=None, end="2015-12-31", symbols=None):
    """Select columns/dates and optional ticker batches in read-only SQL."""
    database = Path(database) if database else Path(__file__).parent / "data/equities_class.duckdb"
    if not database.is_file():
        raise FileNotFoundError(f"Download the modeling database from bCourses to {database}")
    symbol_filter = " AND symbol IN (SELECT unnest(?))" if symbols is not None else ""
    symbol_params = [list(symbols)] if symbols is not None else []
    with duckdb.connect(str(database), read_only=True) as con:
        prices = con.execute(f"""SELECT symbol,date,close,adj_close,volume FROM prices
            WHERE date<=? {symbol_filter} ORDER BY symbol,date""", [end] + symbol_params).fetchdf()
        fundamentals = con.execute(f"""SELECT symbol,fiscal_date,period,filing_date,
            reported_currency,revenue,net_income,operating_income,total_assets,
            total_debt,cash_and_equivalents,free_cash_flow,operating_cash_flow
            FROM fundamentals WHERE fiscal_date<=? AND (filing_date<=? OR filing_date IS NULL)
            {symbol_filter} ORDER BY symbol,fiscal_date""", [end, end] + symbol_params).fetchdf()
        universe = con.execute(f"SELECT symbol,sector FROM universe WHERE true {symbol_filter} ORDER BY symbol", symbol_params).fetchdf()
        benchmark = con.execute("SELECT date,adj_close FROM benchmark WHERE date<=? ORDER BY date", [end]).fetchdf()
    for frame, columns in [(prices, ["date"]), (fundamentals, ["fiscal_date", "filing_date"]), (benchmark, ["date"])]:
        for column in columns:
            frame[column] = pd.to_datetime(frame[column]).astype("datetime64[ns]")
    assert not prices.duplicated(["symbol", "date"]).any()
    invalid_price = ~np.isfinite(prices.adj_close) | prices.adj_close.le(0)
    if invalid_price.any():
        warnings.warn(f"Quarantining {int(invalid_price.sum())} invalid adjusted-price rows; inspect the raw database and delivery audit.", stacklevel=2)
        prices = prices.loc[~invalid_price].copy()
    assert not benchmark.date.duplicated().any()
    assert (np.isfinite(benchmark.adj_close) & benchmark.adj_close.gt(0)).all()
    off_calendar = ~prices.date.isin(benchmark.date)
    if off_calendar.any():
        warnings.warn(f"Quarantining {int(off_calendar.sum())} price rows outside the supplied market calendar; inspect the raw database and delivery audit.", stacklevel=2)
        prices = prices.loc[~off_calendar].copy()
    return prices, fundamentals, universe, benchmark


def load_membership(database=None, end="2015-12-31"):
    """Quarterly membership is a dated decision input, not the static stock pool."""
    database = Path(database) if database else Path(__file__).parent / "data/equities_class.duckdb"
    with duckdb.connect(str(database), read_only=True) as con:
        membership = con.execute("""SELECT symbol,quarter_start,quarter_end,selection_date,universe_rank
            FROM universe_membership WHERE quarter_start<=? ORDER BY quarter_start,universe_rank""", [end]).fetchdf()
    for column in ["quarter_start", "quarter_end", "selection_date"]:
        membership[column] = pd.to_datetime(membership[column]).astype("datetime64[ns]")
    return membership
