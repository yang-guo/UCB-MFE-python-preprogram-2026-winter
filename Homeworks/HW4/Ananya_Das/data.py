"""Load HW4 data without ever returning information past the allowed date.

Two layers of protection against looking at the final test period:

1. Before touching the database, `allowed_end` / `check_end` reject any
   requested end date after BUFFER_END (Mar 31, 2016), unless
   config.RUN_FINAL_TEST is True.
2. After loading, `check_no_rows_after` verifies that no returned date column
   goes past the requested end, in case a query filter is ever wrong.

Adapted from Lectures/modeling/course_data.py (`load_data`,
`load_membership`): the read-only DuckDB queries, the optional symbol batching,
and the quarantine of invalid or off-calendar prices follow that code. Changes:
the date guard, the post-load date check, loading the membership's
capitalization proxy, and excluding fundamentals that have no filing date.
"""

import warnings

import duckdb
import numpy as np
import pandas as pd

import config


class LookaheadError(RuntimeError):
    """Raised when code asks for data it is not allowed to see yet."""


def allowed_end():
    """Latest date any loader may return right now."""
    return config.TEST_END if config.RUN_FINAL_TEST else config.BUFFER_END


def check_end(end):
    """Turn `end` into a Timestamp, or raise LookaheadError if it is too late."""
    end = pd.Timestamp(end)
    if pd.isna(end):
        raise ValueError("end must be a real date")
    limit = allowed_end()
    if end > limit:
        reason = ("the final test has not been approved (config.RUN_FINAL_TEST is False)"
                  if not config.RUN_FINAL_TEST else "the data stops at the test end")
        raise LookaheadError(f"Refusing to load data through {end.date()}: the limit is "
                             f"{limit.date()} because {reason}.")
    return end


def check_no_rows_after(frame, columns, end, name):
    """Second line of defense: fail loudly if any loaded date is after `end`."""
    for column in columns:
        latest = frame[column].max()
        if pd.notna(latest) and latest > end:
            raise LookaheadError(f"{name}.{column} contains {latest.date()}, after {end.date()}")


def _to_datetime(frame, columns):
    for column in columns:
        frame[column] = pd.to_datetime(frame[column]).astype("datetime64[ns]")


def load_data(end, database=None, symbols=None):
    """Return (prices, fundamentals, universe, benchmark, membership) through `end`.

    end:      last date to load; must not be after allowed_end().
    database: path to a DuckDB file (default: config.DATABASE).
    symbols:  optional list of tickers, for loading the big database in batches.

    Only static, non-survivorship universe fields are loaded (symbol, sector).
    Active/delisted flags and selection counts are deliberately left out, since
    they describe the future relative to most of the sample.
    """
    end = check_end(end)  # Guard runs BEFORE the database is opened.
    database = config.DATABASE if database is None else database
    if not database.is_file():
        raise FileNotFoundError(f"Download the database from bCourses to {database}")

    symbol_filter = " AND symbol IN (SELECT unnest(?))" if symbols is not None else ""
    symbol_params = [list(symbols)] if symbols is not None else []
    with duckdb.connect(str(database), read_only=True) as con:
        prices = con.execute(f"""
            SELECT symbol, date, close, adj_close, volume FROM prices
            WHERE date <= ? {symbol_filter} ORDER BY symbol, date""",
            [end] + symbol_params).fetchdf()
        # A record counts only once it has been filed, so filter on filing_date.
        # Records without a filing date cannot be timed and are excluded.
        fundamentals = con.execute(f"""
            SELECT symbol, fiscal_date, period, filing_date, reported_currency,
                   revenue, net_income, operating_income, total_assets, total_debt,
                   cash_and_equivalents, free_cash_flow, operating_cash_flow
            FROM fundamentals
            WHERE filing_date IS NOT NULL AND filing_date <= ? AND fiscal_date <= ?
            {symbol_filter} ORDER BY symbol, fiscal_date""",
            [end, end] + symbol_params).fetchdf()
        universe = con.execute(f"""
            SELECT symbol, sector FROM universe WHERE true {symbol_filter}
            ORDER BY symbol""", symbol_params).fetchdf()
        benchmark = con.execute("""
            SELECT date, adj_close FROM benchmark WHERE date <= ? ORDER BY date""",
            [end]).fetchdf()
        # Only quarters that have started by `end`; a later quarter's list is
        # for entries we are not allowed to make yet.
        membership = con.execute(f"""
            SELECT symbol, quarter_start, quarter_end, selection_date,
                   market_cap_proxy, universe_rank
            FROM universe_membership
            WHERE quarter_start <= ? AND selection_date <= ? {symbol_filter}
            ORDER BY quarter_start, universe_rank""",
            [end, end] + symbol_params).fetchdf()

    _to_datetime(prices, ["date"])
    _to_datetime(fundamentals, ["fiscal_date", "filing_date"])
    _to_datetime(benchmark, ["date"])
    _to_datetime(membership, ["quarter_start", "quarter_end", "selection_date"])

    check_no_rows_after(prices, ["date"], end, "prices")
    check_no_rows_after(fundamentals, ["fiscal_date", "filing_date"], end, "fundamentals")
    check_no_rows_after(benchmark, ["date"], end, "benchmark")
    check_no_rows_after(membership, ["quarter_start", "selection_date"], end, "membership")

    # Basic integrity checks (same as the course loader).
    assert not prices.duplicated(["symbol", "date"]).any(), "duplicate price rows"
    assert not benchmark.date.duplicated().any(), "duplicate benchmark dates"
    assert (np.isfinite(benchmark.adj_close) & benchmark.adj_close.gt(0)).all()
    assert not membership.duplicated(["symbol", "quarter_start"]).any(), "duplicate membership"

    # Quarantine prices we cannot use, and say so, rather than silently fixing them.
    invalid = ~np.isfinite(prices.adj_close) | prices.adj_close.le(0)
    if invalid.any():
        warnings.warn(f"Quarantining {int(invalid.sum())} non-positive/non-finite adjusted prices.",
                      stacklevel=2)
        prices = prices.loc[~invalid].copy()
    off_calendar = ~prices.date.isin(benchmark.date)
    if off_calendar.any():
        warnings.warn(f"Quarantining {int(off_calendar.sum())} price rows not on the market calendar.",
                      stacklevel=2)
        prices = prices.loc[~off_calendar].copy()

    return prices, fundamentals, universe, benchmark, membership
