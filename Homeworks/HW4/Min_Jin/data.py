"""
Part 1: Load the HW4 database using the course's shared helpers.

Reused code: 
-- ``load_data``
-- ``load_membership`` come unchanged from``Lectures/modeling/course_data.py``. 
They return long tables, quarantine invalid prices, and never fill price
gaps. The difference I have added is that HW4 points them at the full
``equities_top2000.duckdb`` file and reads through the end of the test period.
"""

from pathlib import Path
import sys
import pandas as pd

# Course setup similar to the notebooks
course_root = next(p for p in [Path(__file__).resolve(), *Path(__file__).resolve().parents]
                   if (p / "pyproject.toml").is_file())
modeling_dir = course_root / "Lectures/modeling"
if str(modeling_dir) not in sys.path:
    sys.path.insert(0, str(modeling_dir))

from course_data import load_data, load_membership

# HW4 experiment settings
DATABASE = modeling_dir / "data/equities_top2000.duckdb"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

experiment = {
    "database": DATABASE.name,
    "development_start": "2010-01-01",
    "development_end": "2015-12-31",
    "buffer_end": "2016-03-31",
    "test_start": "2016-04-01",
    "test_end": "2026-09-30",
    "holding_calendar_days": 90,
    "initial_cash": 1000000,
    "risk_free_hurdle": 0.02,
}


def load_hw4_data(end=experiment["test_end"]):
    """
    Return prices, fundamentals, universe, benchmark, membership, calendar
    """
    prices, fundamentals, universe, benchmark = load_data(DATABASE, end=end)
    membership = load_membership(DATABASE, end=end)
    calendar = pd.DatetimeIndex(benchmark.date)
    return prices, fundamentals, universe, benchmark, membership, calendar


def summarize(prices, fundamentals, universe, benchmark, membership, calendar):
    """Print a short report so we can check the load against the manifest"""
    print(f"Sessions: {len(calendar):,}  ({calendar[0].date()} -> {calendar[-1].date()})")
    print(f"Price rows kept: {len(prices):,}  for {prices.symbol.nunique():,} stocks")
    counts = membership.groupby("quarter_start").size()
    print(f"Quarterly lists: {len(counts)}  (members per quarter: min {counts.min()}, max {counts.max()})")
    print(f"Fundamental rows: {len(fundamentals):,}  (quarterly: {fundamentals.period.eq('quarter').sum():,})")
    print(f"Universe stocks: {len(universe):,}")
    bench = benchmark.set_index("date").adj_close
    for day in [experiment["test_start"], experiment["test_end"]]:
        print(f"^SP500TR on {day}: {bench.loc[day]:,.2f}")
    print(f"prices table: {prices.memory_usage(deep=True).sum() / 1e6:,.0f} MB in pandas")

# print data reprot or loads functions but not data
if __name__ == "__main__":
    summarize(*load_hw4_data())
