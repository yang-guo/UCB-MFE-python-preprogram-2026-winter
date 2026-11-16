"""Tests that the date guard in data.py blocks the final test period.

Run from Homeworks/HW4/Ananya_Das/:
    ../../../.venv/bin/python -m unittest -v tests.test_data
"""

from pathlib import Path
import sys
import unittest
from unittest import mock

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # Make config/data importable.

import config  # noqa: E402
import data  # noqa: E402


class TestDateGuard(unittest.TestCase):
    def test_first_test_day_is_refused_before_opening_database(self):
        # A path that does not exist: if the guard did not run first, we would
        # get FileNotFoundError instead of LookaheadError.
        missing = Path("does_not_exist.duckdb")
        with self.assertRaises(data.LookaheadError):
            data.load_data("2016-04-01", database=missing)

    def test_far_future_dates_are_refused(self):
        for end in ["2020-06-30", "2026-09-30", pd.Timestamp("2016-03-31 00:00:01")]:
            with self.subTest(end=end), self.assertRaises(data.LookaheadError):
                data.check_end(end)

    def test_last_buffer_day_is_allowed(self):
        self.assertEqual(data.check_end("2016-03-31"), config.BUFFER_END)

    def test_switch_is_off_by_default(self):
        self.assertFalse(config.RUN_FINAL_TEST)
        self.assertEqual(data.allowed_end(), pd.Timestamp("2016-03-31"))

    def test_even_with_switch_on_nothing_after_test_end(self):
        with mock.patch.object(config, "RUN_FINAL_TEST", True):
            with self.assertRaises(data.LookaheadError):
                data.check_end("2026-10-01")

    def test_post_load_check_catches_a_late_row(self):
        # Simulates a buggy query that let one test-period row through.
        frame = pd.DataFrame({"date": pd.to_datetime(["2015-12-31", "2016-04-01"])})
        with self.assertRaises(data.LookaheadError):
            data.check_no_rows_after(frame, ["date"], config.BUFFER_END, "prices")


@unittest.skipUnless(config.CLASS_DB.is_file(), "class database not downloaded")
class TestClassDatabaseLoad(unittest.TestCase):
    def test_allowed_load_returns_no_late_rows(self):
        prices, fundamentals, universe, benchmark, membership = data.load_data(
            config.DEV_END, database=config.CLASS_DB)
        self.assertLessEqual(prices.date.max(), config.DEV_END)
        self.assertLessEqual(fundamentals.filing_date.max(), config.DEV_END)
        self.assertLessEqual(benchmark.date.max(), config.DEV_END)
        self.assertLessEqual(membership.quarter_start.max(), config.DEV_END)
        self.assertEqual(list(universe.columns), ["symbol", "sector"])  # No survivorship fields.
        print(f"\n  loaded {len(prices):,} prices, {len(fundamentals):,} filings, "
              f"{universe.symbol.nunique()} symbols, {len(membership):,} membership rows; "
              f"last price {prices.date.max().date()}")


if __name__ == "__main__":
    unittest.main()
