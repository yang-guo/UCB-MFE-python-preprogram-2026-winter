"""Quarantine invalid observations without altering source data or filling gaps."""

from pathlib import Path
import tempfile
import unittest
import warnings

import duckdb
import numpy as np
import pandas as pd

from data_loading import load_data


class DataLoadingChecks(unittest.TestCase):
    def make_database(self, path, invalid_benchmark=False):
        prices = pd.DataFrame({
            "symbol": ["A"] * 7,
            "date": pd.date_range("2020-01-01", periods=7),
            "adj_close": [10.0, 0.0, np.inf, np.nan, -1.0, 12.0, 13.0],
            "close": 10.0, "volume": 100,
        })
        # January 7 is deliberately outside the benchmark calendar.
        benchmark = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=6),
                                  "adj_close": [100.0] * 5 + ([0.0] if invalid_benchmark else [100.0])})
        with duckdb.connect(str(path)) as con:
            con.execute("CREATE TABLE prices AS SELECT * FROM prices")
            con.execute("CREATE TABLE benchmark AS SELECT * FROM benchmark")
            con.execute("CREATE TABLE universe (symbol VARCHAR, sector VARCHAR)")
            con.execute("""CREATE TABLE fundamentals (
                symbol VARCHAR, fiscal_date DATE, period VARCHAR, filing_date DATE,
                reported_currency VARCHAR, revenue DOUBLE, net_income DOUBLE,
                operating_income DOUBLE, total_assets DOUBLE, total_debt DOUBLE,
                cash_and_equivalents DOUBLE, free_cash_flow DOUBLE, operating_cash_flow DOUBLE)""")

    def test_quarantine_retains_valid_history_and_raw_source(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.duckdb"
            self.make_database(path)
            with warnings.catch_warnings(record=True) as notices:
                warnings.simplefilter("always")
                prices, _, _, benchmark = load_data(path, end="2026-09-30")
            self.assertEqual(prices.adj_close.tolist(), [10.0, 12.0])
            self.assertEqual(len(benchmark), 6)
            self.assertTrue(any("4 invalid" in str(w.message) for w in notices))
            self.assertTrue(any("1 price rows outside" in str(w.message) for w in notices))
            with duckdb.connect(str(path), read_only=True) as con:
                self.assertEqual(con.execute("SELECT count(*) FROM prices").fetchone()[0], 7)

    def test_invalid_benchmark_is_not_silently_removed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.duckdb"
            self.make_database(path, invalid_benchmark=True)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with self.assertRaises(AssertionError):
                    load_data(path, end="2026-09-30")
