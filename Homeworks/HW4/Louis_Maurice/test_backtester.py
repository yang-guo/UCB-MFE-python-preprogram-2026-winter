"""Hand-checkable accounting and timing tests for backtester.py."""

import unittest

import numpy as np
import pandas as pd

from backtester import Backtester

DAYS = pd.to_datetime(["2020-01-30", "2020-01-31", "2020-02-03", "2020-02-04",
                       "2020-02-05", "2020-02-06", "2020-02-07"])


class BacktesterTests(unittest.TestCase):
    def run_case(self, a_prices):
        quotes = pd.DataFrame({"A": a_prices, "B": 50.0}, index=DAYS)
        scores = {pd.Timestamp("2020-01-31"): pd.Series({"A": 0.10, "B": 0.05})}
        bt = Backtester(quotes, scores, initial_cash=1000, fee_bps=100,
                        top_n=1, cohort_fraction=0.5, holding_days=3)
        return bt.run("2020-01-31", "2020-02-07")

    def test_buy_and_sell_by_hand(self):
        history, fills, open_lots, _ = self.run_case([9, 9, 10, 11, 11, 12, 12])
        units = 500 / (10 * 1.01)
        self.assertEqual(fills.date.iloc[0], pd.Timestamp("2020-02-03"))
        self.assertAlmostEqual(fills.units.iloc[0], units)
        self.assertAlmostEqual(history.loc["2020-02-03", "cash"], 500)
        self.assertAlmostEqual(history.loc["2020-02-04", "nav"], 500 + units * 11)
        self.assertEqual(fills.date.iloc[1], pd.Timestamp("2020-02-06"))
        self.assertAlmostEqual(history.loc["2020-02-06", "cash"], 500 + units * 12 * 0.99)
        self.assertTrue(open_lots.empty)

    def test_missing_quote_delays_exit_and_keeps_last_mark(self):
        history, fills, _, _ = self.run_case([9, 9, 10, 11, 11, np.nan, 13])
        units = 500 / (10 * 1.01)
        self.assertAlmostEqual(history.loc["2020-02-06", "nav"], 500 + units * 11)
        self.assertEqual(history.loc["2020-02-06", "stale_lots"], 1)
        self.assertEqual(fills.date.iloc[1], pd.Timestamp("2020-02-07"))
        self.assertAlmostEqual(history.loc["2020-02-07", "cash"], 500 + units * 13 * 0.99)


if __name__ == "__main__":
    unittest.main(verbosity=2)
