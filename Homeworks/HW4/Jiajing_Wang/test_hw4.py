import unittest

import numpy as np
import pandas as pd

from hw4 import FEATURES, backtest, monthly_panel


class BacktestTests(unittest.TestCase):
    def test_round_trip_after_holiday(self):
        dates = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-03-31", "2020-04-02"])
        quotes = pd.DataFrame({"A": 10.0}, index=dates)
        history, fills, lots, *_ = backtest(
            quotes, {dates[1]: ["A"]}, capital=300, fee=0.01, start=dates[1]
        )
        self.assertEqual(fills.date.tolist(), [dates[1], dates[-1]])
        self.assertEqual(fills.due.iloc[1], pd.Timestamp("2020-04-01"))
        self.assertAlmostEqual(history.cash.iloc[1], 200)
        self.assertAlmostEqual(history.holdings.iloc[1], 100 / 1.01)
        self.assertAlmostEqual(history.nav.iloc[-1], 300 - 2 / 1.01)
        self.assertFalse(lots)

    def test_missing_prices_and_overlapping_lots(self):
        dates = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-02-03", "2020-04-01", "2020-04-02"])
        quotes = pd.DataFrame({"A": [10, 10, 12, np.nan, 14], "B": np.nan}, index=dates)
        picks = {dates[1]: ["A", "B"], dates[2]: ["A"]}
        history, fills, lots, rejected, stale, _ = backtest(
            quotes, picks, capital=300, fee=0, start=dates[1]
        )
        self.assertEqual(rejected, 1)
        self.assertEqual(stale, 1)
        self.assertAlmostEqual(history.holdings.loc[dates[3]], 160)
        self.assertEqual(fills.lot.tolist(), [0, 1, 0])
        self.assertEqual(fills.date.iloc[-1], dates[-1])
        self.assertEqual(len(lots), 1)
        self.assertAlmostEqual(history.nav.iloc[-1], 300 - 150 + 14 * (5 + 100 / 12))

    def test_features_ignore_execution_and_future_prices(self):
        dates = pd.bdate_range("2018-01-01", "2020-06-30")
        quotes = pd.DataFrame({"A": np.arange(len(dates)) + 10.0}, index=dates)
        entry = pd.Timestamp("2020-01-01")
        members = pd.DataFrame({"symbol": ["A"], "quarter_start": [entry],
                                "selection_date": [pd.Timestamp("2019-12-31")]})
        baseline = monthly_panel(quotes, members).query("entry == @entry")
        signal = dates.get_loc(entry) - 1
        for window in (5, 10):
            expected = quotes["A"].iloc[signal] / quotes["A"].iloc[signal - window] - 1
            self.assertAlmostEqual(baseline[f"return_{window}"].iloc[0], expected)
        quotes.loc[entry, "A"] *= 2
        changed = monthly_panel(quotes, members).query("entry == @entry")
        pd.testing.assert_frame_equal(baseline[FEATURES], changed[FEATURES])
        self.assertNotEqual(baseline.target.iloc[0], changed.target.iloc[0])


if __name__ == "__main__":
    unittest.main()
