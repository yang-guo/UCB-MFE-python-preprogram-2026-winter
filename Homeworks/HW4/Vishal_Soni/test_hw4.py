import unittest
import numpy as np
import pandas as pd
from hw4 import run_backtest, statistics


class AccountingTests(unittest.TestCase):
    def test_cash_maturity_and_open_lot(self):
        dates = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-04-01", "2020-04-02"])
        quotes = pd.DataFrame({"date": dates, "symbol": ["A"] * 4, "adj_close": [10., 10., 12., 13.]})
        orders = pd.DataFrame({"signal_date": [dates[0]], "symbol": ["A"], "score": [1.]})
        history, fills, lots, stale = run_backtest(dates, quotes, orders, start="2020-01-02", end="2020-04-02", initial=300, fee=0, slots=1)
        self.assertAlmostEqual(history.iloc[0].cash, 200)
        self.assertAlmostEqual(history.iloc[0].value, 300)
        self.assertEqual(fills.side.tolist(), ["buy", "sell"])
        self.assertEqual(fills.iloc[1].date, dates[2])  # 90 calendar days after entry
        self.assertAlmostEqual(history.iloc[-1].value, 320)
        self.assertEqual(lots, [])
        self.assertEqual(stale, 0)

    def test_missing_entry_is_not_filled(self):
        dates = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"])
        quotes = pd.DataFrame({"date": dates, "symbol": ["A"] * 3, "adj_close": [10., np.nan, 20.]})
        orders = pd.DataFrame({"signal_date": [dates[0]], "symbol": ["A"], "score": [1.]})
        h, fills, lots, _ = run_backtest(dates, quotes, orders, start="2020-01-02", end="2020-01-03", initial=300, fee=0)
        self.assertTrue(fills.empty)
        self.assertEqual(h.iloc[-1].value, 300)
        self.assertFalse(lots)

    def test_fee_in_first_day_drawdown(self):
        dates = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"])
        quotes = pd.DataFrame({"date": dates, "symbol": ["A"] * 3, "adj_close": [10., 10., 10.]})
        orders = pd.DataFrame({"signal_date": [dates[0]], "symbol": ["A"], "score": [1.]})
        h, _, lots, _ = run_backtest(dates, quotes, orders, start="2020-01-02", end="2020-01-03", initial=300, fee=.01)
        self.assertEqual(len(lots), 1)
        self.assertAlmostEqual(h.iloc[-1].value, 200 + 100 / 1.01)
        self.assertLess(statistics(h.value, 300)["max_drawdown"], 0)

    def test_stale_holding_stops(self):
        dates = pd.bdate_range("2020-01-01", periods=9)
        quotes = pd.DataFrame({"date": dates, "symbol": ["A"] * len(dates), "adj_close": [10., 10.] + [np.nan] * 7})
        orders = pd.DataFrame({"signal_date": [dates[0]], "symbol": ["A"], "score": [1.]})
        with self.assertRaisesRegex(ValueError, "price gap"):
            run_backtest(dates, quotes, orders, start=dates[1], end=dates[-1], initial=300, fee=0)


if __name__ == "__main__":
    unittest.main()
