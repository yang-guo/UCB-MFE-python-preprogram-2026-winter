"""Small examples with answers calculated independently by hand."""
import unittest

import numpy as np
import pandas as pd

from strategy.backtest import run, metrics
from strategy.model import training_rows
from strategy.data import extra_features


class AccountingTests(unittest.TestCase):
    def setUp(self):
        self.dates = pd.bdate_range("2020-01-01", "2020-05-01")
        self.prices = pd.DataFrame({"AAA": 10.}, index=self.dates)
        self.scores = pd.DataFrame({"AAA": [1.]}, index=self.dates[:1])
        self.config = dict(capital=1000, fee=.01, holding_days=90, top_n=1,
                           tranche_fraction=1., max_stock_weight=1., stale_limit=5,
                           strict_prices=False)

    def execute(self):
        return run(self.prices, self.scores, "2020-01-02", "2020-05-01", self.config)

    def test_budget_fees_and_exact_maturity(self):
        self.prices.loc["2020-04-01":] = 12.
        daily, trades, lots, _ = self.execute()
        units = 1000 / (10 * 1.01)
        self.assertAlmostEqual(trades.iloc[0].units, units)
        self.assertEqual(trades.iloc[0].date, pd.Timestamp("2020-01-02"))
        self.assertEqual(trades.iloc[1].date, pd.Timestamp("2020-04-01"))
        self.assertEqual(daily.loc["2020-03-31", "open_lots"], 1)
        self.assertAlmostEqual(daily.value.iloc[0], 1000/1.01)
        self.assertAlmostEqual(daily.value.iloc[-1], units*12*.99)
        self.assertAlmostEqual(1000+trades.cash_flow.sum(), daily.cash.iloc[-1])
        self.assertTrue(lots.empty)

    def test_missing_entry_is_cancelled(self):
        self.prices.loc["2020-01-02", "AAA"] = np.nan
        daily, trades, _, _ = self.execute()
        self.assertTrue(trades.empty)
        self.assertTrue(daily.cash.eq(1000).all())

    def test_missing_exit_is_not_invented(self):
        self.prices.loc["2020-04-01":"2020-04-02", "AAA"] = np.nan
        daily, trades, _, incidents = self.execute()
        self.assertEqual(trades.iloc[1].date, pd.Timestamp("2020-04-03"))
        self.assertEqual(daily.loc["2020-04-02", "open_lots"], 1)
        self.assertEqual(len(incidents), 2)

    def test_weekend_maturity(self):
        # Jan 6 + 90 days = Sunday Apr 5; exit Monday Apr 6.
        self.scores.index = pd.DatetimeIndex(["2020-01-03"])
        _, trades, _, _ = self.execute()
        self.assertEqual(trades.iloc[0].date, pd.Timestamp("2020-01-06"))
        self.assertEqual(trades.iloc[1].date, pd.Timestamp("2020-04-06"))

    def test_strict_stale_policy(self):
        self.prices.loc["2020-03-02":, "AAA"] = np.nan
        self.config["strict_prices"] = True
        with self.assertRaisesRegex(ValueError, "Unresolved"):
            self.execute()

    def test_maturity_after_end_stays_open(self):
        daily, trades, lots, _ = run(self.prices, self.scores, "2020-01-02", "2020-02-03", self.config)
        self.assertEqual(len(lots), 1)
        self.assertEqual(len(trades), 1)
        self.assertAlmostEqual(daily.value.iloc[-1], daily.cash.iloc[-1]+lots.units.iloc[0]*lots.mark.iloc[0])

    def test_first_day_fees_and_drawdown(self):
        daily, _, _, _ = self.execute()
        result = metrics(daily.value, 1000)
        self.assertAlmostEqual(result["max_drawdown"], .99/1.01-1)
        toy = pd.Series([100., 120., 90., 130.], index=pd.to_datetime(["2020-01-01", "2020-06-01", "2020-09-01", "2021-01-01"]))
        self.assertAlmostEqual(metrics(toy, 100)["max_drawdown"], -.25)
        self.assertAlmostEqual(metrics(toy, 100)["CAGR"], 1.3**(365.25/366)-1)

    def test_training_purges_unfinished_targets(self):
        panel = pd.DataFrame({"date": pd.to_datetime(["2014-06-01", "2014-12-01", "2015-01-02"]),
                              "label_end": pd.to_datetime(["2014-09-01", "2015-03-01", "2015-04-02"]),
                              "target": [.1, .2, .3]})
        self.assertEqual(training_rows(panel, pd.Timestamp("2015-01-01")).index.tolist(), [0])

    def test_added_features_are_causal(self):
        p = self.prices.rename(columns={"AAA": "adj_close"})
        p.iloc[10:] += np.arange(len(p)-10)[:, None]
        full = extra_features(p.copy())
        short = extra_features(p.iloc[:40].copy())
        pd.testing.assert_frame_equal(full.iloc[:40], short)


if __name__ == "__main__":
    unittest.main()
