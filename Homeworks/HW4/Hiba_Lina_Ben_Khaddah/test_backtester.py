"""Hand-checkable accounting tests for backtester.py.

Run from this folder with:  python test_backtester.py
"""

import unittest

import numpy as np
import pandas as pd

from backtester import (
    Backtester,
    BacktestAnalytics,
    OrderManagementSystem,
    TopNScoreStrategy,
)


def run(dates, prices, scores, *, start, cash=1_000, fee_bps=0, top_n=1,
        cohort_fraction=1.0, holding_days=90, max_exit_delay=5):
    """Build tiny price/benchmark/score tables and run one backtest."""
    dates = pd.to_datetime(dates)
    long_prices = (
        pd.DataFrame(prices, index=dates).rename_axis("date").reset_index()
        .melt(id_vars="date", var_name="symbol", value_name="adj_close").dropna()
    )
    benchmark = pd.DataFrame({"date": dates, "adj_close": 100.0})
    scores = pd.DataFrame(scores, columns=["date", "symbol", "score"])
    strategy = TopNScoreStrategy(scores, top_n=top_n, cohort_fraction=cohort_fraction,
                                 holding_days=holding_days)
    backtester = Backtester(long_prices, benchmark, strategy,
                            OrderManagementSystem(cash, fee_bps),
                            max_exit_delay=max_exit_delay)
    return backtester.run_backtest(start, dates[-1])


class AccountingTests(unittest.TestCase):
    def test_fee_inclusive_buy_and_weekend_maturity(self):
        # Signal Fri Jan 31, buy Mon Feb 3 at 100 with a 1% fee.
        # Feb 3 + 90 days = Sun May 3, so the sale is Mon May 4 at 110.
        result = run(["2020-01-31", "2020-02-03", "2020-05-01", "2020-05-04"],
                     {"AAA": [100, 100, 108, 110]}, [("2020-01-31", "AAA", 1.0)],
                     start="2020-02-03", fee_bps=100)
        units = 1_000 / (100 * 1.01)
        buy, sell = result.fills.to_dict("records")
        self.assertAlmostEqual(buy["units"], units)
        self.assertAlmostEqual(buy["cash_after"], 0.0)
        self.assertEqual(sell["date"], pd.Timestamp("2020-05-04"))
        self.assertAlmostEqual(result.history["nav"].iloc[-1], units * 110 * 0.99)

    def test_same_symbol_twice_stays_two_lots(self):
        result = run(["2020-01-31", "2020-02-03", "2020-02-28", "2020-03-02",
                      "2020-05-04", "2020-06-01"],
                     {"AAA": [100] * 6},
                     [("2020-01-31", "AAA", 1.0), ("2020-02-28", "AAA", 1.0)],
                     start="2020-02-03", cohort_fraction=0.4)
        sells = result.fills.loc[result.fills["side"].eq("SELL")]
        self.assertEqual(sells["lot_id"].tolist(), [0, 1])
        self.assertEqual(sells["date"].tolist(),
                         [pd.Timestamp("2020-05-04"), pd.Timestamp("2020-06-01")])

    def test_cohort_budget_is_cut_to_available_cash(self):
        # 60% of NAV per cohort: the second cohort wants 600 but only 400 is left.
        result = run(["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-06"],
                     {"AAA": [100] * 4},
                     [("2020-01-01", "AAA", 1.0), ("2020-01-02", "AAA", 1.0)],
                     start="2020-01-02", cohort_fraction=0.6, holding_days=365)
        buys = result.fills.loc[result.fills["side"].eq("BUY"), "notional"].tolist()
        self.assertEqual(buys, [600, 400])
        self.assertEqual(len(result.budget_cuts), 1)
        self.assertGreaterEqual(result.history["cash"].min(), 0)

    def test_missing_exit_quote_delays_the_sale(self):
        # Due Jan 4, but there is no quote until Jan 5: sell then, one session late.
        result = run(["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04", "2020-01-05"],
                     {"AAA": [100, 100, 101, np.nan, 120]}, [("2020-01-01", "AAA", 1.0)],
                     start="2020-01-02", holding_days=2)
        sell = result.fills.loc[result.fills["side"].eq("SELL")].iloc[0]
        self.assertEqual(sell["date"], pd.Timestamp("2020-01-05"))
        self.assertEqual(result.diagnostics["event"].tolist(), ["delayed_exit"])

    def test_unresolved_lot_keeps_last_mark_and_is_reported(self):
        # The quote disappears after Jan 3: no sale is invented.
        result = run(pd.date_range("2020-01-01", periods=8),
                     {"AAA": [100, 100, 105] + [np.nan] * 5},
                     [("2020-01-01", "AAA", 1.0)],
                     start="2020-01-02", holding_days=2, max_exit_delay=2)
        self.assertTrue(result.fills["side"].eq("BUY").all())
        self.assertAlmostEqual(result.history["nav"].iloc[-1], 1_050)
        self.assertIn("unresolved_exit", result.diagnostics["event"].tolist())
        report = BacktestAnalytics(result, max_exit_delay=2).diagnostics()
        self.assertAlmostEqual(report["stale_share_of_final_nav"], 1.0)
        self.assertAlmostEqual(report["final_nav_if_stale_worth_zero"], 0.0)

    def test_held_price_jump_is_flagged(self):
        # 10 units bought at 100; the vendor price jumps to 2,000 (20x), then 2,100.
        result = run(["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-06"],
                     {"AAA": [100, 100, 2_000, 2_100]}, [("2020-01-01", "AAA", 1.0)],
                     start="2020-01-02", holding_days=365)
        jump = result.diagnostics.iloc[0]
        self.assertEqual(jump["event"], "held_price_jump")
        self.assertEqual(jump["lot_id"], 0)
        self.assertEqual(jump["previous_mark"], 100)
        # The official NAV keeps the vendor price; the event is reported as a limitation.
        self.assertAlmostEqual(result.history["nav"].iloc[-1], 21_000)

    def test_sale_cash_cannot_enlarge_an_order_planned_the_day_before(self):
        # All cash is invested on Jan 2. The Jan 3 signal is planned with $0 cash,
        # so the Jan 4 cohort is skipped even though the first lot sells on Jan 4.
        result = run(["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04"],
                     {"AAA": [100, 100, 100, 110], "BBB": [50] * 4},
                     [("2020-01-01", "AAA", 1.0), ("2020-01-03", "BBB", 1.0)],
                     start="2020-01-02", holding_days=2)
        self.assertEqual(result.fills["side"].tolist(), ["BUY", "SELL"])
        self.assertEqual(len(result.budget_cuts), 1)
        self.assertAlmostEqual(result.history.loc["2020-01-04", "cash"], 1_100)

    def test_nav_equals_cash_plus_holdings(self):
        result = run(["2020-01-31", "2020-02-03", "2020-02-28", "2020-03-02", "2020-05-04"],
                     {"AAA": [100, 101, 99, 98, 103], "BBB": [50, 52, 49, 51, 55]},
                     [("2020-01-31", "AAA", 2.0), ("2020-01-31", "BBB", 1.0),
                      ("2020-02-28", "BBB", 1.0)],
                     start="2020-02-03", top_n=2, cohort_fraction=0.5, fee_bps=5)
        h = result.history
        self.assertTrue(np.allclose(h["nav"], h["cash"] + h["holdings_value"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
