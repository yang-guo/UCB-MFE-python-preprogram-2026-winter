"""Checks of the run reports and an end-to-end comparison with the lecture backtester."""

import importlib.util
from pathlib import Path
import tempfile
import unittest
import warnings

import numpy as np
import pandas as pd

import data
import hw4
import model
import report
from backtest import Account, FEE_RATE, TopNStrategy, run_backtest
from test_backtest import SIGNAL, flat_quotes


CLASS_DATABASE = data.MODEL_DIR / "data" / "equities_class.duckdb"
NO_BREAKS = pd.DataFrame(columns=["symbol", "break_date", "vendor_last_date", "vendor_last_close"])


class ReportTests(unittest.TestCase):
    def frozen_run(self):
        quotes = flat_quotes("2024-04-15", symbols=("A",))
        quotes.loc["2024-04-05":, "A"] = np.nan
        return quotes, run_backtest(quotes, TopNStrategy({SIGNAL: ["A"]}, sleeve_fraction=0.25), SIGNAL)

    def test_frozen_holding_is_listed_with_both_valuations(self):
        quotes, result = self.frozen_run()
        value = 250_000 / (1 + FEE_RATE)
        nav = 750_000 + value
        breaks = pd.DataFrame([{"symbol": "A", "break_date": pd.Timestamp("2024-04-05"),
                                "vendor_last_date": pd.Timestamp("2024-04-15"), "vendor_last_close": 20.0}])
        lines = report.frozen_holdings(result, quotes.index, breaks)
        self.assertIn(f"${value:,.0f} in 1 lots, {value / nav:.2%} of NAV", lines[0])
        # The vendor's last quote of 20 is a fifth of the last valid quote of 100.
        self.assertIn(f"${nav - value + value / 5:,.0f}", lines[2])
        self.assertIn("unresolved price break on 2024-04-05", lines[-2])
        # The last valid quote is April 4 and the run ends April 15, eleven days later.
        self.assertIn("| 2024-04-04 | 11 |", lines[-2])
        self.assertIn("no quote", report.frozen_holdings(result, quotes.index, NO_BREAKS)[-2])

    def test_label_coverage_counts_only_outcomes_that_were_due(self):
        panel = pd.DataFrame({"entry_date": pd.to_datetime(["2024-01-02", "2024-01-02", "2024-03-01"]),
                              "target": [0.05, np.nan, np.nan]})
        # Two outcomes were due by April 30 and one of them is observed. The March entry is not due yet.
        self.assertIn("Of 2 candidate signals", report.label_coverage(panel, "2024-04-30"))
        self.assertIn("50.00% have an observed outcome", report.label_coverage(panel, "2024-04-30"))

    def test_no_frozen_holding_reports_zero(self):
        quotes = flat_quotes("2024-02-15")
        result = run_backtest(quotes, TopNStrategy({SIGNAL: ["A", "B"]}), SIGNAL)
        self.assertEqual(report.frozen_holdings(result, quotes.index, NO_BREAKS),
                         ["Holdings carried at an older price on the final day: $0 in 0 lots, 0.00% of NAV.", ""])

    def test_reconciliation_reads_the_ledger(self):
        quotes = flat_quotes("2024-06-28")
        rankings = {SIGNAL: ["A", "B"], pd.Timestamp("2024-01-31"): ["B"]}
        result = run_backtest(quotes, TopNStrategy(rankings), SIGNAL)
        checks = report.reconciliation(result, pd.Series(1_000_000.0, index=quotes.index)).Result
        self.assertEqual(checks["First and last session"], "2024-01-08 to 2024-06-28")
        self.assertEqual(checks["Largest gap between cash and the fill ledger"], "$0.000000")
        self.assertEqual(checks["Buys filled after their signal date"], "3 of 3")
        self.assertEqual(checks["Sales on or after maturity"], "3 of 3")
        self.assertEqual(checks["Buys given a 90-calendar-day maturity"], "3 of 3")


@unittest.skipUnless(CLASS_DATABASE.is_file(), "needs the class database from bCourses")
class ClassDatabaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cls.panel, cls.market, cls.quotes, _ = data.load_research_data(CLASS_DATABASE, data.DEV_END, "2013-01-01")
        cls.first_signal = cls.market.index[cls.market.index.get_loc(cls.quotes.index[0]) - 1]
        cls.scores = model.composite_scores(model.add_ranks(cls.panel.loc[cls.panel.date.ge(cls.first_signal)]))

    def test_daily_nav_matches_the_lecture_backtester(self):
        spec = importlib.util.spec_from_file_location("lecture_backtest", data.MODEL_DIR / "backtest.py")
        lecture = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(lecture)
        ours = run_backtest(self.quotes, TopNStrategy(model.to_rankings(self.scores), 20, 1 / 3), self.first_signal,
                            Account(fee_rate=FEE_RATE))
        prices, _, _, benchmark = data.course.load_data(CLASS_DATABASE)
        strategy = lecture.PredictedScoreStrategy(self.scores[["date", "symbol", "score"]], top_n=20,
                                                  cohort_fraction=1 / 3, holding_days=90)
        account = lecture.OrderManagementSystem(initial_cash=1_000_000, fee_bps=FEE_RATE * 10_000)
        theirs = lecture.Backtester(prices, benchmark, strategy, account).run_backtest("2013-01-01", "2015-12-31")
        # The lecture history starts with one extra row of initial capital.
        np.testing.assert_allclose(ours.daily.nav, theirs.history.nav.iloc[1:], atol=1e-6)
        np.testing.assert_allclose(ours.daily.cash, theirs.history.cash.iloc[1:], atol=1e-6)
        self.assertEqual(len(ours.fills), len(theirs.fills))
        self.assertAlmostEqual(ours.fills.fee.sum(), theirs.fills.fee.sum(), places=6)

    def test_final_test_pipeline_runs_on_development_dates(self):
        with tempfile.TemporaryDirectory() as folder, warnings.catch_warnings():
            warnings.simplefilter("ignore")
            hw4.run_test(CLASS_DATABASE, Path(folder), start="2013-01-01", end=data.DEV_END)
            text = (Path(folder) / "results.md").read_text()
            self.assertTrue((Path(folder) / "cumulative_returns.png").is_file())
        headings = ["## Settings", "## Performance", "## Holdings carried at an older price", "## Checks on this run"]
        for heading in headings:
            self.assertIn(heading, text)
        self.assertIn("2013-01-02 to 2015-12-31", text)


if __name__ == "__main__":
    unittest.main()
