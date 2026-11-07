"""Timing and accounting tests, with small independently checkable cases."""

import unittest

import numpy as np
import pandas as pd

from backtester import simulate, UnresolvedPriceGap
from research import (
    FEATURES, KalmanRegression, balanced_sample, build_panel, course, purged_split,
    read_config,
)


class ResearchChecks(unittest.TestCase):
    def test_test_period_labels_cannot_enter_research(self):
        with self.assertRaises(ValueError):
            build_panel("2016-04-01", "2010-01-01", labels=True)

    def test_purge_uses_actual_outcome_end_and_whole_dates(self):
        calendar = pd.bdate_range("2011-01-03", "2012-12-31")
        frame = pd.DataFrame({
            "date": pd.to_datetime(["2011-01-03", "2011-11-01", "2012-01-03", "2012-01-03"]),
            "entry_date": pd.to_datetime(["2011-01-04", "2011-11-02", "2012-01-04", "2012-01-04"]),
            "label_end": pd.to_datetime(["2011-04-04", "2012-01-31", "2012-04-03", "2012-04-03"]),
            "target": [0.1, .2, .3, .4],
        })
        train, valid, boundary = purged_split(frame, calendar, 2012, "2013-01-01")
        self.assertEqual(list(train.index), [0])
        self.assertEqual(list(valid.index), [2, 3])
        self.assertLess(train.label_end.max(), boundary)

    def test_future_filing_and_future_price_do_not_change_past_inputs(self):
        calendar = pd.bdate_range("2010-01-01", periods=320)
        prices = pd.DataFrame({"symbol": "A", "date": calendar,
                               "adj_close": np.linspace(10, 20, 320)})
        before = course.price_features(prices, calendar)
        changed = prices.copy()
        changed.loc[changed.index >= 280, "adj_close"] *= 20
        after = course.price_features(changed, calendar)
        pd.testing.assert_frame_equal(
            before.loc[before.date < calendar[280]],
            after.loc[after.date < calendar[280]],
        )
        events = pd.DataFrame({
            "symbol": ["A", "A"],
            "fiscal_date": pd.to_datetime(["2009-12-31", "2010-03-31"]),
            "filing_date": pd.to_datetime(["2010-02-01", "2010-05-01"]),
            "available_date": pd.to_datetime(["2010-02-02", "2010-05-02"]),
            **{feature: [1.0, 99.0] for feature in course.FINANCIAL_FEATURES},
        })
        sample = pd.DataFrame({"symbol": ["A"], "date": pd.to_datetime(["2010-04-01"])})
        universe = pd.DataFrame({"symbol": ["A"], "sector": ["test"]})
        joined = course.join_financials(sample, events, universe)
        self.assertEqual(joined.net_margin.iloc[0], 1)
        self.assertLessEqual(joined.available_date.iloc[0], sample.date.iloc[0])

    def test_feature_allowlist_excludes_labels_and_snapshot_status(self):
        for field in ["target", "entry_price", "exit_price", "label_end",
                      "is_active_snapshot", "sector"]:
            self.assertNotIn(field, FEATURES)

    def test_sampling_does_not_depend_on_target_values(self):
        frame = pd.DataFrame({"date": np.repeat(pd.date_range("2010-01-01", periods=3), 20),
                              "symbol": [f"S{i}" for i in range(60)], "target": np.arange(60)})
        changed = frame.copy()
        changed.target = -changed.target
        self.assertEqual(
            list(balanced_sample(frame, 12, 7).index),
            list(balanced_sample(changed, 12, 7).index),
        )

    def test_kalman_predictions_do_not_update_state(self):
        X = np.arange(30, dtype=float).reshape(-1, 1) / 30
        model = KalmanRegression().fit(X, 2 * X[:, 0], dates=np.repeat([0, 1, 2], 10))
        before = model.coef_.copy()
        model.predict(X)
        np.testing.assert_array_equal(before, model.coef_)


class AccountingChecks(unittest.TestCase):
    def setUp(self):
        self.calendar = pd.bdate_range("2016-03-31", "2016-07-08")
        self.prices = pd.DataFrame({"A": 10.0}, index=self.calendar)
        self.scores = pd.DataFrame({"date": [pd.Timestamp("2016-03-31")],
                                    "symbol": ["A"], "prediction": [0.1]})
        self.config = read_config()["strategy"]
        self.config.update(initial_capital=1000, monthly_nav_budget=1,
                           maximum_name_weight=1, number_of_stocks=1)

    def test_fee_cash_maturity_and_no_reinvestment_without_signal(self):
        result = simulate(self.prices, self.calendar, self.scores,
                          "2016-04-01", "2016-07-08", self.config)
        fills = result["fills"]
        self.assertEqual(list(fills.side), ["buy", "sell"])
        self.assertEqual(fills.iloc[0].date, pd.Timestamp("2016-04-01"))
        self.assertEqual(fills.iloc[1].date, pd.Timestamp("2016-06-30"))
        self.assertAlmostEqual(fills.iloc[0].units, 1000 / (10 * 1.001))
        self.assertAlmostEqual(result["history"].iloc[-1].nav, 1000 * .999 / 1.001)
        self.assertGreaterEqual(result["history"].cash.min(), -1e-8)

    def test_open_positions_are_included_at_test_end(self):
        result = simulate(self.prices, self.calendar, self.scores,
                          "2016-04-01", "2016-04-08", self.config)
        self.assertAlmostEqual(result["history"].iloc[-1].nav, 1000 / 1.001)
        self.assertEqual(len(result["open_lots"]), 1)

    def test_weekend_maturity_exits_next_session(self):
        self.config["holding_days"] = 1
        result = simulate(self.prices, self.calendar, self.scores,
                          "2016-04-01", "2016-04-08", self.config)
        self.assertEqual(result["fills"].iloc[1].date, pd.Timestamp("2016-04-04"))

    def test_two_orders_share_cash_and_fees(self):
        self.config["number_of_stocks"] = 2
        scores = pd.concat([self.scores, self.scores.assign(symbol="B")])
        prices = self.prices.assign(B=20.)
        result = simulate(prices, self.calendar, scores,
                          "2016-04-01", "2016-04-08", self.config)
        self.assertAlmostEqual(result["orders"].budget.sum(), 1000)
        self.assertAlmostEqual(result["fills"].gross.sum() + result["fills"].fee.sum(), 1000)
        self.assertGreaterEqual(result["history"].cash.min(), -1e-8)

    def test_future_fill_price_does_not_change_planned_dollar_budget(self):
        changed = self.prices.copy()
        changed.loc["2016-04-01", "A"] = 20
        result = simulate(changed, self.calendar, self.scores,
                          "2016-04-01", "2016-04-08", self.config)
        self.assertAlmostEqual(result["orders"].iloc[0].budget, 1000)
        self.assertAlmostEqual(result["fills"].iloc[0].units, 1000 / (20 * 1.001))

    def test_unresolved_held_gap_stops_instead_of_assuming_recovery(self):
        broken = self.prices.copy()
        broken.loc["2016-04-04":"2016-04-20", "A"] = np.nan
        with self.assertRaises(UnresolvedPriceGap):
            simulate(broken, self.calendar, self.scores,
                     "2016-04-01", "2016-04-20", self.config)

    def test_allocator_rejects_future_outcome_columns(self):
        scores = self.scores.assign(target=999)
        with self.assertRaises(ValueError):
            simulate(self.prices, self.calendar, scores,
                     "2016-04-01", "2016-04-08", self.config)

    def test_verified_cash_claim_is_valued_but_restricted_until_maturity(self):
        actions = {"A": {"kind": "cash", "effective_date": "2016-04-05",
                          "adjusted_cash_per_unit": 12.0}}
        result = simulate(self.prices, self.calendar, self.scores,
                          "2016-04-01", "2016-07-08", self.config, actions)
        history = result["history"].set_index("date")
        expected = 1000 / (10 * 1.001) * 12
        self.assertAlmostEqual(history.loc["2016-04-05", "nav"], expected)
        self.assertAlmostEqual(history.loc["2016-04-05", "cash"], 0)
        self.assertAlmostEqual(history.loc["2016-06-30", "cash"], expected)
        self.assertEqual(result["fills"].iloc[-1].side, "cash_claim_release")

    def test_stock_conversion_preserves_original_maturity(self):
        actions = {"A": {"kind": "stock", "effective_date": "2016-04-05",
                          "successor": "B", "adjusted_unit_ratio": 2,
                          "successor_initial_mark": 6}}
        result = simulate(self.prices.assign(B=6), self.calendar, self.scores,
                          "2016-04-01", "2016-07-08", self.config, actions)
        sale = result["fills"].iloc[-1]
        self.assertEqual(sale.symbol, "B")
        self.assertEqual(sale.date, pd.Timestamp("2016-06-30"))
        self.assertAlmostEqual(sale.units, 2 * 1000 / (10 * 1.001))

    def test_vendor_ghost_quote_does_not_allow_new_predecessor_entry(self):
        actions = {"A": {"kind": "cash", "effective_date": "2016-04-01",
                          "adjusted_cash_per_unit": 12}}
        result = simulate(self.prices, self.calendar, self.scores,
                          "2016-04-01", "2016-04-08", self.config, actions)
        self.assertTrue(result["fills"].empty)
        self.assertAlmostEqual(result["history"].iloc[-1].cash, 1000)


if __name__ == "__main__":
    unittest.main(verbosity=2)
