"""Run with: python -m unittest discover -s Homeworks/HW4/Mauricio_Gutierrez -p test_feature_engineering.py

Use small cases with known answers to check time boundaries and missing data.
No database is needed. Business days form the synthetic calendar, not a real
exchange calendar. The notebook reads the database's actual market calendar.
"""
import unittest

import numpy as np
import pandas as pd

from feature_engineering import (
    PRICE_FEATURES, annual_split, apply_membership, make_labels,
    make_ridge_pipeline, price_features,
)


class FeatureTimingTests(unittest.TestCase):
    def setUp(self):
        self.calendar = pd.bdate_range("2010-01-01", periods=360)
        self.prices = pd.DataFrame({
            "symbol": "A", "date": self.calendar,
            "adj_close": 100 * 1.001 ** np.arange(len(self.calendar)),
        })

    def test_known_returns_and_future_invariance(self):
        features = price_features(self.prices, self.calendar)
        row = features.iloc[300]
        self.assertAlmostEqual(row.momentum_21, 1.001 ** 21 - 1)
        self.assertAlmostEqual(row.momentum_63, 1.001 ** 63 - 1)
        self.assertAlmostEqual(row.momentum_252_21, 1.001 ** 231 - 1)
        self.assertAlmostEqual(row.drawdown_252, 0)
        self.assertAlmostEqual(row.volatility_63, 0)
        changed = self.prices.copy()
        changed.loc[changed.index > 300, "adj_close"] *= 8
        changed_features = price_features(changed, self.calendar)
        columns = ["date", "symbol", *PRICE_FEATURES, "eligible"]
        pd.testing.assert_frame_equal(features.loc[:300, columns], changed_features.loc[:300, columns])
        truncated = price_features(self.prices.iloc[:301], self.calendar[:301])
        pd.testing.assert_frame_equal(features.loc[:300, columns], truncated[columns])

    def test_missing_session_does_not_become_daily_return(self):
        prices = self.prices.drop(index=100)
        features = price_features(prices, self.calendar).set_index("date")
        # The quote after the gap does not have 63 valid consecutive returns.
        self.assertFalse(features.loc[self.calendar[101], "eligible"])
        self.assertTrue(pd.isna(features.loc[self.calendar[101], "volatility_63"]))
        self.assertFalse(features.loc[self.calendar[163], "eligible"])
        self.assertTrue(features.loc[self.calendar[164], "eligible"])

    def test_next_session_entry_and_calendar_day_exit(self):
        # Friday signal -> Monday entry. Ninety days later is Sunday.
        panel = pd.DataFrame({"symbol": ["A"], "date": [pd.Timestamp("2010-01-01")]})
        result = make_labels(panel, self.prices, self.calendar).iloc[0]
        self.assertEqual(result.entry_date, pd.Timestamp("2010-01-04"))
        self.assertEqual(result.scheduled_exit, pd.Timestamp("2010-04-04"))
        self.assertEqual(result.label_end, pd.Timestamp("2010-04-05"))
        quotes = self.prices.set_index("date").adj_close
        self.assertAlmostEqual(result.target, quotes.loc["2010-04-05"] / quotes.loc["2010-01-04"] - 1)

    def test_missing_quotes_and_unfinished_targets_preserve_rows(self):
        panel = pd.DataFrame({"symbol": ["A"], "date": [pd.Timestamp("2010-01-01")], "eligible": [True]})
        for missing_date in ["2010-01-04", "2010-04-05"]:
            prices = self.prices.loc[self.prices.date.ne(missing_date)]
            result = make_labels(panel, prices, self.calendar)
            self.assertEqual(len(result), 1)
            self.assertTrue(result.eligible.iloc[0])
            self.assertTrue(pd.isna(result.target.iloc[0]))
        late = panel.assign(date=self.calendar[-2])
        result = make_labels(late, self.prices, self.calendar)
        self.assertTrue(result.eligible.iloc[0])
        self.assertTrue(pd.isna(result.target.iloc[0]))

    def test_entry_quarter_and_selection_availability(self):
        calendar = pd.bdate_range("2010-06-29", "2010-07-02")
        panel = pd.DataFrame({"symbol": ["A", "A"], "date": calendar[:2], "eligible": True})
        membership = pd.DataFrame({
            "symbol": ["A"], "quarter_start": pd.to_datetime(["2010-07-01"]),
            "quarter_end": pd.to_datetime(["2010-09-30"]),
            "selection_date": pd.to_datetime(["2010-06-30"]), "universe_rank": [1],
        })
        result = apply_membership(panel, membership, calendar)
        self.assertEqual(result.eligible.tolist(), [False, True])
        membership["selection_date"] = pd.Timestamp("2010-07-01")
        with self.assertRaises(ValueError):
            apply_membership(panel, membership, calendar)

    def test_split_excludes_unfinished_training_but_keeps_unknown_forecasts(self):
        panel = pd.DataFrame({
            "symbol": "A", "eligible": True,
            "date": pd.to_datetime(["2011-01-03", "2012-12-03", "2013-01-02", "2013-12-31"]),
            "label_end": pd.to_datetime(["2011-04-04", "2013-03-04", "2013-04-03", None]),
            "target": [0.1, 0.2, 0.3, np.nan],
        })
        train, signals = annual_split(panel, 2013)
        self.assertEqual(train.index.tolist(), [0])
        self.assertEqual(signals.index.tolist(), [2, 3])

    def test_preprocessing_ignores_future_audit_fields_and_does_not_refit(self):
        train = pd.DataFrame({name: np.arange(40, dtype=float) / 10 for name in PRICE_FEATURES})
        train.loc[0, "momentum_252_21"] = np.nan
        train["target"] = np.linspace(-0.1, 0.3, len(train))
        model = make_ridge_pipeline()
        model.fit(train, train.target)
        numeric = model.named_steps["preprocess"].named_transformers_["numeric"]
        before_mean = numeric.named_steps["scale"].mean_.copy()
        before_bounds = numeric.named_steps["clip"].upper_bounds_.copy()
        predictions = model.predict(train)
        changed_audit = train.assign(target=1e9, exit_price=1e12)
        np.testing.assert_array_equal(predictions, model.predict(changed_audit))
        extreme = train.copy()
        extreme[PRICE_FEATURES] = 1e6
        model.predict(extreme)
        np.testing.assert_array_equal(before_mean, numeric.named_steps["scale"].mean_)
        np.testing.assert_array_equal(before_bounds, numeric.named_steps["clip"].upper_bounds_)


if __name__ == "__main__":
    unittest.main()
