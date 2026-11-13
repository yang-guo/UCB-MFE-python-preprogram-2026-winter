"""Check the expanding training window at month boundaries."""
import unittest

import numpy as np
import pandas as pd

from feature_engineering import annual_split, make_ridge_pipeline, walk_forward_forecasts


class RetrainingTests(unittest.TestCase):
    def setUp(self):
        self.panel = pd.DataFrame({
            "symbol": "A", "eligible": True,
            "date": pd.to_datetime(["2012-01-02", "2012-10-01", "2012-11-01", "2013-01-02", "2013-02-01", "2013-02-04"]),
            "label_end": pd.to_datetime(["2012-04-03", "2013-01-01", "2013-01-31", "2013-04-03", "2013-05-03", None]),
            "target": [0.1, 0.2, 0.3, 0.4, 0.5, np.nan],
            "x": [1.0, 2, 3, 4, 5, 6],
        })

    def test_monthly_fit_adds_only_matured_outcomes(self):
        forecast, log = walk_forward_forecasts(self.panel, ["x"], end="2013-02-28")
        self.assertEqual(log.training_rows.tolist(), [2, 3])
        self.assertEqual(log.new_training_rows.tolist(), [2, 1])
        self.assertTrue(log.last_training_outcome.lt(log.fit_cutoff).all())
        self.assertEqual(len(forecast), 3)  # Keep the row with no future target.
        self.assertTrue(forecast.loc[forecast.date.ge("2013-02-01"), "fit_cutoff"].eq(pd.Timestamp("2013-02-01")).all())
        boundary = self.panel.copy()
        boundary.loc[2, "label_end"] = pd.Timestamp("2013-02-01")
        _, log = walk_forward_forecasts(boundary, ["x"], end="2013-02-28")
        self.assertEqual(log.training_rows.tolist(), [2, 2])

    def test_unavailable_outcomes_cannot_change_earlier_forecasts(self):
        original, _ = walk_forward_forecasts(self.panel, ["x"], end="2013-02-28")
        changed = self.panel.copy()
        changed.loc[changed.date.ge("2013-01-02"), "target"] = 99.0
        future, _ = walk_forward_forecasts(changed, ["x"], end="2013-02-28")
        pd.testing.assert_frame_equal(original, future)
        changed.loc[2, "target"] = 5.0  # This outcome becomes available before February.
        updated, _ = walk_forward_forecasts(changed, ["x"], end="2013-02-28")
        self.assertEqual(original.prediction.iloc[0], updated.prediction.iloc[0])
        self.assertFalse(np.allclose(original.prediction.iloc[1:], updated.prediction.iloc[1:]))

    def test_annual_mode_matches_existing_annual_fit(self):
        train, signals = annual_split(self.panel, 2013)
        pipeline = make_ridge_pipeline(feature_names=["x"])
        pipeline.fit(train[["x"]], train.target)
        forecast, log = walk_forward_forecasts(self.panel, ["x"], frequency="annual")
        self.assertEqual(len(log), 1)
        np.testing.assert_allclose(forecast.prediction, pipeline.predict(signals[["x"]]))


if __name__ == "__main__":
    unittest.main()
