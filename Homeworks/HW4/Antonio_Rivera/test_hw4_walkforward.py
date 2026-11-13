"""Leak tests for walk-forward scoring (class DB).

Run from the repo root:
    uv run python -m unittest discover -s Homeworks/HW4/Antonio_Rivera -v
"""

import unittest

import numpy as np
import pandas as pd

import hw4_data as hd
import hw4_features as hf
import hw4_walkforward as wf

T = pd.Timestamp


class TestFitDates(unittest.TestCase):

    def test_first_session_of_each_quarter(self):
        sessions = pd.bdate_range("2016-03-28", "2016-10-10")
        fits = wf.quarter_fit_dates(sessions, "2016-04-01", "2016-10-10")
        self.assertEqual(list(fits), [T("2016-04-01"), T("2016-07-01"), T("2016-10-03")])


@unittest.skipUnless(hd.CLASS_DB.is_file(), "class database not downloaded")
class TestWalkForwardLeaks(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.panel, _, _ = hf.build_feature_panel(hd.CLASS_DB, end="2015-12-31")
        cls.scores = wf.walk_forward_scores(cls.panel, "ridge", "F1_baseline", "2012-01-01", "2015-12-31")

    def test_each_score_comes_from_the_latest_fit_at_or_before_its_date(self):
        s = self.scores
        self.assertTrue((s.fit_date <= s.date).all())
        nxt = s.fit_date + pd.offsets.QuarterBegin(startingMonth=1)
        self.assertTrue((s.date < nxt + pd.Timedelta(days=7)).all())     # within its quarter

    def test_scores_cover_eligible_rows_only(self):
        keys = self.panel.loc[self.panel.eligible, ["date", "symbol"]]
        merged = self.scores.merge(keys, on=["date", "symbol"], how="left", indicator=True)
        self.assertTrue(merged._merge.eq("both").all())

    def test_future_labels_cannot_change_past_scores(self):
        # Poison every target whose 90-day window ends after 2013-07-01.
        p = self.panel.copy()
        late = p.label_end >= T("2013-07-01")
        p.loc[late, "target"] = np.where(p.loc[late].index % 2 == 0, 50.0, -0.99)
        poisoned = wf.walk_forward_scores(p, "ridge", "F1_baseline", "2012-01-01", "2015-12-31")
        before = self.scores.date < T("2013-07-01")
        pd.testing.assert_frame_equal(self.scores.loc[before].reset_index(drop=True),
                                      poisoned.loc[poisoned.date < T("2013-07-01")].reset_index(drop=True))
        # Sanity: later scores DO change, so the test could have failed.
        after = self.scores.date >= T("2014-01-02")
        self.assertFalse(np.allclose(self.scores.loc[after, "score"].to_numpy(),
                                     poisoned.loc[poisoned.date >= T("2014-01-02"), "score"].to_numpy()))


if __name__ == "__main__":
    unittest.main()
