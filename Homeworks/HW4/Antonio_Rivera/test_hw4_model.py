"""Unit tests for hw4_model.py (selection rule, purge, train-only preprocessing).

Run from the repo root:
    uv run python -m unittest discover -s Homeworks/HW4/Antonio_Rivera -v
"""

import unittest

import numpy as np
import pandas as pd

import hw4_data as hd
import hw4_features as hf
import hw4_model as hm


def toy_summary(ics, ses):
    """ics/ses for ridge F1,F2,F3, hgb F1,F2,F3 (the original 6-run grid)."""
    order = [k for k in hm.SIMPLICITY_ORDER if k[1] != "F4_plus_4ab"]
    return pd.DataFrame({"model": [m for m, _ in order], "features": [f for _, f in order],
                         "mean_IC": ics, "se_IC": ses})


class TestSelectionRule(unittest.TestCase):

    def test_clear_winner_is_picked(self):
        # hgb F3 best by far: 0.10 - 0.01 = 0.09 bar; nobody else reaches it.
        s = toy_summary([0.01, 0.02, 0.03, 0.04, 0.05, 0.10], [0.01] * 6)
        self.assertEqual(hm.select(s)["features"], "F3_ranks")
        self.assertEqual(hm.select(s)["model"], "hgb")

    def test_simpler_within_one_se_wins(self):
        # Best is hgb F2 at 0.06, SE 0.02 -> bar 0.04. Ridge F2 (0.045) is the simplest above it.
        s = toy_summary([0.03, 0.045, 0.05, 0.02, 0.06, 0.01], [0.02] * 6)
        pick = hm.select(s)
        self.assertEqual((pick["model"], pick["features"]), ("ridge", "F2_plus_AB"))

    def test_best_is_simplest(self):
        s = toy_summary([0.08, 0.02, 0.02, 0.02, 0.02, 0.02], [0.01] * 6)
        self.assertEqual((hm.select(s)["model"], hm.select(s)["features"]), ("ridge", "F1_baseline"))


@unittest.skipUnless(hd.CLASS_DB.is_file(), "class database not downloaded")
class TestFoldsAndPreprocessing(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        panel, _, _ = hf.build_feature_panel(hd.CLASS_DB, end=hm.DEV_END)
        cls.frame = hm.dev_frame(panel)
        cls.folds = hd.cd.date_folds(cls.frame, hm.VALIDATION_YEARS, train_start=hm.DEV_START)

    def test_dev_frame_is_eligible_and_inside_development(self):
        f = self.frame
        self.assertTrue(f.eligible.all())
        self.assertGreaterEqual(f.date.min(), pd.Timestamp(hm.DEV_START))
        self.assertLessEqual(f.label_end.max(), pd.Timestamp(hm.DEV_END))

    def test_every_training_label_ends_before_validation_starts(self):
        for tr, va in self.folds:
            self.assertLess(self.frame.iloc[tr].label_end.max(), self.frame.iloc[va].date.min())

    def test_preprocessing_learned_from_train_rows_only(self):
        tr, va = self.folds[0]
        train = self.frame.iloc[tr]
        cols = hm.FEATURE_SETS["F1_baseline"]
        pipe = hd.cd.make_pipeline(hm.make_model("ridge"), cols).fit(train, train.target)
        clip = pipe.named_steps["preprocess"].named_transformers_["numeric"].named_steps["clip"]
        expected = np.nanquantile(train[cols].to_numpy(float), 0.01, axis=0)
        np.testing.assert_allclose(clip.lower_bounds_, expected)
        # Sanity: the full-sample bounds differ, so this test can actually fail.
        full = np.nanquantile(self.frame[cols].to_numpy(float), 0.01, axis=0)
        self.assertFalse(np.allclose(full, expected))

        # Imputer fill values = medians of the CLIPPED TRAINING rows only.
        imp = pipe.named_steps["preprocess"].named_transformers_["numeric"].named_steps["impute"]
        def clipped_median(rows):
            return np.nanmedian(clip.transform(rows[cols]), axis=0)
        np.testing.assert_allclose(imp.statistics_, clipped_median(train))
        self.assertFalse(np.allclose(imp.statistics_, clipped_median(self.frame)))


if __name__ == "__main__":
    unittest.main()
