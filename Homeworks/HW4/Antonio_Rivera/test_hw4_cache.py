"""Batched build == unbatched build, row for row (class DB).

Run from the repo root:
    uv run python -m unittest discover -s Homeworks/HW4/Antonio_Rivera -v
"""

import unittest

import numpy as np
import pandas as pd

import hw4_build_cache as hc
import hw4_data as hd
import hw4_features as hf


@unittest.skipUnless(hd.CLASS_DB.is_file(), "class database not downloaded")
class TestBatchedBuild(unittest.TestCase):

    def test_batches_match_one_shot(self):
        full, _, _ = hf.build_feature_panel(hd.CLASS_DB, end="2015-12-31")
        syms = sorted(full.symbol.unique())
        parts = [hf.build_feature_panel(hd.CLASS_DB, end="2015-12-31", symbols=list(b),
                                        cross_sectional=False)[0]
                 for b in np.array_split(syms, 3)]
        batched = pd.concat(parts, ignore_index=True)
        cols = hc.FINAL_COLS
        a = full[cols].sort_values(["date", "symbol"], ignore_index=True)
        b = batched[cols].sort_values(["date", "symbol"], ignore_index=True)
        pd.testing.assert_frame_equal(a, b, check_dtype=False)


if __name__ == "__main__":
    unittest.main()
