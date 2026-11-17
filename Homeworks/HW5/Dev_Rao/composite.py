"""The HW4 model as an importable sklearn estimator.

My HW4 final experiment froze a fixed-sign multi-factor composite (HW4
frozen_config.json, sha256 140d39c7...). This file ports
`composite_score` and `COMPOSITE_THEMES` from Homeworks/HW4/Dev_Rao/hw4/model.py
(branch hw_4_dev_rao, commit 10d27ee) into a small sklearn estimator so it can be
pickled inside a Pipeline and loaded by the service and the Docker image.
The math is unchanged:

    theme score = mean over the theme's AVAILABLE features of sign * (rank - 0.5)
    composite   = mean over the AVAILABLE theme scores (0.0 if every feature is missing)

Nothing is learned: `fit` only checks and records the column contract. The inputs
are same-date cross-sectional percentile ranks, which HW4 computed upstream among
that date's eligible stocks; ranking needs the whole cross-section, so it cannot run
inside a one-row request.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted

# signs from the literature (see the HW4 README "models and validation" section)
COMPOSITE_THEMES = {
    "value": {"r_ey": 1, "r_bm": 1, "r_sp": 1, "r_fcfy": 1},
    "quality": {"r_gp_a": 1, "r_roe": 1, "r_roa": 1, "r_accruals_a": -1},
    "investment": {"r_asset_growth": -1, "r_share_iss": -1},
    "momentum": {"r_mom_12_1": 1, "r_sector_mom_126": 1, "r_ni_chg_a": 1},
    "low_risk": {"r_ivol_63": -1, "r_max_ret_21": -1},
}
# training/serving column order = theme order above
FEATURES = [c for spec in COMPOSITE_THEMES.values() for c in spec]


class CompositeScorer(RegressorMixin, BaseEstimator):
    """Fixed-sign composite of percentile-rank features (no fitted parameters)."""

    def __init__(self, themes=None):
        self.themes = themes

    def _themes(self):
        return COMPOSITE_THEMES if self.themes is None else self.themes

    def fit(self, X, y=None):
        cols = [c for spec in self._themes().values() for c in spec]
        if not isinstance(X, pd.DataFrame) or list(X.columns) != cols:
            raise ValueError(f"expected a DataFrame with columns {cols}")
        self.feature_names_in_ = np.asarray(cols, dtype=object)
        self.n_features_in_ = len(cols)
        return self

    def predict(self, X):
        check_is_fitted(self, "feature_names_in_")
        if not isinstance(X, pd.DataFrame) or list(X.columns) != list(self.feature_names_in_):
            raise ValueError(
                f"expected a DataFrame with columns {list(self.feature_names_in_)}"
            )
        X = X.astype("float64")
        themes = []
        for spec in self._themes().values():
            parts = [(X[c] - 0.5) * s for c, s in spec.items()]
            themes.append(pd.concat(parts, axis=1).mean(axis=1))
        return pd.concat(themes, axis=1).mean(axis=1).fillna(0.0).to_numpy()
