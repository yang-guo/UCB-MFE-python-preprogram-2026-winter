"""The HW4 feature contract and the custom transformer needed to load the artifact.

HW4 model (see Homeworks/HW4/Nirmit_Ghughu, branch ``hw_4_Nirmit_Ghughu``):
    Ridge(alpha=1000) trained on a *rank* target, refit quarterly.

HW4 preprocessing has two stages:

1. **Per-date percentile ranking** (``model.add_rank_features`` in HW4).  On every
   weekly signal date each raw feature is ranked across that date's tradable
   stocks: ``rank(pct=True) - 0.5``, so values lie in (-0.5, 0.5].  Ranking needs
   the whole cross-section of the day, so it cannot be computed from one
   stock's request; it is done upstream by the HW4 data pipeline and the API
   receives the ranked values.  It has no fitted state.
2. **Ridge design matrix** (``model._design(kind="ridge")`` in HW4): missing rank ->
   0 (the cross-sectional median), one ``<feature>_missing`` indicator per
   feature, and one-hot sector dummies.  The sector list is learned when the
   model is fitted and stored in the artifact.  ``HW4RidgeDesign`` reproduces
   this stage exactly so the served pipeline is ``HW4RidgeDesign -> Ridge``.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# Same names and order as HW4 ``data.NUMERIC_FEATURES``.
NUMERIC_FEATURES = [
    "mom_5",
    "mom_21",
    "mom_63",
    "mom_126",
    "mom_252_21",
    "vol_63",
    "max_ret_21",
    "drawdown_252",
    "beta_252",
    "log_dollar_vol_63",
    "gross_profitability",
    "operating_roa",
    "net_margin",
    "debt_to_assets",
    "cash_to_assets",
    "fcf_to_assets",
    "accruals_to_assets",
    "asset_growth",
    "earnings_yield",
    "book_to_market",
    "filing_age_days",
    "log_mcap",
]
CATEGORICAL_FEATURES = ["sector"]
INPUT_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


class HW4RidgeDesign(TransformerMixin, BaseEstimator):
    """Ranked features + sector -> the exact matrix the HW4 Ridge was fitted on.

    ``sectors`` is the list of sector columns seen at fit time (sorted, as
    ``pd.get_dummies`` produced them).  An unseen sector would get all-zero
    dummies, but the API rejects unseen sectors before they reach the model.
    """

    def __init__(self, numeric_features=None, sectors=None):
        self.numeric_features = numeric_features
        self.sectors = sectors

    def fit(self, X=None, y=None):
        self.feature_names_out_ = np.array(
            list(self.numeric_features)
            + [f"{c}_missing" for c in self.numeric_features]
            + list(self.sectors)
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in INPUT_COLUMNS if c not in X.columns]
        if missing:
            raise ValueError(f"missing input columns: {missing}")
        # HW4 stored ranks as float32; keep that dtype so predictions match exactly.
        numeric = X[list(self.numeric_features)].astype(np.float32)
        indicators = numeric.isna().astype(np.float32).add_suffix("_missing")
        dummies = pd.get_dummies(X["sector"], dtype=np.float32).reindex(
            columns=list(self.sectors), fill_value=0.0
        )
        out = pd.concat([numeric.fillna(0.0), indicators, dummies], axis=1)
        return out[self.feature_names_out_]

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_out_
