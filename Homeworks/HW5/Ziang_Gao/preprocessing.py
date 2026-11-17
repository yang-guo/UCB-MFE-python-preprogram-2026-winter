"""Feature contract and preprocessing for the HW4 return model.

Adapted from Lectures/modeling/course_data.py (Lectures 7 and 8): `QuantileClipper` and
`make_pipeline` are copied from that file. My HW4 notebook calls
`course_data.make_pipeline(model, feats, include_sector=False)`. The function below builds
the same pipeline for that case.

Why a copy: a saved sklearn pipeline stores the import path of each class, not its code.
The lecture file also imports duckdb and the data loaders. The service needs only the two
pieces below. This file must be next to app.py when the saved model loads, because the
artifact refers to `preprocessing.QuantileClipper`.
"""

import numpy as np
from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted, validate_data

MODEL_VERSION = "hw4-round1-ridge-fit-2016-03-31"

# Feature set "C momentum + financials" from hw4.ipynb, in the training order.
MOMENTUM_FEATURES = [
    "momentum_21",
    "momentum_63",
    "momentum_252_21",
    "mom63_vs_sector",
    "mom21_vs_sector",
]
FINANCIAL_FEATURES = [
    "net_margin",
    "operating_roa",
    "debt_to_assets",
    "cash_to_assets",
    "fcf_to_assets",
    "accruals_to_assets",
    "filing_age_days",
]
FEATURES = MOMENTUM_FEATURES + FINANCIAL_FEATURES


class QuantileClipper(TransformerMixin, OneToOneFeatureMixin, BaseEstimator):
    """Learn per-column tail cutoffs on training rows; preserve missing values."""

    def __init__(self, lower=0.01, upper=0.99):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        if not 0 <= self.lower < self.upper <= 1:
            raise ValueError("Require 0 <= lower < upper <= 1")
        a = validate_data(self, X, reset=True, dtype=float, ensure_all_finite="allow-nan")
        # All-missing columns are left unbounded and handled by the imputer.
        self.lower_bounds_ = np.full(a.shape[1], -np.inf)
        self.upper_bounds_ = np.full(a.shape[1], np.inf)
        observed = np.isfinite(a).any(axis=0)
        self.lower_bounds_[observed] = np.nanquantile(a[:, observed], self.lower, axis=0)
        self.upper_bounds_[observed] = np.nanquantile(a[:, observed], self.upper, axis=0)
        return self

    def transform(self, X):
        check_is_fitted(self, "lower_bounds_")
        a = validate_data(self, X, reset=False, dtype=float, ensure_all_finite="allow-nan")
        return np.clip(a, self.lower_bounds_, self.upper_bounds_)


def make_pipeline(model=None, numeric_features=None):
    """Clip, impute, scale, then the model. Numeric features only (no sector column)."""
    numeric_features = FEATURES if numeric_features is None else numeric_features
    numeric = Pipeline(
        [
            ("clip", QuantileClipper()),
            (
                "impute",
                SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
            ),
            ("scale", StandardScaler()),
        ]
    )
    preprocess = ColumnTransformer([("numeric", numeric, numeric_features)], remainder="drop")
    return Pipeline(
        [("preprocess", preprocess), ("model", Ridge(alpha=100) if model is None else model)]
    )
