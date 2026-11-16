"""HW4's fitted numeric preprocessing and Ridge pipeline.

QuantileClipper and the pipeline structure are adapted from
Lectures/modeling/course_data.py (Lecture 8). Keep this module importable when
loading a saved pipeline, including inside Docker.
"""

import numpy as np
from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted, validate_data

FEATURES = [
    "return_5",
    "momentum_21",
    "momentum_252_21",
    "volatility_63",
    "drawdown_252",
    "relative_momentum_63",
    "beta_252",
    "range_21",
    "volume_ratio_21_252",
    "log_dollar_volume_21",
    "operating_roa",
    "gross_margin",
    "revenue_yoy",
    "asset_growth_yoy",
    "debt_to_assets",
    "cash_to_assets",
    "fcf_to_assets",
    "accruals_to_assets",
    "current_ratio",
    "filing_age_days",
]


class QuantileClipper(TransformerMixin, OneToOneFeatureMixin, BaseEstimator):
    """Learn per-column tail cutoffs on training rows; preserve missing values."""

    def __init__(self, lower=0.01, upper=0.99):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        if not 0 <= self.lower < self.upper <= 1:
            raise ValueError("Require 0 <= lower < upper <= 1")
        a = validate_data(self, X, reset=True, dtype=float, ensure_all_finite="allow-nan")
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


def make_model(alpha=1000):
    """Build the HW4 numeric pipeline; fitting occurs only in the export script."""
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
    return Pipeline(
        [
            (
                "preprocess",
                ColumnTransformer([("numeric", numeric, FEATURES)], remainder="drop"),
            ),
            ("model", Ridge(alpha=alpha)),
        ]
    )
