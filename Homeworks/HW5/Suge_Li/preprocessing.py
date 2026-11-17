"""Learned preprocessing for the served HW4 Ridge pipeline.

``QuantileClipper`` is copied from ``Lectures/modeling/course_data.py`` so a
joblib artifact can be loaded in this service and in the container. The class
must stay importable as ``preprocessing.QuantileClipper``; that is the module
path stored in the artifact. Clip bounds, imputer medians, and scaler moments
are learned in ``train_artifact.py`` and are not refit on requests.
"""

import numpy as np
from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted, validate_data


class QuantileClipper(TransformerMixin, OneToOneFeatureMixin, BaseEstimator):
    """Learn per-column tail cutoffs on training rows; preserve missing values."""

    def __init__(self, lower: float = 0.01, upper: float = 0.99):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        if not 0 <= self.lower < self.upper <= 1:
            raise ValueError("Require 0 <= lower < upper <= 1")
        values = validate_data(self, X, reset=True, dtype=float, ensure_all_finite="allow-nan")
        # All-missing columns are left unbounded and handled by the imputer.
        self.lower_bounds_ = np.full(values.shape[1], -np.inf)
        self.upper_bounds_ = np.full(values.shape[1], np.inf)
        observed = np.isfinite(values).any(axis=0)
        self.lower_bounds_[observed] = np.nanquantile(values[:, observed], self.lower, axis=0)
        self.upper_bounds_[observed] = np.nanquantile(values[:, observed], self.upper, axis=0)
        return self

    def transform(self, X):
        check_is_fitted(self, "lower_bounds_")
        values = validate_data(self, X, reset=False, dtype=float, ensure_all_finite="allow-nan")
        return np.clip(values, self.lower_bounds_, self.upper_bounds_)


def make_ridge_pipeline(feature_names: list[str]) -> Pipeline:
    """Match ``course_data.make_pipeline(Ridge(alpha=100), include_sector=False)``."""
    numeric = Pipeline(
        [
            ("clip", QuantileClipper()),
            (
                "impute",
                SimpleImputer(
                    strategy="median",
                    add_indicator=True,
                    keep_empty_features=True,
                ),
            ),
            ("scale", StandardScaler()),
        ]
    )
    preprocess = ColumnTransformer(
        [("numeric", numeric, list(feature_names))],
        remainder="drop",
    )
    return Pipeline([("preprocess", preprocess), ("model", Ridge(alpha=100))])
