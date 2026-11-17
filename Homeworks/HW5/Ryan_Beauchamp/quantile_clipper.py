"""QuantileClipper: the custom transformer used inside the HW4 model pipeline.

Copied (same logic) from Lectures/modeling/course_data.py, class QuantileClipper.
It lives in its own module because pickle/joblib store a class by its module path:
the exported model refers to ``quantile_clipper.QuantileClipper``, so the service
(and the Docker image) only needs this file, not the Lectures/ folder.
"""

import numpy as np
from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin
from sklearn.utils.validation import check_is_fitted, validate_data


class QuantileClipper(TransformerMixin, OneToOneFeatureMixin, BaseEstimator):
    """Learn per-column tail cutoffs on training rows; preserve missing values."""

    def __init__(self, lower=0.01, upper=0.99):
        # sklearn convention: __init__ only stores the hyperparameters.
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        # The quantile levels must be ordered and inside [0, 1].
        if not 0 <= self.lower < self.upper <= 1:
            raise ValueError("Require 0 <= lower < upper <= 1")
        # Convert to a float array; NaN is allowed (the imputer handles it), inf is not.
        # reset=True records n_features_in_ and feature_names_in_ for later checks.
        a = validate_data(self, X, reset=True, dtype=float, ensure_all_finite="allow-nan")
        # All-missing columns are left unbounded and handled by the imputer.
        self.lower_bounds_ = np.full(a.shape[1], -np.inf)
        self.upper_bounds_ = np.full(a.shape[1], np.inf)
        observed = np.isfinite(a).any(axis=0)
        # Per-column 1st / 99th percentiles of the observed (non-NaN) training values.
        self.lower_bounds_[observed] = np.nanquantile(a[:, observed], self.lower, axis=0)
        self.upper_bounds_[observed] = np.nanquantile(a[:, observed], self.upper, axis=0)
        return self

    def transform(self, X):
        # Refuse to transform before fit, and check the columns match those seen in fit.
        check_is_fitted(self, "lower_bounds_")
        a = validate_data(self, X, reset=False, dtype=float, ensure_all_finite="allow-nan")
        # Clip each column to its learned bounds; np.clip leaves NaN as NaN.
        return np.clip(a, self.lower_bounds_, self.upper_bounds_)
