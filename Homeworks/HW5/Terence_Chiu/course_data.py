"""Custom transformer needed to load the HW4 pipeline.

Copied unchanged from Lectures/modeling/course_data.py (QuantileClipper only). The
module keeps the name `course_data` because the pickled pipeline refers to
`course_data.QuantileClipper`; the course file's data-loading code (and its duckdb
import) is not needed for serving.
"""

import numpy as np
from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin
from sklearn.utils.validation import check_is_fitted, validate_data


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
