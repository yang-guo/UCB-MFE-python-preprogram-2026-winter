"""QuantileClipper from Lectures/modeling/course_data.py at HW4 commit fca22b5.

The frozen artifact refers to course_data.QuantileClipper by this module name.
Only this transformer and its imports are needed for inference.
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
        values = validate_data(self, X, reset=True, dtype=float, ensure_all_finite="allow-nan")
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
