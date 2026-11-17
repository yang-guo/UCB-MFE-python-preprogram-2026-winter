"""Custom preprocessing required to unpickle the HW4 pipeline.

The fitted artifact stores ``course_data.QuantileClipper``. This is the same
transformer HW4 used from ``Lectures/modeling/course_data.py``. The class is
copied here so the service and the container can import it without the lecture
tree or the HW4 database.
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
        # All-missing columns are left unbounded and handled downstream.
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
