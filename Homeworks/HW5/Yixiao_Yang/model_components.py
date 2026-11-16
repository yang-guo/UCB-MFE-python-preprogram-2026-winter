"""The HW4 feature order and the small transformer needed to load its model.

TrainingQuantileClipper is copied from my HW4 submission, without changing
its calculations. Keeping it here lets the service load the fitted pipeline
without importing HW4's database and backtesting code.
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

FEATURE_COLUMNS = [
    "momentum_21",
    "momentum_63",
    "momentum_252_21",
    "volatility_63",
    "drawdown_252",
    "net_margin",
    "operating_roa",
    "debt_to_assets",
    "cash_to_assets",
    "fcf_to_assets",
    "accruals_to_assets",
    "filing_age_days",
]
MODEL_VERSION = "hw4-yixiao-ridge-2016q2-v1"


class TrainingQuantileClipper(TransformerMixin, BaseEstimator):
    """Learn 1%/99% feature limits on each training fold only.

    Adapted from course_data.QuantileClipper. All-missing columns stay
    unbounded, then SimpleImputer handles them without dropping columns.
    """

    def __init__(self, lower: float = 0.01, upper: float = 0.99):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        values = np.asarray(X, dtype=float)
        if values.ndim != 2 or not 0 <= self.lower < self.upper <= 1:
            raise ValueError("Expected a matrix and 0 <= lower < upper <= 1")
        self.n_features_in_ = values.shape[1]
        self.lower_bounds_ = np.full(values.shape[1], -np.inf)
        self.upper_bounds_ = np.full(values.shape[1], np.inf)
        observed = np.isfinite(values).any(axis=0)
        self.lower_bounds_[observed] = np.nanquantile(values[:, observed], self.lower, axis=0)
        self.upper_bounds_[observed] = np.nanquantile(values[:, observed], self.upper, axis=0)
        return self

    def transform(self, X):
        if not hasattr(self, "lower_bounds_"):
            raise ValueError("Fit the clipper before transforming features")
        values = np.asarray(X, dtype=float)
        if values.ndim != 2 or values.shape[1] != self.n_features_in_:
            raise ValueError("Feature shape changed after fitting")
        return np.clip(values, self.lower_bounds_, self.upper_bounds_)
