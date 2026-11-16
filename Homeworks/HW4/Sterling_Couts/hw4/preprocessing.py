"""Serializable preprocessing; adapted from Lecture 7/course_data.QuantileClipper.

Only fit learns clipping bounds. Prediction never learns from the request batch.
Keeping this class in an importable module also permits HW5 artifact loading.
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.linear_model import Ridge
from sklearn.utils.validation import check_is_fitted


class TrainingClipper(TransformerMixin, BaseEstimator):
    """Clip numeric features to training quantiles, preserving missing values."""

    def __init__(self, lower=0.01, upper=0.99):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        if not 0 <= self.lower < self.upper <= 1:
            raise ValueError("Invalid quantile bounds")
        values = np.asarray(X, dtype=float)
        if values.ndim != 2 or np.isinf(values).any():
            raise ValueError("Training features must be a finite-or-missing matrix")
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = values.shape[1]
        self.bounds_ = np.array(
            [[-np.inf] * values.shape[1], [np.inf] * values.shape[1]]
        )
        present = np.isfinite(values).any(axis=0)
        self.bounds_[:, present] = np.nanquantile(
            values[:, present], [self.lower, self.upper], axis=0
        )
        return self

    def transform(self, X):
        check_is_fitted(self, "bounds_")
        if hasattr(self, "feature_names_in_") and hasattr(X, "columns"):
            if list(X.columns) != list(self.feature_names_in_):
                raise ValueError("Feature names and order must match training")
        values = np.asarray(X, dtype=float)
        if values.ndim != 2 or values.shape[1] != self.n_features_in_:
            raise ValueError("Feature count does not match training")
        if np.isinf(values).any():
            raise ValueError("Infinite feature input")
        return np.clip(values, *self.bounds_)


class TargetClippedRegressor(RegressorMixin, BaseEstimator):
    """Regressor fitted to winsorized returns; prices/PnL are never clipped.

    This deliberately estimates a robust, clipped-return response rather than
    the unconstrained conditional mean. Quantiles are learned anew at each fit.
    It limits sensitivity to extreme labels but does NOT repair vendor prices.
    """

    def __init__(self, estimator=None, quantiles=(0.01, 0.99)):
        self.estimator = estimator
        self.quantiles = quantiles

    def fit(self, X, y, sample_weight=None):
        target = np.asarray(y, dtype=float)
        if target.ndim != 1 or not np.isfinite(target).all():
            raise ValueError("Training returns must be finite")
        lo, hi = self.quantiles
        if not 0 <= lo < hi <= 1:
            raise ValueError("Invalid target quantiles")
        self.target_bounds_ = np.quantile(target, [lo, hi])
        clipped = np.clip(target, *self.target_bounds_)
        self.clipped_labels_ = int(np.count_nonzero(clipped != target))
        self.estimator_ = (
            clone(self.estimator) if self.estimator is not None else Ridge(alpha=100.0)
        )
        self.estimator_.fit(X, clipped, sample_weight=sample_weight)
        self.n_features_in_ = self.estimator_.n_features_in_
        return self

    def predict(self, X):
        check_is_fitted(self, "estimator_")
        return self.estimator_.predict(X)
