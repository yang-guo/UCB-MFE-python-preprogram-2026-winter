"""Custom sklearn pieces for the HW5 service (adapted from the HW4 Fama-MacBeth alpha model).

HW4 z-scored each feature within sector and date. That is a cross-sectional step: it needs the
whole week's stock universe, so it cannot run on a single API request. For serving, the preprocessing
is made per-row instead, with statistics learned once on the training set:
clip at training quantiles -> standardize with training mean/std -> clip at +/-3.
The Fama-MacBeth premia are re-estimated on these exact features, so training and serving agree.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin


class QuantileClipper(BaseEstimator, TransformerMixin):
    """Clip each feature to its [lower, upper] training quantiles (winsorize)."""

    def __init__(self, lower=0.01, upper=0.99):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.low_ = X.quantile(self.lower).to_numpy()
        self.high_ = X.quantile(self.upper).to_numpy()
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.low_, self.high_)


class ZClip(BaseEstimator, TransformerMixin):
    """Clip standardized features to [-limit, limit], as HW4 did after z-scoring."""

    def __init__(self, limit=3.0):
        self.limit = limit

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), -self.limit, self.limit)


class FamaMacBethRegressor(RegressorMixin, BaseEstimator):
    """Linear score = X @ premia, where premia are shrunken Fama-MacBeth averages.

    fit() runs one cross-sectional OLS per date, averages the slopes, computes Newey-West
    t-statistics (labels overlap across ~13 weekly dates) and shrinks each premium by
    t^2 / (t^2 + 4), exactly as in HW4's benchrel.py.
    """

    def __init__(self, lag=13, min_names=60):
        self.lag = lag
        self.min_names = min_names

    def fit(self, X, y, dates):
        X, y, dates = np.asarray(X, float), np.asarray(y, float), np.asarray(dates)
        slopes = []
        for d in np.unique(dates):
            ix = np.flatnonzero(dates == d)
            if len(ix) <= self.min_names:
                continue
            A = np.c_[np.ones(len(ix)), X[ix]]
            slopes.append(np.linalg.lstsq(A, y[ix], rcond=1e-8)[0][1:])
        G = np.stack(slopes)
        mean, T = G.mean(0), len(G)
        e = G - mean
        var = (e**2).sum(0) / T
        for k in range(1, self.lag + 1):
            var = var + 2 * (1 - k / (self.lag + 1)) * (e[k:] * e[:-k]).sum(0) / T
        t = mean / np.sqrt(np.maximum(var, 1e-12) / T)
        self.mean_slopes_, self.t_stats_ = mean, t
        self.coef_ = mean * t**2 / (t**2 + 4.0)
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        return np.asarray(X, dtype=float) @ self.coef_
