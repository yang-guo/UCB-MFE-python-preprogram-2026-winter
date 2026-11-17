"""Model registry.  Every model is an sklearn estimator over ranked features.

Inputs are cross-sectional percentile ranks (no fitted parameters), so the
only learned preprocessing is the imputer/scaler inside each pipeline, which
is refitted on each fold's training rows only (Lecture 7 pattern).
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class FixedComposite(RegressorMixin, BaseEstimator):
    """Average of ranked features with pre-specified signs; learns nothing.

    A transparent factor-composite baseline: if a fitted model cannot beat a
    fixed, theory-signed composite, its extra flexibility is not paying off.
    Missing inputs contribute 0 (the cross-sectional median rank).
    """

    def __init__(self, signs=None):
        self.signs = signs

    def fit(self, X, y=None):
        X = np.asarray(X, float)
        self.n_features_in_ = X.shape[1]
        s = np.ones(X.shape[1]) if self.signs is None else np.asarray(self.signs, float)
        if len(s) != X.shape[1]:
            raise ValueError("One sign per feature")
        self.coef_ = s / np.abs(s).sum()
        return self

    def predict(self, X):
        return np.nan_to_num(np.asarray(X, float), nan=0.0) @ self.coef_


def _ridge(alpha: float) -> Pipeline:
    return Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
                     ("scale", StandardScaler()),
                     ("model", Ridge(alpha=alpha))])


class TimeTunedRidge(RegressorMixin, BaseEstimator):
    """Ridge whose penalty is chosen on the most recent slice of its *own training rows*.

    Rows must arrive in time order (the panel is sorted by date, and both the
    validation folds and the walk-forward scorer slice it without reordering).
    Each candidate penalty is fitted on the first ``fit_frac`` of rows and
    checked on the last ``check_frac`` (correlation of prediction with the
    target); the rows in between are skipped so the inner fit's 90-day labels
    barely overlap the inner check.  The chosen penalty is then refitted on all
    rows.  Only training rows are ever seen, so the outer validation years and
    the final test play no part in the choice.
    """

    def __init__(self, alphas=(1.0, 10.0, 100.0, 1000.0, 10000.0), fit_frac=0.60, check_frac=0.25):
        self.alphas = alphas
        self.fit_frac = fit_frac
        self.check_frac = check_frac

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y, float)
        a, b = int(len(y) * self.fit_frac), len(y) - int(len(y) * self.check_frac)
        self.inner_scores_ = {}
        for alpha in self.alphas:
            pred = _ridge(alpha).fit(X[:a], y[:a]).predict(X[b:])
            self.inner_scores_[float(alpha)] = float(np.corrcoef(pred, y[b:])[0, 1])
        self.alpha_ = max(self.inner_scores_, key=lambda k: (self.inner_scores_[k], k))  # ties: more shrinkage
        self.model_ = _ridge(self.alpha_).fit(X, y)
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        return self.model_.predict(np.asarray(X, float))


def build_model(kind: str, params: dict | None = None):
    params = dict(params or {})
    if kind == "ridge_tuned":
        return TimeTunedRidge(**params)
    if kind == "mean":
        return DummyRegressor(strategy="mean")
    if kind == "ridge":
        return _ridge(params.get("alpha", 10.0))
    if kind == "hgb":
        defaults = dict(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=500,
                        l2_regularization=1.0, early_stopping=False, random_state=42)
        defaults.update(params)
        return HistGradientBoostingRegressor(**defaults)  # handles NaN natively
    if kind == "composite":
        return FixedComposite(signs=params.get("signs"))
    raise KeyError(f"Unknown model kind {kind!r}")
