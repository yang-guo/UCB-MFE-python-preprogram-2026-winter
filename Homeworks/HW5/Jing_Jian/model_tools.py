"""Inference contract and training transformer adapted from modeling/course_data.py."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin
from sklearn.utils.validation import check_is_fitted, validate_data

ROOT = Path(__file__).resolve().parent
FEATURES = ["momentum_21", "momentum_63", "momentum_252_21", "volatility_63", "drawdown_252"]
MODEL_VERSION = "hw4-price-top20-ridge100-20160331-v1"


class QuantileClipper(TransformerMixin, OneToOneFeatureMixin, BaseEstimator):
    """Same training-only 1%/99% clipping as the HW4 course transformer."""

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


def load_bundle():
    """Load only our locally generated, trusted artifact."""
    bundle = joblib.load(ROOT / "artifacts/model.joblib")
    if bundle["features"] != FEATURES or bundle["model_version"] != MODEL_VERSION:
        raise ValueError("Artifact does not match the service contract")
    if list(bundle["pipeline"].feature_names_in_) != FEATURES:
        raise ValueError("Artifact feature order differs from HW4")
    return bundle


def feature_frame(values):
    """Retain the named DataFrame representation consumed by ColumnTransformer."""
    return pd.DataFrame([[values[name] for name in FEATURES]], columns=FEATURES)
