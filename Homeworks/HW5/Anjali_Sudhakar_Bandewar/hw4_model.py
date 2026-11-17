"""Reusable model logic for the HW4 stock-return model (no web code here).

The fitted artifact is a scikit-learn Pipeline([QuantileClipper, HistGradientBoostingRegressor])
whose pickle references ``hw4_model.QuantileClipper``, so this module must be importable
wherever the artifact is loaded (the service and the Docker image both include it).
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin
from sklearn.utils.validation import check_is_fitted, validate_data

# Artifact paths are resolved relative to this file, not the caller's working directory.
MODEL_DIR = Path(__file__).resolve().parent / "model"
ARTIFACT_PATH = MODEL_DIR / "hw4_hgb_fold2015.joblib"
REFERENCE_PATH = MODEL_DIR / "reference_prediction.json"

MODEL_VERSION = "hw4-hgb-fold2015-lr0.05-leaves15-iter100"

# Exact training feature names and column order (HW4 src/model_dataset.py FEATURES).
FEATURES = [
    "momentum_1m",
    "momentum_3m",
    "momentum_6m",
    "momentum_12m",
    "universe_rank",
    "profit_margin",
    "ebitda_margin",
    "debt_to_equity",
    "asset_turnover",
    "fcf_margin",
    "fundamental_age_days",
]


class QuantileClipper(TransformerMixin, OneToOneFeatureMixin, BaseEstimator):
    """Clip each column to tail quantiles learned on training rows only.

    Copied verbatim from HW4 src/train_baselines.py (itself adapted from QuantileClipper in
    Lectures/modeling/course_data.py). It lives here so the artifact can be unpickled without
    the HW4 training script and its duckdb dependency.
    Missing values are preserved for the imputer or the tree model.
    """

    def __init__(self, lower=0.01, upper=0.99):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        a = validate_data(self, X, reset=True, dtype=float, ensure_all_finite="allow-nan")
        self.lower_bounds_ = np.nanquantile(a, self.lower, axis=0)
        self.upper_bounds_ = np.nanquantile(a, self.upper, axis=0)
        return self

    def transform(self, X):
        check_is_fitted(self, "lower_bounds_")
        a = validate_data(self, X, reset=False, dtype=float, ensure_all_finite="allow-nan")
        return np.clip(a, self.lower_bounds_, self.upper_bounds_)


def load_pipeline(path: Path = ARTIFACT_PATH):
    """Load the trusted fitted HW4 pipeline and check it expects the training features."""
    pipeline = joblib.load(path)
    if list(pipeline.feature_names_in_) != FEATURES:
        raise ValueError(f"Artifact features {list(pipeline.feature_names_in_)} != {FEATURES}")
    return pipeline


def to_frame(features: dict) -> pd.DataFrame:
    """One-row DataFrame in training column order; None becomes NaN (missing, as in HW4)."""
    row = {name: [np.nan if features[name] is None else features[name]] for name in FEATURES}
    frame = pd.DataFrame(row, columns=FEATURES)
    # Training dtypes: universe_rank was int64, every other feature float64.
    return frame.astype({name: "float64" for name in FEATURES if name != "universe_rank"})


def predict_one(pipeline, features: dict) -> float:
    """Predict the 90-day forward-return percentile-rank score for one stock-date."""
    prediction = float(pipeline.predict(to_frame(features))[0])
    if not np.isfinite(prediction):
        raise ValueError(f"Model returned a non-finite prediction: {prediction}")
    return prediction
