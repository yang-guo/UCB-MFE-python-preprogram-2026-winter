"""The code of my HW4 model.

QuantileClipper and make_pipeline come from the course file course_data.py.
Python needs QuantileClipper to open the saved model, so I copy it here.
"""

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted, validate_data

# The 13 features, with the same names and order as in HW4.
PRICE_FEATURES = [
    "momentum_21",
    "momentum_63",
    "momentum_252_21",
    "volatility_63",
    "drawdown_252",
]
FINANCIAL_FEATURES = [
    "net_margin",
    "operating_roa",
    "debt_to_assets",
    "cash_to_assets",
    "fcf_to_assets",
    "accruals_to_assets",
]
NUMERIC_FEATURES = PRICE_FEATURES + FINANCIAL_FEATURES + ["filing_age_days"]
FEATURES = NUMERIC_FEATURES + ["sector"]


class QuantileClipper(TransformerMixin, OneToOneFeatureMixin, BaseEstimator):
    """Cut the extreme values. Same code as in the course."""

    def __init__(self, lower=0.01, upper=0.99):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        if not 0 <= self.lower < self.upper <= 1:
            raise ValueError("Require 0 <= lower < upper <= 1")
        a = validate_data(self, X, reset=True, dtype=float, ensure_all_finite="allow-nan")
        # If a column has no values, do not cut it.
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


def make_pipeline(model):
    """The HW4 pipeline. Same steps as in the course."""
    # Numbers: cut extreme values, fill missing values with the median, scale.
    numeric = Pipeline(
        [
            ("clip", QuantileClipper()),
            (
                "impute",
                SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
            ),
            ("scale", StandardScaler()),
        ]
    )
    # Sector: one column for each sector.
    transforms = [
        ("numeric", numeric, NUMERIC_FEATURES),
        ("sector", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["sector"]),
    ]
    return Pipeline(
        [
            ("preprocess", ColumnTransformer(transforms, remainder="drop")),
            ("model", model),
        ]
    )


def load_bundle(path):
    """Open the saved model. Stop if something is not like in HW4."""
    bundle = joblib.load(path)  # only open files you made yourself
    if bundle["sklearn_version"] != sklearn.__version__:
        raise RuntimeError("Wrong scikit-learn version: use the HW4 version")
    if bundle["features"] != FEATURES:
        raise RuntimeError("Wrong feature list")
    return bundle


def predict_return(pipeline, features):
    """Predict the 90-day return of one stock."""
    # One row, with the HW4 columns. None (unknown) becomes NaN.
    frame = pd.DataFrame([features], columns=FEATURES)
    frame[NUMERIC_FEATURES] = frame[NUMERIC_FEATURES].astype(float)
    return float(pipeline.predict(frame)[0])
