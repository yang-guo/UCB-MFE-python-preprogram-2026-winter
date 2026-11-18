"""The HW4 feature list and preprocessing pipeline, importable without the course database.

The saved joblib artifact contains a ``QuantileClipper`` object. Unpickling it needs the
class importable under the same module path it was saved with, so the class lives here
(and is copied into the Docker image) instead of in ``Lectures/modeling/course_data.py``,
which also imports duckdb and data-loading code the service does not need.

Adapted from ``Lectures/modeling/course_data.py`` (``QuantileClipper``, ``make_pipeline``,
and the feature lists). The logic is unchanged, so a pipeline built here is the same
pipeline HW4 fitted; ``train_model.py`` checks that its predictions match the scores the
HW4 notebook saved.
"""

import numpy as np
from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted, validate_data

# Feature order used in HW4 training. The pipeline selects columns by name, so requests
# are turned into a one-row DataFrame with exactly these columns in this order.
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

# The 11 sector labels present in the HW4 training data.
SECTORS = (
    "Basic Materials",
    "Communication Services",
    "Consumer Cyclical",
    "Consumer Defensive",
    "Energy",
    "Financial Services",
    "Healthcare",
    "Industrials",
    "Real Estate",
    "Technology",
    "Utilities",
)


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


def make_pipeline(alpha: float) -> Pipeline:
    """Clip -> median-impute (+ missing indicators) -> scale, one-hot sector, then Ridge."""
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
    preprocess = ColumnTransformer(
        [
            ("numeric", numeric, NUMERIC_FEATURES),
            ("sector", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["sector"]),
        ],
        remainder="drop",
    )
    return Pipeline([("preprocess", preprocess), ("model", Ridge(alpha=alpha))])
