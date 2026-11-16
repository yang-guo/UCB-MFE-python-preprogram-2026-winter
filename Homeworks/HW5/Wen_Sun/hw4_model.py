"""Serialization-safe HW4 model components."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted, validate_data

MODEL_VERSION = "hw4-lasso-alpha-0.01-selected-55-asof-2026-08-31"
MODEL_ARTIFACT_FILENAME = "model.joblib"

# features
BASE_NUMERIC_FEATURES = [
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

RAW_EXTRA_PRICE_FEATURES = [
    "momentum_10",
    "momentum_5_10_5",
    "momentum_5_21_5",
    "momentum_5_63_5",
    "momentum_5_252_5",
    "volatility_10",
    "volatility_21",
    "relative_volume_10",
    "relative_volume_21",
    "relative_volume_63",
    "amihud_illiquidity_10",
    "amihud_illiquidity_21",
    "amihud_illiquidity_63",
    "return_volume_corr_10",
    "return_volume_corr_21",
    "return_volume_corr_63",
    "turnover_10",
    "turnover_21",
    "turnover_63",
]

SECTOR_NEUTRAL_BASE_FEATURES = BASE_NUMERIC_FEATURES + RAW_EXTRA_PRICE_FEATURES
SECTOR_RELATIVE_FEATURES = [
    f"sector_relative_{feature}" for feature in SECTOR_NEUTRAL_BASE_FEATURES
]
EXTRA_PRICE_FEATURES = (
    RAW_EXTRA_PRICE_FEATURES + SECTOR_RELATIVE_FEATURES + ["sector_profitability_rank"]
)

# screening kept factors with absolute mean daily IC >= 0.01
SELECTED_NUMERIC_FEATURES = [
    "operating_roa",
    "sector_relative_operating_roa",
    "sector_profitability_rank",
    "sector_relative_fcf_to_assets",
    "fcf_to_assets",
    "volatility_63",
    "sector_relative_volatility_63",
    "sector_relative_net_margin",
    "cash_to_assets",
    "debt_to_assets",
    "volatility_21",
    "amihud_illiquidity_63",
    "amihud_illiquidity_21",
    "net_margin",
    "sector_relative_cash_to_assets",
    "sector_relative_volatility_21",
    "amihud_illiquidity_10",
    "sector_relative_amihud_illiquidity_63",
    "volatility_10",
    "sector_relative_amihud_illiquidity_21",
    "drawdown_252",
    "momentum_252_21",
    "sector_relative_volatility_10",
    "sector_relative_amihud_illiquidity_10",
    "momentum_5_252_5",
    "sector_relative_drawdown_252",
    "sector_relative_momentum_252_21",
    "sector_relative_debt_to_assets",
    "sector_relative_momentum_5_252_5",
    "momentum_10",
    "turnover_21",
    "turnover_63",
    "momentum_5_10_5",
    "sector_relative_momentum_10",
    "turnover_10",
    "sector_relative_momentum_5_10_5",
    "relative_volume_63",
    "sector_relative_relative_volume_63",
    "return_volume_corr_63",
    "relative_volume_21",
    "relative_volume_10",
    "filing_age_days",
    "sector_relative_relative_volume_21",
    "sector_relative_return_volume_corr_63",
    "sector_relative_relative_volume_10",
    "sector_relative_filing_age_days",
    "sector_relative_turnover_10",
    "momentum_5_63_5",
    "sector_relative_turnover_21",
    "sector_relative_turnover_63",
    "momentum_63",
    "sector_relative_momentum_5_63_5",
    "sector_relative_return_volume_corr_10",
    "return_volume_corr_10",
    "sector_relative_momentum_63",
]

NUMERIC_FEATURES = SELECTED_NUMERIC_FEATURES
FEATURES = NUMERIC_FEATURES + ["sector"]


class QuantileClipper(TransformerMixin, OneToOneFeatureMixin, BaseEstimator):
    """Learn per-column tail cutoffs on training rows; preserve missing values."""

    def __init__(self, lower: float = 0.01, upper: float = 0.99):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        if not 0 <= self.lower < self.upper <= 1:
            raise ValueError("Require 0 <= lower < upper <= 1")
        a = validate_data(
            self,
            X,
            reset=True,
            dtype=float,
            ensure_all_finite="allow-nan",
        )
        self.lower_bounds_ = np.full(a.shape[1], -np.inf)
        self.upper_bounds_ = np.full(a.shape[1], np.inf)
        observed = np.isfinite(a).any(axis=0)
        self.lower_bounds_[observed] = np.nanquantile(a[:, observed], self.lower, axis=0)
        self.upper_bounds_[observed] = np.nanquantile(a[:, observed], self.upper, axis=0)
        return self

    def transform(self, X):
        check_is_fitted(self, "lower_bounds_")
        a = validate_data(
            self,
            X,
            reset=False,
            dtype=float,
            ensure_all_finite="allow-nan",
        )
        return np.clip(a, self.lower_bounds_, self.upper_bounds_)


def build_pipeline(
    model=None,
    numeric_features: list[str] | None = None,
    include_sector: bool = True,
) -> Pipeline:
    """Return the same preprocessing/model pipeline used by HW4."""
    numeric_features = NUMERIC_FEATURES if numeric_features is None else numeric_features
    numeric = Pipeline(
        [
            ("clip", QuantileClipper()),
            (
                "impute",
                SimpleImputer(
                    strategy="median",
                    add_indicator=True,
                    keep_empty_features=True,
                ),
            ),
            ("scale", StandardScaler()),
        ]
    )
    transforms = [("numeric", numeric, numeric_features)]
    if include_sector:
        transforms.append(
            (
                "sector",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ["sector"],
            )
        )
    return Pipeline(
        [
            (
                "preprocess",
                ColumnTransformer(transforms, remainder="drop"),
            ),
            ("model", Lasso(alpha=0.01, max_iter=10000) if model is None else model),
        ]
    )


def build_features_frame(numeric_values: list[float], sector: str) -> pd.DataFrame:
    """Build a one-row DataFrame with the exact training column names/order."""
    if len(numeric_values) != len(NUMERIC_FEATURES):
        raise ValueError(
            f"Expected {len(NUMERIC_FEATURES)} numeric features, got {len(numeric_values)}"
        )
    return pd.DataFrame([[*numeric_values, sector]], columns=FEATURES)
