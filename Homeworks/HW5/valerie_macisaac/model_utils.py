"""Importable preprocessing and ensemble model classes for HW4 stock-return prediction."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV

FEATURES_15: list[str] = [
    "ebitda_to_assets",
    "roa",
    "fcf_to_assets",
    "ocf_to_assets",
    "op_margin",
    "gross_margin",
    "ret_252d",
    "mom_12m_skip1m",
    "earnings_yield",
    "fcf_yield",
    "book_to_market",
    "neg_universe_rank",
    "vol_63d",
    "avg_hl_spread_21d",
    "mcap_to_rev_ttm",
]


class CrossSectionalRankTransformer(BaseEstimator, TransformerMixin):
    """Maps raw features to centered empirical percentile ranks in [-0.5, 0.5].

    During ``fit``, stores the sorted finite training distribution per feature so
    single-row inference in production reproduces cross-sectional ranking without
    querying the historical database. Missing values impute to ``0.0`` (median).
    """

    def __init__(self, feature_names: list[str] | None = None) -> None:
        self.feature_names = feature_names

    def fit(self, X: pd.DataFrame | np.ndarray, y: np.ndarray | None = None):
        cols = self.feature_names or FEATURES_15
        if isinstance(X, pd.DataFrame):
            df = X[cols]
        else:
            df = pd.DataFrame(X, columns=cols)

        self.feature_names_in_ = np.asarray(cols, dtype=object)
        self.sorted_refs_: dict[str, np.ndarray] = {}
        for col in cols:
            vals = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
            finite_vals = np.sort(vals[np.isfinite(vals)])
            self.sorted_refs_[col] = finite_vals
        return self

    def transform(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        cols = list(self.feature_names_in_)
        if isinstance(X, pd.DataFrame):
            if list(X.columns) != cols:
                df = X[cols]
            else:
                df = X
        else:
            arr = np.asarray(X, dtype=float)
            if arr.ndim != 2 or arr.shape[1] != len(cols):
                raise ValueError(f"Expected {len(cols)} features, got shape {arr.shape}")
            df = pd.DataFrame(arr, columns=cols)

        out = np.zeros((len(df), len(cols)), dtype=float)
        for j, col in enumerate(cols):
            vals = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
            ref = self.sorted_refs_[col]
            if ref.size == 0:
                continue
            valid_mask = np.isfinite(vals)
            if np.any(valid_mask):
                v = vals[valid_mask]
                left = np.searchsorted(ref, v, side="left")
                right = np.searchsorted(ref, v, side="right")
                avg_rank = 0.5 * (left + right + 1.0)
                pct = np.clip(avg_rank / ref.size, 0.0, 1.0)
                out[valid_mask, j] = pct - 0.5
        return out


class HW4StockReturnEnsemble(BaseEstimator, RegressorMixin):
    """60/40 RidgeCV + HistGradientBoostingRegressor ensemble for 90d return rank."""

    def __init__(
        self,
        ridge_weight: float = 0.6,
        gbdt_weight: float = 0.4,
        random_state: int = 42,
    ) -> None:
        self.ridge_weight = ridge_weight
        self.gbdt_weight = gbdt_weight
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.ridge_ = RidgeCV(alphas=np.logspace(1, 5, 20))
        self.gbdt_ = HistGradientBoostingRegressor(
            max_depth=2,
            learning_rate=0.02,
            max_iter=80,
            min_samples_leaf=200,
            l2_regularization=50.0,
            random_state=self.random_state,
        )
        self.ridge_.fit(X, y)
        self.gbdt_.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        pred_ridge = self.ridge_.predict(X)
        pred_gbdt = self.gbdt_.predict(X)
        return self.ridge_weight * pred_ridge + self.gbdt_weight * pred_gbdt
