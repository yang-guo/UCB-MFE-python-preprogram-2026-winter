"""The HW4 model's feature contract and its fixed (non-learned) preprocessing.

HW4 (``Homeworks/HW4/Dhruv_Chopra/model.py``, class ``RidgeOnRanks``) fits a Ridge
regression on a design matrix built from cross-sectional percentile ranks:

* each rank r in (0, 1] is centred to r - 0.5 (a missing rank becomes 0, the
  cross-sectional median), and
* one 0/1 missing-value indicator per feature is appended.

Nothing in that step is learned from data, so the serialized artifact is an
sklearn ``Pipeline`` of ``FunctionTransformer(design)`` followed by the fitted
``Ridge``. The pickle stores ``design`` by reference, which is why this module
must ship with the service and the Docker image.
"""

import numpy as np
import pandas as pd

# HW4 v1 feature order (features.FEATURES). The Ridge coefficients are tied to it.
FEATURES = [
    "mom_5",  # 1-week return
    "mom_21",  # 1-month return
    "mom_63",  # 3-month return
    "mom_126",  # 6-month return
    "mom_252_21",  # 12-1 month momentum
    "vol_63",  # 3-month annualised volatility
    "max_ret_21",  # largest daily return in the last 21 sessions
    "drawdown_252",  # distance from the 52-week high
    "log_dollar_vol_63",  # log of 63-day average dollar volume
    "gross_profitability",  # gross profit / assets
    "operating_roa",  # operating income / assets
    "roe",  # net income / equity
    "net_margin",  # net income / revenue
    "debt_to_assets",
    "cash_to_assets",
    "fcf_to_assets",
    "accruals_to_assets",  # (net income - operating cash flow) / assets
    "revenue_growth_yoy",
    "asset_growth_yoy",
    "filing_age_days",  # days since the latest quarterly filing became available
]


def design(X) -> np.ndarray:
    """HW4 ``RidgeOnRanks._design``: centred ranks plus missing indicators.

    Accepts a DataFrame (columns must be exactly ``FEATURES`` in order) or an
    array of shape (n, len(FEATURES)). Returns shape (n, 2 * len(FEATURES)).
    """
    if isinstance(X, pd.DataFrame):
        if list(X.columns) != FEATURES:
            raise ValueError("Input columns must match the HW4 feature order")
        X = X.to_numpy(dtype=float)
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[1] != len(FEATURES):
        raise ValueError(f"Expected shape (n, {len(FEATURES)}), got {X.shape}")
    missing = np.isnan(X)
    return np.hstack([np.where(missing, 0.0, X - 0.5), missing.astype(float)])


def build_pipeline(fitted_ridge):
    """Wrap an already fitted HW4 Ridge with the fixed design step (no refit)."""
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import FunctionTransformer

    return Pipeline(
        [
            ("rank_design", FunctionTransformer(design, validate=False)),
            ("ridge", fitted_ridge),
        ]
    )
