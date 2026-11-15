"""Model logic for the HW5 service, kept separate from request handling.

The artifact is the fitted HW4 scikit-learn ``Pipeline`` (imputer + one-hot
encoder + Ridge) wrapped in a small dict with its version and feature contract.
It contains only scikit-learn objects, so no custom transformer modules are
needed to load it.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ARTIFACT_PATH = Path(__file__).resolve().parent / "artifacts" / "model.joblib"

# Training column order of the HW4 Ridge model (HW4 features.py: PRICE_FEATURES +
# FUND_FEATURES). Every value is the stock's percentile rank among that signal
# date's eligible stocks, as produced by HW4 ``rank_features``.
PRICE_FEATURES = (
    "mom_1m",
    "mom_3m",
    "mom_6m_1m",
    "mom_12_1",
    "vol_63",
    "vol_252",
    "max_ret_21",
    "dist_high_252",
    "beta_252",
    "idio_vol_252",
    "log_dollar_vol",
    "log_amihud",
)
FUND_FEATURES = (
    "log_mcap",
    "earnings_yield",
    "book_to_market",
    "sales_to_price",
    "cashflow_to_price",
    "gross_profit_to_assets",
    "roe",
    "asset_growth",
    "sales_growth",
    "earnings_change_to_price",
    "accruals_to_assets",
    "debt_to_assets",
    "share_issuance",
    "filing_age_days",
)
FEATURES = PRICE_FEATURES + FUND_FEATURES
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
    "Unknown",
)


class ModelService:
    """A loaded model bundle and the one way to call it."""

    def __init__(self, bundle: dict):
        if tuple(bundle["features"]) != FEATURES:
            raise ValueError("Artifact feature order does not match this service")
        self.pipeline = bundle["pipeline"]
        self.model_version: str = bundle["model_version"]
        self.nullable_features: tuple[str, ...] = tuple(bundle["nullable_features"])

    @classmethod
    def load(cls, path: Path = ARTIFACT_PATH) -> "ModelService":
        # joblib/pickle can execute code on load: only load artifacts we created.
        return cls(joblib.load(path))

    def to_frame(self, features: dict[str, float | None], sector: str) -> pd.DataFrame:
        """One-row frame with the column names, order and dtypes used in training."""
        row = {name: features[name] for name in FEATURES}
        frame = pd.DataFrame([row], columns=list(FEATURES)).astype("float32")  # None -> NaN
        frame["sector"] = sector
        return frame

    def predict(self, features: dict[str, float | None], sector: str) -> float:
        prediction = float(self.pipeline.predict(self.to_frame(features, sector))[0])
        if not np.isfinite(prediction):
            raise ValueError("Model returned a non-finite prediction")
        return prediction
