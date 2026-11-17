"""Model loading and inference, kept separate from HTTP request handling.

The served model is ``final_relative_model`` from HW4 (cell 55): a scikit-learn
Pipeline of ColumnTransformer[SimpleImputer(median, add_indicator) ->
StandardScaler] -> Ridge(alpha=10000), fitted on 2010-01 to 2015-12 signals.
It predicts a stock's 90-calendar-day forward return minus the cross-sectional
median return of the universe on the same signal date (decimal, 0.01 = 1%).
"""

from __future__ import annotations

import hashlib
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

MODEL_NAME = "hw4-final-relative-ridge-a10000"
DEFAULT_MODEL_PATH = Path(__file__).parent / "artifacts" / "ridge_model.joblib"

# Order matters: this is HW4's `original_features`, the column order the
# ColumnTransformer was fitted with.
FEATURE_NAMES: tuple[str, ...] = (
    "book_to_market",
    "earnings_yield",
    "sales_to_market",
    "cash_flow_to_market",
    "momentum_1m",
    "momentum_3m",
    "momentum_6m",
    "momentum_12m",
    "momentum_12m_1m",
    "roe",
    "roa",
    "gross_profitability",
    "operating_cash_flow_to_assets",
    "net_margin",
    "debt_to_assets",
    "volatility_21d",
    "volatility_63d",
    "volatility_126d",
)


@dataclass(frozen=True)
class LoadedModel:
    pipeline: Any
    version: str


def model_path() -> Path:
    return Path(os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH))


def load_model(path: Path | None = None) -> LoadedModel:
    """Load the trusted, self-created HW4 artifact. Never refits."""
    path = path or model_path()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
    pipeline = joblib.load(path)

    fitted_columns = tuple(getattr(pipeline, "feature_names_in_", ()))
    if fitted_columns != FEATURE_NAMES:
        raise RuntimeError(
            f"Artifact feature order {fitted_columns} does not match {FEATURE_NAMES}"
        )
    return LoadedModel(pipeline=pipeline, version=f"{MODEL_NAME}+sha256:{digest}")


def to_frame(features: dict[str, float | None]) -> pd.DataFrame:
    """One-row DataFrame with the training column names and order.

    ``None`` becomes NaN, which the fitted SimpleImputer handles exactly as it
    did for missing values during HW4 training.
    """
    row = {name: features[name] for name in FEATURE_NAMES}
    return pd.DataFrame([row], columns=list(FEATURE_NAMES), dtype="float64")


def predict_one(pipeline: Any, features: dict[str, float | None]) -> float:
    prediction = float(pipeline.predict(to_frame(features))[0])
    if not math.isfinite(prediction):
        raise ValueError("Model returned a non-finite prediction")
    return prediction
