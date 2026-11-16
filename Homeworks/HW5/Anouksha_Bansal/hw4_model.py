"""Reusable model logic for the HW4 `ridge_core_ret` model, kept apart from the API.

`app.py` handles HTTP and validation; this module knows the training contract:
feature names and order, the float32 input representation, and how to load the
saved artifact.
"""

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

# Version of the HW4 model being served: config name + walk-forward fit date.
MODEL_VERSION = "hw4-ridge_core_ret-fit-2026-01-02"

# Exact training column names and order (HW4 `CORE`). Each value is the stock's
# same-date percentile rank among the screened universe, minus 0.5.
FEATURES = [
    "r_gross_profitability",
    "r_operating_roa",
    "r_fcf_to_assets",
    "r_earnings_yield",
    "r_sales_to_price",
    "r_sales_growth_yoy",
    "r_mom_252_21",
]

ARTIFACT_PATH = Path(__file__).resolve().parent / "artifacts" / "model.joblib"


def load_bundle(path: Path = ARTIFACT_PATH) -> dict[str, Any]:
    """Load the trusted, self-created artifact and check it matches this contract.

    joblib/pickle can run arbitrary code while loading, so only load files
    created by `export_model.py`.
    """
    bundle = joblib.load(path)
    if list(bundle["pipeline"].feature_names_in_) != FEATURES:
        raise RuntimeError("Artifact feature order does not match FEATURES")
    if bundle["model_version"] != MODEL_VERSION:
        raise RuntimeError(f"Unexpected artifact version {bundle['model_version']!r}")
    return bundle


def to_frame(features: dict[str, float | None]) -> pd.DataFrame:
    """One-row DataFrame in the training layout: named columns, fixed order, float32.

    `None` (feature unavailable) becomes NaN; the pipeline's imputer then fills it
    with 0, the median rank, as HW4's `fillna(0.0)` did.
    """
    row = [np.nan if features[name] is None else features[name] for name in FEATURES]
    return pd.DataFrame([row], columns=FEATURES, dtype="float32")


def predict_one(pipeline: Any, features: dict[str, float | None]) -> float:
    """Predicted 90-day excess return (decimal) for one stock-date."""
    prediction = float(pipeline.predict(to_frame(features))[0])
    if not np.isfinite(prediction):
        raise ValueError("Model produced a non-finite prediction")
    return prediction
