"""Model logic with no web code: load the saved HW4 bundle and score one row."""

import math
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from hw4_pipeline import FEATURES, NUMERIC_FEATURES

ARTIFACT_PATH = Path(__file__).resolve().parent / "artifacts" / "model.joblib"


def load_bundle(path: Path = ARTIFACT_PATH) -> dict[str, Any]:
    """Load the trusted artifact written by train_model.py (never load untrusted pickles)."""
    bundle = joblib.load(path)
    if bundle["features"] != FEATURES:
        raise RuntimeError(f"Artifact features {bundle['features']} differ from {FEATURES}")
    return bundle


def features_frame(values: dict[str, Any]) -> pd.DataFrame:
    """One request -> the one-row DataFrame HW4 trained on (same column names and order).

    ``None`` (a feature that is missing, as in the HW4 panel) becomes NaN, which the
    pipeline's imputer fills with the training median.
    """
    frame = pd.DataFrame([values], columns=FEATURES)
    frame[NUMERIC_FEATURES] = frame[NUMERIC_FEATURES].astype("float64")
    if frame.shape != (1, len(FEATURES)):
        raise ValueError(f"Expected 1 x {len(FEATURES)} features, got {frame.shape}")
    return frame


def predict_one(pipeline: Any, values: dict[str, Any]) -> float:
    """Score one stock. Raises if the model fails or returns a non-finite number."""
    prediction = float(pipeline.predict(features_frame(values))[0])
    if not math.isfinite(prediction):
        raise ValueError(f"Model returned a non-finite prediction: {prediction}")
    return prediction
