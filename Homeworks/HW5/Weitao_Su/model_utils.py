"""Load the saved pipeline and preserve HW4's DataFrame feature order."""

from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent
FEATURES = [
    "momentum_21",
    "momentum_63",
    "momentum_252_21",
    "volatility_63",
    "drawdown_252",
    "relative_momentum_63",
    "momentum_acceleration",
]


def load_model():
    # This is our own trusted export. Never load a user-supplied pickle/joblib file.
    bundle = joblib.load(ROOT / "artifacts" / "model.joblib")
    if bundle["feature_names"] != FEATURES:
        raise ValueError("The saved feature order does not match the API")
    if list(bundle["pipeline"].feature_names_in_) != FEATURES:
        raise ValueError("The fitted pipeline does not match the API features")
    return bundle


def predict_return(pipeline, features):
    row = pd.DataFrame([features], columns=FEATURES, dtype=float)
    return float(pipeline.predict(row)[0])
