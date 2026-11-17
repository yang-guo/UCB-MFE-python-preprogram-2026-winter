"""Load the trusted HW4 artifact and score one row of ranks in training column order."""

from pathlib import Path

import joblib
import pandas as pd

ARTIFACT = Path(__file__).resolve().parent / "artifacts" / "model.joblib"
REQUIRED_KEYS = {"model", "model_version", "features"}


def load_bundle(path=ARTIFACT):
    """Load the bundle written by export_model.py. Only this course artifact is ever loaded."""
    bundle = joblib.load(path)
    missing = REQUIRED_KEYS - set(bundle)
    if missing:
        raise ValueError(f"{path.name} lacks {sorted(missing)}")
    return bundle


def predict_one(bundle, features):
    """Score one stock from its four ranks, keeping the column names and order used in training."""
    frame = pd.DataFrame([features])[bundle["features"]]
    return float(bundle["model"].predict(frame)[0])
