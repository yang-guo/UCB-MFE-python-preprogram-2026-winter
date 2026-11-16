import hashlib
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import course_data  # noqa: F401  (needed to unpickle QuantileClipper)
from course_data import FEATURES

MODEL_PATH = Path(__file__).parent / "model" / "model.joblib"
MODEL_NAME = "hw4-boosting-rank90d-refit-2026-01-30"


def load_model(path=MODEL_PATH):
    model = joblib.load(path)
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]
    return model, f"{MODEL_NAME}-{digest}"


def to_frame(features):
    # one row with the same column names and order used in HW4; None -> NaN (imputed)
    row = {name: features[name] for name in FEATURES}
    frame = pd.DataFrame([row], columns=FEATURES)
    numeric = frame.columns.drop("sector")
    frame[numeric] = frame[numeric].astype(float)
    return frame


def predict_one(model, features):
    prediction = float(model.predict(to_frame(features))[0])
    if not np.isfinite(prediction):
        raise ValueError("Model returned a non-finite prediction")
    return prediction
