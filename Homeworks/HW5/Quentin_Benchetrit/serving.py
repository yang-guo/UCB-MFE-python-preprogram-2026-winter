"""Load the saved HW4 model and turn one request into a forecast.

Nothing here knows about HTTP, so the tests can call the model the same way.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ARTIFACT = Path(__file__).resolve().parent / "artifacts" / "model.joblib"


def load_bundle(path=ARTIFACT):
    """Load the artifact written by export_model.py: model, version and feature order.

    joblib runs pickle, so only load a file you created yourself.
    """
    return joblib.load(path)


def to_frame(values, features):
    """One row in the HW4 representation: training column order, float32, NaN when missing."""
    row = [np.nan if values[name] is None else values[name] for name in features]
    return pd.DataFrame([row], columns=features, dtype="float32")


def predict_one(bundle, values):
    """Forecast for one stock; `values` maps every feature name to a rank or None."""
    return float(bundle["model"].predict(to_frame(values, bundle["features"]))[0])
