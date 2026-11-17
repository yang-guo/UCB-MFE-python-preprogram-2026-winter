"""The model part, without any FastAPI: load the saved HW4 model and predict one stock."""

from pathlib import Path

import joblib
import pandas as pd

# path from this file, so it works from any folder and inside the container
ARTIFACT = Path(__file__).resolve().parent / "artifacts" / "model.joblib"


def load_bundle(path=ARTIFACT):
    """Read the artifact made by export_model.py: the fitted model, its version, the 18 names.

    joblib can run code when it loads a file, so we only load the file we created ourselves.
    """
    return joblib.load(path)


def predict_one(bundle, features):
    """Predicted 90-day return of one stock, minus the average stock. features: name -> rank."""
    # same representation as in HW4 training: a DataFrame with the same columns, same order
    row = pd.DataFrame([features], columns=bundle["features"])
    return float(bundle["model"].predict(row)[0])
