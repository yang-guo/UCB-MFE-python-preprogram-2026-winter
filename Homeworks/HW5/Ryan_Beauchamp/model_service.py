"""Reusable model logic for the HW5 service (no FastAPI here).

load_artifact checks that the saved HW4 pipeline is the exact file we exported and
that this environment has the library versions it was trained with, then loads it.
predict_one turns one validated feature dict into one finite score.
"""

import hashlib
import json
import math
from importlib.metadata import version
from pathlib import Path

import joblib
import pandas as pd

# The pickle refers to quantile_clipper.QuantileClipper; importing it here makes that
# dependency explicit and fails at import time if the module is missing.
import quantile_clipper  # noqa: F401

MODEL_FILE = "model.joblib"
METADATA_FILE = "model_metadata.json"

# metadata["library_versions"] key -> installed distribution name to compare against.
CHECKED_LIBRARIES = {
    "sklearn": "scikit-learn",
    "numpy": "numpy",
    "pandas": "pandas",
    "scipy": "scipy",
    "joblib": "joblib",
}


def load_artifact(directory):
    """Return (pipeline, metadata) after verifying the file hash and library versions."""
    directory = Path(directory)
    # Metadata written by export_model.py when the model was exported.
    metadata = json.loads((directory / METADATA_FILE).read_text())

    # 1. The model file must be byte-for-byte the one recorded in the metadata.
    model_bytes = (directory / MODEL_FILE).read_bytes()
    digest = hashlib.sha256(model_bytes).hexdigest()
    if digest != metadata["model_sha256"]:
        raise RuntimeError(
            f"{MODEL_FILE} sha256 {digest} does not match metadata {metadata['model_sha256']}"
        )

    # 2. sklearn pickles are only reliable with the versions used for training.
    expected = metadata["library_versions"]
    mismatches = [
        f"{key}: installed {version(dist)}, trained with {expected[key]}"
        for key, dist in CHECKED_LIBRARIES.items()
        if version(dist) != expected[key]
    ]
    if mismatches:
        raise RuntimeError("Library versions differ from training: " + "; ".join(mismatches))

    # 3. Only now unpickle (the file is our own trusted, verified artifact).
    pipeline = joblib.load(directory / MODEL_FILE)
    return pipeline, metadata


def predict_one(pipeline, feature_order, features):
    """Score one row; features is a dict whose keys are exactly feature_order."""
    # A missing key would otherwise silently become a NaN column.
    if set(features) != set(feature_order):
        raise ValueError(f"features must have exactly these keys: {feature_order}")
    # One-row DataFrame with the training column names and order.
    row = pd.DataFrame([features], columns=feature_order)
    prediction = float(pipeline.predict(row)[0])
    # Never return a non-finite score.
    if not math.isfinite(prediction):
        raise ValueError(f"model returned a non-finite prediction: {prediction}")
    return prediction
