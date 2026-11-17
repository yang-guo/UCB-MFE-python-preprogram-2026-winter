"""Model logic for the service. It has no web code, so it can be tested and reused alone."""

import json
from pathlib import Path

import joblib
import pandas as pd
import sklearn

ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"


def load_bundle(artifact_dir=ARTIFACT_DIR):
    """Load the fitted HW4 pipeline one time. Load only artifacts that you made or trust.

    The bundle is a dict with the keys `pipeline`, `model_version`, and `features`.
    sklearn does not support loading a model with a different sklearn version, so this
    function stops when the installed version is not the training version.
    """
    artifact_dir = Path(artifact_dir)
    metadata = json.loads((artifact_dir / "model_metadata.json").read_text())
    trained_with = metadata["versions"]["scikit-learn"]
    if sklearn.__version__ != trained_with:
        raise RuntimeError(
            f"The model was fitted with scikit-learn {trained_with}, "
            f"but this environment has {sklearn.__version__}"
        )
    bundle = joblib.load(artifact_dir / "model.joblib")
    if bundle["model_version"] != metadata["model_version"]:
        raise RuntimeError("model.joblib and model_metadata.json are from different exports")
    return bundle


def predict_return(bundle, values):
    """Predict the 90-day return (a fraction) for one stock.

    `values` maps each feature name to a number. The pipeline selects its columns by name,
    so the input must be a DataFrame with the training column names, in the training order.
    """
    frame = pd.DataFrame([values], columns=bundle["features"], dtype=float)
    return float(bundle["pipeline"].predict(frame)[0])
