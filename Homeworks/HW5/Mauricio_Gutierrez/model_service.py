"""Load the trusted artifact once; keep model logic separate from HTTP."""

import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent


def load_bundle(artifact_dir=None):
    artifact_dir = ROOT / "artifacts" if artifact_dir is None else Path(artifact_dir)
    metadata = json.loads((artifact_dir / "model_metadata.json").read_text(encoding="utf-8"))
    for package in ("numpy", "pandas", "scikit-learn", "scipy", "joblib", "threadpoolctl"):
        if version(package) != metadata["package_versions"][package]:
            raise RuntimeError(f"The serving version of {package} differs from HW4 training")
    model_path = artifact_dir / "model.joblib"
    if hashlib.sha256(model_path.read_bytes()).hexdigest() != metadata["model_sha256"]:
        raise RuntimeError("The model file differs from the recorded export")
    pipeline = joblib.load(model_path)
    if list(pipeline.feature_names_in_) != metadata["features"]:
        raise RuntimeError("The feature order differs from the fitted model")
    return {"pipeline": pipeline, **metadata}


def predict_return(bundle, values):
    # JSON key order does not control the model's column order.
    row = pd.DataFrame([values], columns=bundle["features"])
    predictions = np.asarray(bundle["pipeline"].predict(row), dtype=float)
    if predictions.shape != (1,) or not np.isfinite(predictions).all():
        raise ValueError("The model must return one finite prediction")
    return float(predictions[0])
