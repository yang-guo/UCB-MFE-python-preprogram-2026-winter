"""Load the versioned HW4 pipeline and preserve its DataFrame contract."""

import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import joblib
import pandas as pd
from sklearn.utils.validation import check_is_fitted

FEATURES = (
    "ret21",
    "ret63",
    "ret126",
    "ret252skip21",
    "vol63",
    "op_margin",
    "cashflow_assets",
    "debt_assets",
)
MODEL_VERSION = "hw4-ridge-20160401-v1"
MODEL_PACKAGES = ("numpy", "pandas", "scikit-learn", "scipy", "joblib", "threadpoolctl")
ARTIFACT = Path(__file__).resolve().parent / "artifacts/hw4_ridge.joblib"


def load_pipeline():
    manifest = json.loads(ARTIFACT.with_suffix(".json").read_text(encoding="utf-8"))
    if manifest["model_version"] != MODEL_VERSION or tuple(manifest["features"]) != FEATURES:
        raise RuntimeError("Model manifest does not match the service contract")
    installed = {name: version(name) for name in MODEL_PACKAGES}
    if installed != manifest["packages"]:
        raise RuntimeError(f"HW4 model package mismatch: installed {installed}")
    if hashlib.sha256(ARTIFACT.read_bytes()).hexdigest() != manifest["sha256"]:
        raise RuntimeError("HW4 model checksum mismatch")
    pipeline = joblib.load(ARTIFACT)
    check_is_fitted(pipeline)
    if tuple(pipeline.feature_names_in_) != FEATURES:
        raise RuntimeError("Saved pipeline feature order does not match the service contract")
    return pipeline


def feature_frame(values):
    # Explicit float dtype converts JSON null into the NaN expected by the imputer.
    return pd.DataFrame([values], columns=FEATURES, dtype="float64")
