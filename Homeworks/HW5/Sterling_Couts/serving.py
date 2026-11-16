"""Artifact loading and inference, callable without HTTP or a database."""

import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent
ARTIFACT = ROOT / "artifacts/fixed_anchor_2012.joblib"
FEATURES = (
    "rank_momentum_21",
    "rank_momentum_63",
    "rank_momentum_252_21",
    "rank_volatility_63",
    "rank_drawdown_252",
    "rank_momentum_126_21",
    "rank_operating_roa",
    "rank_fcf_to_assets",
)


def load_pipeline() -> tuple[Pipeline, dict]:
    """Fail startup if the packaged model, feature order, or model stack differs.

    A checksum detects accidental replacement; it does not make an untrusted
    pickle safe. This artifact was produced and verified in our own HW4.
    """
    metadata = json.loads((ROOT / "artifacts/model_metadata.json").read_text())
    if hashlib.sha256(ARTIFACT.read_bytes()).hexdigest() != metadata["artifact_sha256"]:
        raise RuntimeError("Packaged HW4 artifact checksum does not match metadata")
    for package, expected in metadata["model_library_versions"].items():
        if version(package) != expected:
            raise RuntimeError(f"{package} must match the HW4 training version {expected}")
    if metadata["features"] != list(FEATURES):
        raise RuntimeError("Metadata feature order differs from the serving contract")
    pipeline = joblib.load(ARTIFACT)  # Trusted local HW4 artifact only.
    # The first step is 'passthrough'; its feature_names_in_ is unavailable.
    # The fitted imputer retains the original DataFrame names and ordering.
    if list(pipeline.named_steps["impute"].feature_names_in_) != list(FEATURES):
        raise RuntimeError("Fitted imputer feature names do not match the contract")
    return pipeline, metadata


def feature_frame(values: dict) -> pd.DataFrame:
    """Restore training column order and map explicit financial nulls to NaN."""
    if set(values) != set(FEATURES):
        raise ValueError("Exactly the eight named HW4 features are required")
    return pd.DataFrame(
        [[np.nan if values[name] is None else values[name] for name in FEATURES]],
        columns=list(FEATURES),
        dtype=float,
    )


def predict_one(pipeline: Pipeline, values: dict) -> float:
    """Predict only: no fitting, ranking, or database access occurs."""
    predictions = np.asarray(pipeline.predict(feature_frame(values)), dtype=float)
    if predictions.shape != (1,) or not np.isfinite(predictions).all():
        raise ValueError("The model must return one finite prediction")
    return float(predictions[0])
