"""Model logic kept separate from request handling: load the trusted artifact,
check it against its metadata, and score one row in the training representation."""

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = ROOT / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "model.joblib"
METADATA_PATH = ARTIFACT_DIR / "model_metadata.json"


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass
class ModelBundle:
    pipeline: object
    model_version: str
    features: list[str]
    metadata: dict


def load_bundle(
    model_path: Path = MODEL_PATH, metadata_path: Path = METADATA_PATH
) -> ModelBundle:
    """Load the artifact once. Refuse it if its hash or column order disagree with
    the metadata written at export time (only load artifacts we created)."""
    metadata = json.loads(metadata_path.read_text())
    actual = file_sha256(model_path)
    if actual != metadata["artifact_sha256"]:
        raise RuntimeError(f"model artifact hash mismatch: {actual} != metadata")
    pipeline = joblib.load(model_path)
    features = list(metadata["features"])
    if list(pipeline.feature_names_in_) != features:
        raise RuntimeError("artifact feature order differs from metadata")
    return ModelBundle(pipeline, metadata["model_version"], features, metadata)


def to_frame(values: dict, features: list[str]) -> pd.DataFrame:
    """One row with the exact training column names and order; None -> NaN."""
    row = {
        name: (math.nan if values[name] is None else float(values[name])) for name in features
    }
    return pd.DataFrame([row], columns=features, dtype="float64")


def predict_one(bundle: ModelBundle, values: dict) -> float:
    """Score one stock. Raises on anything unexpected; never returns a default."""
    prediction = bundle.pipeline.predict(to_frame(values, bundle.features))
    if len(prediction) != 1:
        raise RuntimeError(f"expected 1 prediction, got {len(prediction)}")
    value = float(prediction[0])
    # the composite is a mean of terms in [-0.5, 0.5]; anything else means a broken model
    if not math.isfinite(value) or abs(value) > 0.5:
        raise RuntimeError(f"invalid prediction {value!r}")
    return value
