"""Model logic for my HW4 return-rank model: load the saved pipeline and score one stock.

Nothing here knows about HTTP. ``app.py`` turns a validated request into the ``features``
mapping used below, and turns the returned number (or the raised exception) into a response.
"""

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.pipeline import Pipeline

# Resolved from this file, so the service works from any working directory.
ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "hw4_ridge_rank.joblib"
METADATA_PATH = ARTIFACT_DIR / "model_metadata.json"


@dataclass(frozen=True)
class LoadedModel:
    """The fitted HW4 pipeline with the two facts needed to call it correctly."""

    pipeline: Pipeline  # missing rank -> 0.0, then the Ridge fitted in HW4
    model_version: str
    features: tuple[str, ...]  # column order the Ridge was trained with


def load_model(
    model_path: Path = MODEL_PATH, metadata_path: Path = METADATA_PATH
) -> LoadedModel:
    """Load the HW4 artifact written by ``export_hw4_model.py``, and nothing else.

    Unpickling runs code, so two checks come first: the file must be byte-for-byte the one I
    exported (SHA-256 recorded in the metadata), and scikit-learn must be the version that
    fitted it, because loading a model with another version is unsupported.
    """
    metadata = json.loads(metadata_path.read_text())
    if hashlib.sha256(model_path.read_bytes()).hexdigest() != metadata["artifact_sha256"]:
        raise RuntimeError(f"{model_path.name} is not the artifact described by the metadata")
    trained_with = metadata["versions"]["scikit-learn"]
    if sklearn.__version__ != trained_with:
        raise RuntimeError(
            f"The model was fitted with scikit-learn {trained_with}; "
            f"this environment has {sklearn.__version__}"
        )
    bundle = joblib.load(model_path)
    return LoadedModel(bundle["pipeline"], bundle["model_version"], tuple(bundle["features"]))


def predict_one(model: LoadedModel, features: Mapping[str, float | None]) -> float:
    """Return the predicted 90-day return rank of one stock.

    ``features`` gives a centred rank for every name in ``model.features``, or None where HW4
    would have had a missing value. The row is built the way HW4's ``design_matrix`` built it:
    training column order, float32, missing left as NaN for the pipeline to set to 0.0.
    """
    ranks = [features[name] for name in model.features]
    row = np.array([[np.nan if rank is None else rank for rank in ranks]], dtype=np.float32)
    prediction = float(model.pipeline.predict(row)[0])
    if not math.isfinite(prediction):
        raise ValueError("The model returned a non-finite prediction")
    return prediction
