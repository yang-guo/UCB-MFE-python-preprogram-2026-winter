"""Load and run the frozen HW4 model.  No web code here; request handling is in app.py.

The artifact is the estimator that HW4's own walk-forward code fitted for the submitted
strategy (experiment ``P1_top50__R1-f06421a6``); ``scripts/export_hw4_model.py`` wrote it.
It is a ``quantlab.models.FixedComposite``: the signed average of seven same-date
percentile ranks, with a missing rank (NaN) counted as the median (0).

HW4 fed the model a float64 NumPy array with the columns in ``FEATURES`` order
(``panel[features].to_numpy(float)``); ``LoadedModel.predict_one`` builds exactly that.
"""

from __future__ import annotations

import hashlib
import json
import math
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import sklearn
from sklearn.exceptions import InconsistentVersionWarning

# Resolved relative to this file, so the service works from any working directory.
ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"

# HW4 model input columns, in training order.  load_model checks this against the
# metadata written at export time, so a reordering cannot go unnoticed.
FEATURES: tuple[str, ...] = (
    "gross_profitability_csr",
    "operating_roa_csr",
    "accruals_to_assets_csr",
    "book_to_market_csr",
    "vol_63_csr",
    "mom_252_21_csr",
    "ret_21_csr",
)


class ArtifactError(RuntimeError):
    """The artifact is missing, altered, or incompatible with this environment."""


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class LoadedModel:
    """The fitted HW4 estimator plus the metadata recorded when it was exported."""

    estimator: Any
    metadata: dict

    @property
    def version(self) -> str:
        return self.metadata["model_version"]

    def predict_one(self, values: Sequence[float | None]) -> float:
        """Score one stock.  ``values`` follows ``FEATURES``; None means unavailable."""
        if len(values) != len(FEATURES):
            raise ValueError(f"Expected {len(FEATURES)} features, got {len(values)}")
        row = np.array([[np.nan if v is None else v for v in values]], dtype=float)
        output = np.asarray(self.estimator.predict(row), dtype=float)
        if output.shape != (1,):
            raise ValueError(f"Expected one prediction, got shape {output.shape}")
        prediction = float(output[0])
        if not math.isfinite(prediction):
            raise ValueError("Model returned a non-finite prediction")
        return prediction


def load_model(artifact_dir: Path = ARTIFACT_DIR) -> LoadedModel:
    """Verify, then unpickle, the trusted HW4 artifact.  Never fits or refits anything."""
    metadata = json.loads((artifact_dir / "model_metadata.json").read_text())
    path = artifact_dir / metadata["artifact"]

    # Unpickling runs code, so check the file is the one we exported before loading it.
    digest = file_sha256(path)
    if digest != metadata["artifact_sha256"]:
        raise ArtifactError(f"{path.name} SHA-256 {digest[:12]} does not match the metadata")
    # sklearn does not support loading a pickle from another sklearn version.
    if sklearn.__version__ != metadata["sklearn_version"]:
        raise ArtifactError(
            f"scikit-learn {sklearn.__version__} installed, artifact was saved with "
            f"{metadata['sklearn_version']}; install the versions in requirements.txt"
        )
    with warnings.catch_warnings():
        warnings.simplefilter("error", InconsistentVersionWarning)
        estimator = joblib.load(path)

    class_path = f"{type(estimator).__module__}.{type(estimator).__qualname__}"
    if class_path != metadata["model_class"]:
        raise ArtifactError(f"Loaded {class_path}, expected {metadata['model_class']}")
    if tuple(metadata["features"]) != FEATURES:
        raise ArtifactError("Artifact feature order differs from the service's FEATURES")
    if getattr(estimator, "n_features_in_", None) != len(FEATURES):
        raise ArtifactError("Estimator was fitted on a different number of features")
    return LoadedModel(estimator=estimator, metadata=metadata)
