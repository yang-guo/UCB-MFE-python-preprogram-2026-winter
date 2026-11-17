"""Load the trusted local artifact and preserve the training DataFrame contract."""

import hashlib
import json
import math
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.utils.validation import check_is_fitted

FEATURES = (
    "momentum_21",
    "momentum_63",
    "momentum_252_21",
    "volatility_63",
    "drawdown_252",
    "net_margin",
    "operating_roa",
    "debt_to_assets",
    "cash_to_assets",
    "fcf_to_assets",
    "accruals_to_assets",
    "filing_age_days",
    "reversal_5",
    "volatility_21",
)
ARTIFACTS = Path(__file__).resolve().parent / "artifacts"


def feature_frame(values):
    return pd.DataFrame([values], columns=list(FEATURES), dtype=float)


@dataclass
class ModelService:
    model: object
    model_version: str

    @classmethod
    def load(cls, directory=ARTIFACTS):
        metadata = json.loads((directory / "metadata.json").read_text())
        if metadata["features"] != list(FEATURES):
            raise ValueError("Saved feature order differs from API contract")
        for package, expected in metadata["packages"].items():
            if version(package) != expected:
                raise RuntimeError(f"{package} must match training version {expected}")
        path = directory / "model.joblib"
        if hashlib.sha256(path.read_bytes()).hexdigest() != metadata["artifact_sha256"]:
            raise ValueError("Model artifact checksum mismatch")
        model = joblib.load(path)  # Trusted artifact produced by export_model.py only.
        check_is_fitted(model)
        if list(model.feature_names_in_) != list(FEATURES):
            raise ValueError("Pipeline columns differ from API contract")
        return cls(model=model, model_version=metadata["model_version"])

    def predict(self, values):
        result = np.asarray(self.model.predict(feature_frame(values)))
        if result.shape != (1,):
            raise ValueError("Expected exactly one prediction")
        prediction = float(result[0])
        if not math.isfinite(prediction):
            raise ValueError("Model returned a non-finite prediction")
        return prediction
