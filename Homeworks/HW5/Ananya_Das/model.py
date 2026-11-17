"""Inference layer for my HW4 stock-return model, kept separate from HTTP handling.

The saved artifact (artifacts/model.joblib, made by export_model.py) holds the fitted HW4
pipeline, SimpleImputer(constant 0.5) -> Ridge(alpha=1e7), its version, and the feature
order. This module only loads and scores it: it never fits, and it does not import hw4.py,
DuckDB, or matplotlib. Load only artifacts you created or trust (joblib uses pickle).
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

MODEL_PATH = Path(__file__).resolve().parent / "artifacts" / "model.joblib"


class InferenceError(RuntimeError):
    """The model ran but did not produce a usable prediction."""


@dataclass(frozen=True)
class Prediction:
    value: float  # predicted 90-day return minus the date's median stock, as a fraction
    model_version: str


class StockReturnModel:
    """The fitted HW4 pipeline with the exact feature order it was trained on."""

    def __init__(self, pipeline: Pipeline, model_version: str, features: list[str]) -> None:
        if list(getattr(pipeline, "feature_names_in_", [])) != features:
            raise ValueError("artifact feature order does not match the fitted pipeline")
        self.pipeline = pipeline
        self.model_version = model_version
        self.features = features

    @classmethod
    def load(cls, path: Path = MODEL_PATH) -> "StockReturnModel":
        """Read the trusted joblib bundle written by export_model.py."""
        bundle = joblib.load(path)
        return cls(bundle["pipeline"], str(bundle["model_version"]), list(bundle["features"]))

    def to_frame(self, values: Mapping[str, float | None]) -> pd.DataFrame:
        """One row with HW4's column names and order; None becomes NaN for the imputer."""
        if set(values) != set(self.features):
            raise ValueError("features must be exactly the model's eight features")
        row = [
            np.nan if values[name] is None else float(values[name]) for name in self.features
        ]
        return pd.DataFrame([row], columns=self.features, dtype=float)

    def predict(self, values: Mapping[str, float | None]) -> Prediction:
        """Score one stock. Failures raise; they never become a default prediction."""
        value = float(self.pipeline.predict(self.to_frame(values))[0])
        if not math.isfinite(value):
            raise InferenceError("the model returned a non-finite prediction")
        return Prediction(value=value, model_version=self.model_version)
