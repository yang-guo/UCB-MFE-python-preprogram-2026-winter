"""Inference steps adapted from HW4 research.py::predict_candidate."""

import hashlib
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

FEATURE_NAMES = [
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
]
ARTIFACT_SHA256 = "eeda5faa3f14cc59c718c0471475172dcd245b79c57355df1365eb3b245e5f23"
MODEL_VERSION = "hw4-fca22b5-xgboost-eeda5faa3f14"


class HW4Predictor:
    """Apply the already-fitted preprocessing, feature mask and return scaling."""

    def __init__(self, artifact_path: Path):
        artifact_hash = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
        if artifact_hash != ARTIFACT_SHA256:
            raise ValueError("The model artifact does not match the HW4 reference model")
        trained = joblib.load(artifact_path)
        self.estimator = trained["model"]
        self.preprocessing = trained["features"]
        self.model_version = MODEL_VERSION
        trained_features = self.preprocessing["clipping"].feature_names_in_.tolist()
        if trained_features != FEATURE_NAMES:
            raise ValueError(
                "The serving feature order differs from the training feature order"
            )

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        preprocessing = self.preprocessing
        values = preprocessing["clipping"].transform(frame.loc[:, FEATURE_NAMES])
        values = preprocessing["imputer"].transform(values)
        values = preprocessing["scaler"].transform(values)
        values = values[:, preprocessing["selected"]]
        return (
            self.estimator.predict(values) * preprocessing["target_scale"]
            + preprocessing["target_mean"]
        )
