"""Reusable model logic for the HW4 stock-ranking model (no HTTP code here).

The HW4 model (``hw4lib.model.RidgeRanker`` with ``alpha=1000`` and the
``y_rank`` target) has two stages:

1. **Same-date cross-sectional ranking** (``hw4lib.model.add_model_columns``):
   every raw feature becomes its percentile rank among the ~1,100-1,900 eligible
   stocks on the same signal date. This step learns nothing across time and
   needs the whole cross-section, so it stays upstream in the HW4 feature
   pipeline. The service receives the ranked ``r_*`` values, which are exactly
   the columns the HW4 model was trained on.
2. **The model design and the Ridge fit** (``RidgeRanker._design`` + ``Ridge``):
   missing rank -> 0.5 (the median), centring by -0.5, one missing-value flag per
   feature, and sector one-hot columns, followed by the fitted Ridge
   coefficients. ``RankFeatureDesign`` below reproduces ``_design`` exactly, and
   the export script checks this against the HW4 code.

The artifact is a ``sklearn.pipeline.Pipeline([RankFeatureDesign, Ridge])``.
Pickle stores this class by its import path, ``model_pipeline.RankFeatureDesign``,
so this module must be importable wherever the artifact is loaded, including the
Docker image.
"""

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

ROOT = Path(__file__).resolve().parent
ARTIFACT_PATH = ROOT / "artifacts" / "model.joblib"
METADATA_PATH = ROOT / "artifacts" / "model_metadata.json"

# Feature contract. The order is hw4lib.features.NUMERIC_FEATURES with the "r_"
# prefix that hw4lib.model.add_model_columns adds; the model was fitted on this order.
RANK_FEATURES = [
    "r_mom_5",
    "r_mom_21",
    "r_mom_63",
    "r_mom_126",
    "r_mom_252_21",
    "r_vol_63",
    "r_vol_252",
    "r_max_ret_21",
    "r_drawdown_252",
    "r_beta_252",
    "r_log_dollar_volume",
    "r_gp_to_assets",
    "r_roa",
    "r_roe",
    "r_op_margin",
    "r_accruals",
    "r_fcf_to_assets",
    "r_debt_to_assets",
    "r_cash_to_assets",
    "r_asset_growth",
    "r_sales_growth",
    "r_net_buyback",
    "r_log_mcap",
    "r_earnings_yield",
    "r_book_to_price",
    "r_sales_to_price",
    "r_fcf_yield",
    "r_filing_age_days",
]
# hw4lib.model.SECTORS: the categorical levels in their training order. "Unknown"
# is HW4's label for a symbol without a sector; it never occurred in training rows.
SECTORS = [
    "Basic Materials",
    "Communication Services",
    "Consumer Cyclical",
    "Consumer Defensive",
    "Energy",
    "Financial Services",
    "Healthcare",
    "Industrials",
    "Real Estate",
    "Technology",
    "Utilities",
    "Unknown",
]
INPUT_COLUMNS = RANK_FEATURES + ["sector"]


class RankFeatureDesign(TransformerMixin, BaseEstimator):
    """Turn ranked features + sector into HW4's Ridge design matrix (68 columns).

    Same output as ``hw4lib.model.RidgeRanker._design`` (my HW4 code), packaged as
    an sklearn transformer so that it is serialized together with the Ridge fit:

    * 28 centred ranks: ``rank.fillna(0.5) - 0.5``;
    * 28 missing flags: ``1.0`` where the rank was missing, else ``0.0``;
    * 12 sector dummies in ``SECTORS`` order.

    It has no fitted statistics; ``fit`` only records and checks the expected
    input columns so that ``transform`` can reject a frame with the wrong
    columns or column order instead of predicting from shifted features.
    """

    def fit(self, X, y=None):
        self._check_columns(X, INPUT_COLUMNS)
        self.feature_names_in_ = np.array(INPUT_COLUMNS, dtype=object)
        self.n_features_in_ = len(INPUT_COLUMNS)
        self.sectors_ = list(SECTORS)
        return self

    def transform(self, X):
        check_is_fitted(self, "feature_names_in_")
        self._check_columns(X, list(self.feature_names_in_))
        ranks = X[RANK_FEATURES].astype(float)
        centred = ranks.fillna(0.5) - 0.5
        missing = ranks.isna().astype(float).add_suffix("_missing")
        if not X["sector"].isin(self.sectors_).all():
            raise ValueError("Unknown or missing sector value")
        sector = pd.Categorical(X["sector"], categories=self.sectors_)
        dummies = pd.get_dummies(sector, prefix="sector", dtype=float)
        dummies.index = X.index
        return pd.concat([centred, missing, dummies], axis=1).to_numpy()

    def get_feature_names_out(self, input_features=None):
        return np.array(
            [f"{c}_centred" for c in RANK_FEATURES]
            + [f"{c}_missing" for c in RANK_FEATURES]
            + [f"sector_{s}" for s in SECTORS],
            dtype=object,
        )

    @staticmethod
    def _check_columns(X, expected):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("RankFeatureDesign expects a pandas DataFrame with named columns")
        if list(X.columns) != list(expected):
            raise ValueError(f"Expected columns {expected}, got {list(X.columns)}")


@dataclass(frozen=True)
class ModelBundle:
    """The loaded artifact plus what the API needs to report about it."""

    pipeline: Pipeline
    model_version: str
    metadata: dict


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_bundle(
    artifact_path: Path = ARTIFACT_PATH, metadata_path: Path = METADATA_PATH
) -> ModelBundle:
    """Load the trusted artifact once, refusing a modified file or a version mismatch.

    joblib/pickle can execute code while loading, so the file's SHA-256 is
    checked against the hash recorded by ``scripts/export_hw4_model.py``
    before the file is unpickled.
    """
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    digest = sha256_of(artifact_path)
    if digest != metadata["artifact_sha256"]:
        raise RuntimeError(f"Artifact hash {digest} does not match model_metadata.json")
    trained_with = metadata["library_versions"]["scikit-learn"]
    if sklearn.__version__ != trained_with:
        raise RuntimeError(
            f"scikit-learn {sklearn.__version__} installed, artifact saved with {trained_with}"
        )
    pipeline = joblib.load(artifact_path)
    if list(pipeline.named_steps["design"].feature_names_in_) != INPUT_COLUMNS:
        raise RuntimeError("Artifact feature order differs from model_pipeline.INPUT_COLUMNS")
    return ModelBundle(pipeline, metadata["model_version"], metadata)


def to_model_frame(features: dict) -> pd.DataFrame:
    """One request -> a one-row DataFrame with the training column names and order.

    ``None`` becomes NaN, the same missing-value representation HW4 trained on.
    """
    row = {name: features[name] for name in INPUT_COLUMNS}
    frame = pd.DataFrame([row], columns=INPUT_COLUMNS)
    frame[RANK_FEATURES] = frame[RANK_FEATURES].astype(float)
    return frame


def predict_score(pipeline: Pipeline, features: dict) -> float:
    """Return one finite score. Raises if inference fails or produces a non-finite value."""
    output = pipeline.predict(to_model_frame(features))
    if np.shape(output) != (1,):
        raise ValueError(f"Expected one prediction, got shape {np.shape(output)}")
    score = float(output[0])
    if not math.isfinite(score):
        raise ValueError(f"Model produced a non-finite prediction: {score}")
    return score
