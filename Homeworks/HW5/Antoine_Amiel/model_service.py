"""Model logic shared by the API and the tests: artifact loading and inference.

No HTTP code here. The input representation is the one used to train HW4:
a float32 numpy array, one row, columns in FEATURE_ORDER, NaN for a missing
value (``hw4.model.fit`` and ``hw4.model.predict`` call
``frame[RANK_FEATURES].to_numpy(dtype="float32")``).
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

ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"

# hw4.features.RANK_FEATURES, in training order.
FEATURE_ORDER = (
    "r_rev_5",
    "r_mom_21",
    "r_mom_63",
    "r_mom_126",
    "r_mom_252_21",
    "r_vol_63",
    "r_vol_252",
    "r_drawdown_252",
    "r_max_ret_21",
    "r_beta_252",
    "r_idio_vol_252",
    "r_log_dollar_volume_63",
    "r_gross_margin",
    "r_operating_margin",
    "r_roa",
    "r_roe",
    "r_accruals",
    "r_fcf_to_assets",
    "r_debt_to_assets",
    "r_cash_to_assets",
    "r_sales_growth",
    "r_asset_growth",
    "r_earnings_yield",
    "r_book_to_market",
    "r_fcf_yield",
    "r_filing_age",
    "r_log_size",
    "r_universe_rank_pct",
)

# Never missing in the HW4 training rows, so the pipeline has no missing-value
# indicator for them. Eligibility requires 63 consecutive daily returns, which
# defines the five price features, and every listed stock has a universe rank
# and a market capitalization. The other features can be missing (short price
# history, no recent filing); the pipeline imputes them with its training median
# and a missing-value indicator. export_model.py and the tests check this set
# against the fitted imputer.
NEVER_MISSING = frozenset(
    {
        "r_rev_5",
        "r_mom_21",
        "r_mom_63",
        "r_vol_63",
        "r_max_ret_21",
        "r_log_size",
        "r_universe_rank_pct",
    }
)
NULLABLE_FEATURES = frozenset(FEATURE_ORDER) - NEVER_MISSING


@dataclass(frozen=True)
class LoadedModel:
    pipeline: object
    metadata: dict

    @property
    def model_version(self) -> str:
        return self.metadata["model_version"]


def load_model(artifact_dir: Path = ARTIFACT_DIR) -> LoadedModel:
    """Load the exported HW4 pipeline after checking it matches its metadata."""
    metadata = json.loads((artifact_dir / "model_metadata.json").read_text())
    path = artifact_dir / metadata["artifact"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != metadata["sha256"]:
        raise RuntimeError(f"{path.name} does not match the checksum in model_metadata.json")
    if tuple(metadata["features"]) != FEATURE_ORDER:
        raise RuntimeError("feature order differs from the exported model")
    if sklearn.__version__ != metadata["versions"]["scikit-learn"]:
        raise RuntimeError(
            f"scikit-learn {sklearn.__version__} installed, "
            f"model exported with {metadata['versions']['scikit-learn']}"
        )
    pipeline = joblib.load(path)  # trusted artifact written by export_model.py
    if pipeline.n_features_in_ != len(FEATURE_ORDER):
        raise RuntimeError(f"model expects {pipeline.n_features_in_} features")
    return LoadedModel(pipeline, metadata)


def to_matrix(features: Mapping[str, float | None]) -> np.ndarray:
    """One training-format row: float32, FEATURE_ORDER columns, None -> NaN."""
    row = [np.nan if features[name] is None else features[name] for name in FEATURE_ORDER]
    return np.asarray([row], dtype=np.float32)


def predict_one(model: LoadedModel, features: Mapping[str, float | None]) -> float:
    prediction = model.pipeline.predict(to_matrix(features))
    if np.shape(prediction) != (1,):
        raise ValueError(f"expected one prediction, got shape {np.shape(prediction)}")
    value = float(prediction[0])
    if not math.isfinite(value):
        raise ValueError("non-finite prediction")
    return value
