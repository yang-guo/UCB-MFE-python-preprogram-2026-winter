"""Model logic for the HW5 service, kept separate from HTTP request handling.

This module knows how to load the trusted HW4 artifact and turn one validated
feature mapping into one prediction. It knows nothing about FastAPI, so the same
functions are used by the API, by the tests' direct-pipeline comparison, and by
anyone who wants to score a row from Python.
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import joblib
import lightgbm
import numpy as np
import pandas as pd
import sklearn

ROOT = Path(__file__).resolve().parent
ARTIFACT_PATH = ROOT / "artifacts" / "model.joblib"

# The 12 HW4 features, in the exact order the model was trained on
# (run_hw4.FEATURES). The model sees them as columns named "r_<feature>".
FEATURES = (
    "idio_vol_252",
    "earnings_yield",
    "debt_to_assets",
    "sales_to_price",
    "gross_profitability",
    "earnings_change",
    "mom_12_1",
    "volume_trend",
    "accruals",
    "asset_growth",
    "rev_1m",
    "beta_252",
)

# Libraries whose pickled objects are only supported on the version that wrote them.
_PINNED = {"lightgbm": lightgbm.__version__, "scikit-learn": sklearn.__version__}


@dataclass
class ModelBundle:
    """The fitted HW4 estimator plus everything needed to call it correctly."""

    estimator: lightgbm.LGBMRegressor
    model_version: str
    features: tuple[str, ...]
    columns: tuple[str, ...]


def load_bundle(path: Path = ARTIFACT_PATH) -> ModelBundle:
    """Load the trusted artifact written by export_hw4_model.py. Never retrains.

    joblib/pickle can run arbitrary code, so only load artifacts you created.
    Fails fast (the server will not start) if the library versions or the feature
    order differ from the ones the artifact was trained with.
    """
    raw = joblib.load(path)
    trained = raw["versions"]
    for lib, installed in _PINNED.items():
        if trained[lib] != installed:
            raise RuntimeError(
                f"{lib} {installed} is installed but the artifact was written by "
                f"{trained[lib]}; install the pinned requirements.txt"
            )
    if tuple(raw["features"]) != FEATURES:
        raise RuntimeError(f"artifact feature order {raw['features']} != {FEATURES}")

    estimator = raw["estimator"]
    # One request = one row: a single prediction thread avoids thread start-up and
    # oversubscription under concurrent requests. This only affects predict();
    # the fitted trees, and therefore the predictions, are unchanged.
    estimator.set_params(n_jobs=1)
    return ModelBundle(
        estimator=estimator,
        model_version=raw["model_version"],
        features=tuple(raw["features"]),
        columns=tuple(raw["columns"]),
    )


def to_frame(bundle: ModelBundle, values: Mapping[str, float | None]) -> pd.DataFrame:
    """One-row DataFrame in the training representation.

    Same column names and order as HW4's ``RankLGBM._X`` (``r_<feature>``),
    float64 dtype, and missing values as NaN, which LightGBM routes exactly as it
    did in training.
    """
    row = [np.nan if values[f] is None else float(values[f]) for f in bundle.features]
    return pd.DataFrame([row], columns=list(bundle.columns), dtype="float64")


def predict_one(bundle: ModelBundle, values: Mapping[str, float | None]) -> float:
    """Predicted 90-day excess return (fraction) for one stock."""
    frame = to_frame(bundle, values)
    prediction = float(bundle.estimator.predict(frame, validate_features=True)[0])
    if not math.isfinite(prediction):
        raise ValueError(f"model returned a non-finite prediction: {prediction}")
    return prediction
