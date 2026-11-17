"""Load the saved HW4 model and predict with it. No HTTP code lives here."""

import json
import math
from pathlib import Path

import joblib
import pandas as pd
import sklearn

ARTIFACTS = Path(__file__).resolve().parent / "artifacts"

# The 20 HW4 features, in training order (HW4 research.FEATURES).
FEATURES = (
    "mom_1m",
    "mom_3m",
    "mom_6m_skip1m",
    "mom_12m_skip1m",
    "vol_3m",
    "vol_12m",
    "drawdown_12m",
    "max_ret_1m",
    "beta_12m",
    "idio_vol_12m",
    "log_dollar_volume",
    "roa",
    "accruals",
    "earnings_yield",
    "sales_yield",
    "fcf_yield",
    "debt_to_assets",
    "cash_to_assets",
    "asset_growth",
    "sales_growth",
)


def load_bundle(artifacts: Path = ARTIFACTS) -> dict:
    """Load the fitted model once. Fail at startup if it can't be trusted to predict.

    joblib files run code when loaded, so this only ever loads the file that
    export_model.py wrote and that ships with this folder.
    """
    metadata = json.loads((artifacts / "model_metadata.json").read_text())
    if sklearn.__version__ != metadata["sklearn_version"]:
        raise RuntimeError(
            f"model was saved with scikit-learn {metadata['sklearn_version']} "
            f"but this environment has {sklearn.__version__}"
        )
    bundle = joblib.load(artifacts / "model.joblib")
    if (
        tuple(bundle["features"]) != FEATURES
        or tuple(bundle["model"].feature_names_in_) != FEATURES
    ):
        raise RuntimeError("saved model's features differ from the service's feature list")
    return bundle


def predict_one(bundle: dict, features: dict[str, float]) -> float:
    """Predict for one stock. The model was trained on a DataFrame with named
    columns, so it gets the same thing here, in training order."""
    frame = pd.DataFrame([[features[name] for name in FEATURES]], columns=list(FEATURES))
    prediction = float(bundle["model"].predict(frame)[0])
    if not math.isfinite(prediction):
        raise ValueError("model returned a non-finite prediction")
    return prediction
