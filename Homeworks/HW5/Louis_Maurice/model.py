"""Model logic, kept separate from HTTP handling in app.py."""

from pathlib import Path

import joblib
import pandas as pd
import sklearn

# Same names and order as course_data.FEATURES used to train HW4.
NUMERIC_FEATURES = [
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
FEATURES = NUMERIC_FEATURES + ["sector"]
ARTIFACT_PATH = Path(__file__).resolve().parent / "artifacts" / "model.joblib"


def load_bundle(path: Path = ARTIFACT_PATH) -> dict:
    """Load the trusted HW4 bundle (pipeline + metadata) created by export_model.py."""
    # joblib uses pickle: only load artifacts you created yourself.
    bundle = joblib.load(path)
    if bundle["sklearn_version"] != sklearn.__version__:
        raise RuntimeError(
            f"Model saved with scikit-learn {bundle['sklearn_version']}, "
            f"but {sklearn.__version__} is installed"
        )
    if bundle["features"] != FEATURES:
        raise RuntimeError("Artifact feature list does not match the service contract")
    return bundle


def to_frame(features: dict) -> pd.DataFrame:
    """One-row DataFrame with the training column names, order, and dtypes.

    Missing (None) numeric values become NaN, which the saved pipeline imputes with
    the training medians exactly as in HW4.
    """
    frame = pd.DataFrame([features], columns=FEATURES)
    frame[NUMERIC_FEATURES] = frame[NUMERIC_FEATURES].astype(float)
    return frame


def predict_one(pipeline, features: dict) -> float:
    """Predicted 90-day return (fraction) for one stock."""
    return float(pipeline.predict(to_frame(features))[0])
