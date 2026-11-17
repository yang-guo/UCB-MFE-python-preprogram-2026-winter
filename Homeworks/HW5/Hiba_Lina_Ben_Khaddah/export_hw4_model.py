"""Export the fitted HW4 model and one reference prediction.

Call ``export_model(fixed_model, test_candidates, FEATURES)`` from the executed
HW4 final-backtest notebook. It does not refit the model.
"""

import json
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

MODEL_VERSION = "hiba-hw4-rank-tree-v1"
FEATURE_ORDER = [
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
    "sector",
]


def json_value(value):
    if pd.isna(value):
        return None
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def export_model(fixed_model, test_candidates, features, output_dir=None):
    """Save the fitted pipeline, metadata, and one known HW4 prediction."""

    if list(features) != FEATURE_ORDER:
        raise RuntimeError("The exported feature order does not match HW4 FEATURES")

    if output_dir is None:
        output_dir = Path(__file__).resolve().parent / "artifacts"
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    artifact_path = output_dir / "model.joblib"
    reference_path = output_dir / "reference_prediction.json"
    metadata_path = output_dir / "model_metadata.json"
    joblib.dump(fixed_model, artifact_path)

    reference_row = (
        test_candidates.sort_values(["date", "symbol"]).dropna(subset=["score"]).iloc[0]
    )
    reference_frame = pd.DataFrame(
        [{feature: reference_row[feature] for feature in FEATURE_ORDER}],
        columns=FEATURE_ORDER,
    )
    expected_prediction = float(fixed_model.predict(reference_frame)[0])

    reference = {
        "model_version": MODEL_VERSION,
        "source": {
            "symbol": str(reference_row["symbol"]),
            "signal_date": str(pd.Timestamp(reference_row["date"]).date()),
            "notebook": "HW4/02_final_backtest.ipynb",
        },
        "input": {feature: json_value(reference_row[feature]) for feature in FEATURE_ORDER},
        "expected_prediction": expected_prediction,
    }
    reference_path.write_text(json.dumps(reference, indent=2) + "\n", encoding="utf-8")

    metadata = {
        "model_version": MODEL_VERSION,
        "model_type": "HistGradientBoostingRegressor with fitted preprocessing",
        "prediction": "within-date percentile rank of the 90-calendar-day return minus 0.5",
        "feature_order": FEATURE_ORDER,
        "fit_cutoff": "2016-03-31",
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "joblib": joblib.__version__,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    print(f"Saved model: {artifact_path}")
    print(f"Saved reference: {reference_path}")
    print(f"Expected prediction: {expected_prediction:.12f}")
