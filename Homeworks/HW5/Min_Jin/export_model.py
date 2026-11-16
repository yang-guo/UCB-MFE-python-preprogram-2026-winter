"""
Save the HW4 model artifact and record a reference prediction from the HW4 code.

Steps
1. Save artifacts/factor_score.joblib: the frozen HW4 FactorScore plus the
   feature order, model version and library versions it was saved with.
2. If the HW4 folder and its feature panel exist, take one row (AAPL on the
   last signal date, 2026-08-31), score it with HW4's own model.FactorScore,
   and save the input and the HW4 score to reference_example.json. The tests
   compare the API against this recorded number.
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from factor_model import FACTOR_FEATURES, FactorScore, to_frame

# define saving path and reference path
HERE = Path(__file__).resolve().parent
ARTIFACT_PATH = HERE / "artifacts" / "factor_score.joblib"
REFERENCE_PATH = HERE / "reference_example.json"
HW4_DIR = HERE.parents[1] / "HW4" / "Min_Jin"

MODEL_VERSION = "hw4-factor-score-v1"
REFERENCE_SYMBOL = "AAPL"
REFERENCE_DATE = "2026-08-31"


def save_artifact():
    """Save the frozen HW4 model with the metadata the API checks at startup."""
    artifact = {
        "model": FactorScore().fit(None),
        "features": FACTOR_FEATURES,
        "model_version": MODEL_VERSION,
        "hw4_commit": "7934f43",
        "sklearn_version": sklearn.__version__,
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
    }
    ARTIFACT_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(artifact, ARTIFACT_PATH)
    print(
        f"Saved {ARTIFACT_PATH.relative_to(HERE)} ({MODEL_VERSION}, sklearn {sklearn.__version__})"
    )
    return artifact


def record_reference(artifact):
    """Score one real HW4 row with the HW4 code itself and save the input + score."""
    panel_path = HW4_DIR / "outputs" / "panel.parquet"
    # catch not found error
    if not panel_path.is_file():
        print(f"Skipping reference: {panel_path} not found (needs the HW4 feature panel).")
        return
    # import HW4 model.py
    sys.path.insert(0, str(HW4_DIR))
    from model import FINAL_MODEL, candidates

    # check feature order
    hw4_model, hw4_features, _ = candidates[FINAL_MODEL]
    assert hw4_features == FACTOR_FEATURES, "feature order differs from HW4"

    # load AAPL's 5 factor ranks on the reference date from the HW4 panel
    panel = pd.read_parquet(panel_path, columns=["date", "symbol", *FACTOR_FEATURES])
    row = panel.loc[panel.date.eq(REFERENCE_DATE) & panel.symbol.eq(REFERENCE_SYMBOL)]
    assert len(row) == 1, "reference row not found"
    features = row[FACTOR_FEATURES].iloc[0].astype(float).to_dict()

    # check that the saved model gives the same score as the HW4 code
    hw4_score = float(hw4_model.fit(None).predict(row[FACTOR_FEATURES])[0])
    served_score = float(artifact["model"].predict(to_frame([features]))[0])
    assert served_score == hw4_score, "served model disagrees with HW4"

    # save the reference input and score for the tests and example request for curl
    reference = {
        "source": f"HW4 model.FactorScore on {REFERENCE_SYMBOL} at signal date {REFERENCE_DATE}",
        "hw4_commit": artifact["hw4_commit"],
        "input": features,
        "expected_prediction": hw4_score,
    }
    REFERENCE_PATH.write_text(json.dumps(reference, indent=2) + "\n")
    (HERE / "example_request.json").write_text(json.dumps(features, indent=2) + "\n")
    print(f"Recorded {REFERENCE_SYMBOL} {REFERENCE_DATE}: HW4 score = {hw4_score!r}")


if __name__ == "__main__":
    record_reference(save_artifact())
