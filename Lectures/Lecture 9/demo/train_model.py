"""Make a tiny deterministic teaching artifact; these are synthetic data, not a forecast."""

import json
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
FEATURES = ["momentum_20d", "volatility_20d"]
MODEL_VERSION = "classroom-v1"


def main():
    rows = np.array([[m, v] for m in [-0.1, 0.0, 0.1] for v in [0.05, 0.15, 0.3]])
    targets = 0.4 * rows[:, 0] - 0.1 * rows[:, 1] + 0.002
    pipeline = make_pipeline(StandardScaler(), LinearRegression())
    pipeline.fit(rows, targets)
    bundle = {"pipeline": pipeline, "model_version": MODEL_VERSION, "features": FEATURES}
    (ROOT / "artifacts").mkdir(exist_ok=True)
    joblib.dump(bundle, ROOT / "artifacts/model.joblib")
    metadata = {
        "model_version": MODEL_VERSION,
        "features": FEATURES,
        "sklearn_version": sklearn.__version__,
        "data": "synthetic 3-by-3 grid",
        "target": "0.4 * momentum_20d - 0.1 * volatility_20d + 0.002",
        "units": "Returns are fractions; volatility is annualized, as a fraction.",
    }
    (ROOT / "artifacts/model_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print("Prepared classroom-v1; no network or external data required.")


if __name__ == "__main__":
    main()
