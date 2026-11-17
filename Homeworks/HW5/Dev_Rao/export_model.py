"""Build and save the HW4 model artifact (run with THIS folder's Python 3.12 venv).

    .venv\\Scripts\\python export_model.py        (Windows)
    .venv/bin/python export_model.py             (mac/linux)

The HW4 composite has no learned parameters, so "rebuilding the final fitted
pipeline exactly as frozen" means: same themes, signs, column order and math, wrapped
in a Pipeline and fitted on an empty frame (fit only records the column contract).
Before saving, it must reproduce every score in reference.json, which came straight
from HW4's final-test output (see make_reference.py). Building it under 3.12 with the
pinned library versions means the Docker image never loads a pickle made elsewhere.
"""

import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.pipeline import Pipeline

from composite import COMPOSITE_THEMES, FEATURES, CompositeScorer
from model_service import ARTIFACT_DIR, METADATA_PATH, MODEL_PATH, file_sha256

HERE = Path(__file__).resolve().parent
MODEL_VERSION = "hw4-composite-frozen-2026-11-13"
HW4_FROZEN_CONFIG_SHA256 = "140d39c7ebbb88042ce8d1293f54d3a45075e20a7d9dfa70f63a19e7fe017487"
REFERENCE_TOL = 1e-7  # HW4 scored float32 ranks (~3e-8 spacing near 0.5); we use float64


def main():
    pipeline = Pipeline([("composite", CompositeScorer())])
    pipeline.fit(pd.DataFrame(columns=FEATURES, dtype="float64"))

    reference = json.loads((HERE / "reference.json").read_text())
    assert reference["feature_order"] == FEATURES
    for case in reference["cases"]:
        row = pd.DataFrame([case["features"]], columns=FEATURES, dtype="float64")
        got = float(pipeline.predict(row)[0])
        diff = abs(got - case["hw4_score"])
        print(f"{case['signal_date']} {case['symbol']:<5} hw4={case['hw4_score']:+.9f} "
              f"served={got:+.9f} diff={diff:.2e}")  # fmt: skip
        assert diff <= REFERENCE_TOL, case

    ARTIFACT_DIR.mkdir(exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    metadata = {
        "model_version": MODEL_VERSION,
        "description": "HW4 fixed-sign multi-factor composite (value, quality, investment, "
        "momentum, low risk) of same-date percentile ranks",
        "hw4_source": "Homeworks/HW4/Dev_Rao, branch hw_4_dev_rao, commit 10d27ee",
        "hw4_frozen_config_sha256": HW4_FROZEN_CONFIG_SHA256,
        "training_cutoff": "none: no parameters are fitted; config frozen 2026-11-13 "
        "on development data 2010-2015 only",
        "refits": "none (nothing to refit); the same model scored every HW4 test signal",
        "prediction": "unitless composite score in [-0.5, 0.5]; higher = higher expected "
        "90-calendar-day forward return rank (next-session entry, 90-day hold)",
        "features": FEATURES,
        "feature_units": "same-date percentile rank among eligible stocks, in (0, 1]",
        "themes": COMPOSITE_THEMES,
        "categorical_levels": {},
        "exported_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "versions": {
            "python": platform.python_version(),
            "scikit-learn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "joblib": joblib.__version__,
        },
        "artifact_sha256": file_sha256(MODEL_PATH),
        "artifact_bytes": MODEL_PATH.stat().st_size,
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"saved {MODEL_PATH.name} ({metadata['artifact_bytes']} bytes), "
          f"sha256 {metadata['artifact_sha256'][:16]}...")  # fmt: skip


if __name__ == "__main__":
    main()
