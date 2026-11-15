"""Re-create the last HW4 model fit with the HW4 code and save it for serving.

Run once, from the repository root, in the HW4 training environment (the root
``.venv`` built by ``uv sync --locked``), with the HW4 submission and the course
database in place:

    .venv/bin/python Homeworks/HW5/Oscar_Sallusti/export_model.py

HW4 refits its Ridge model once a year and did not save the fitted objects. This
script repeats the final refit (first signal date on/after 2026-03-31) by
calling HW4's own ``build_panel`` and ``fit``, checks the training-row count
against HW4's ``results/refits.csv``, and writes:

    artifacts/model.joblib            fitted sklearn Pipeline + version + contract
    artifacts/model_metadata.json     human-readable provenance
    example_request.json              a real HW4 feature row (AAPL, last signal date)
    tests/reference_prediction.json   that row's score from HW4's predict()

It is not part of the service or the Docker image.
"""

import hashlib
import json
import platform
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

HERE = Path(__file__).resolve().parent
HW4 = HERE.parents[1] / "HW4" / "Oscar_Sallusti"
sys.path.insert(0, str(HW4))
sys.path.insert(0, str(HERE))

import config as hw4_config  # noqa: E402
import features as hw4_features  # noqa: E402
import model as hw4_model  # noqa: E402
from final_test import DEFAULT_DB, refit_dates  # noqa: E402

from model_service import FEATURES, SECTORS  # noqa: E402

EXAMPLE_SYMBOL = "AAPL"


def main() -> None:
    assert tuple(hw4_features.FEATURES) == FEATURES, "feature order differs from HW4"
    assert tuple(hw4_model.SECTORS) == SECTORS, "sector list differs from HW4"

    market = hw4_features.load_market(DEFAULT_DB, end=hw4_config.TEST_END)
    events = hw4_features.load_fundamental_events(DEFAULT_DB, end=hw4_config.TEST_END)
    panel = hw4_features.build_panel(market, events, "2010-01-01", hw4_config.TEST_END)

    fit_date = refit_dates(market["calendar"])[-1]
    pipeline, train_rows = hw4_model.fit(panel, fit_date, hw4_config.MODEL)

    refits = pd.read_csv(HW4 / "results" / "refits.csv", parse_dates=["fit_date"])
    recorded = refits.loc[refits.fit_date == fit_date, "train_rows"].item()
    assert train_rows == recorded, f"{train_rows} training rows, HW4 recorded {recorded}"

    # Features that were missing somewhere in training have a fitted missing-value
    # indicator. Only those may be null in a request.
    imputer = pipeline.named_steps["pre"].named_transformers_["num"]
    nullable = [FEATURES[i] for i in imputer.indicator_.features_]

    model_version = f"hw4-{hw4_config.MODEL}-fit-{fit_date:%Y-%m-%d}"
    artifacts = HERE / "artifacts"
    artifacts.mkdir(exist_ok=True)
    bundle = {
        "pipeline": pipeline,
        "model_version": model_version,
        "features": list(FEATURES),
        "nullable_features": nullable,
    }
    joblib.dump(bundle, artifacts / "model.joblib")

    # Reference row: a real stock on the last signal date, scored by HW4's predict().
    last_date = panel.date.max()
    row = panel[(panel.date == last_date) & (panel.symbol == EXAMPLE_SYMBOL)]
    assert len(row) == 1
    expected = float(hw4_model.predict(pipeline, row, hw4_config.MODEL)[0])
    values = {
        name: None if pd.isna(row[name].item()) else float(row[name].item())
        for name in FEATURES
    }
    request = {**values, "sector": row.sector.item()}
    (HERE / "example_request.json").write_text(json.dumps(request, indent=2) + "\n")
    reference = {
        "source": (
            "HW4 model.predict() on the HW4 panel row for "
            f"{EXAMPLE_SYMBOL} at signal date {last_date:%Y-%m-%d}, computed by "
            "export_model.py before any API code is involved"
        ),
        "input": request,
        "expected_prediction": expected,
    }
    (HERE / "tests" / "reference_prediction.json").write_text(
        json.dumps(reference, indent=2) + "\n"
    )

    scores = hw4_model.predict(pipeline, panel[panel.date == last_date], hw4_config.MODEL)
    metadata = {
        "model_version": model_version,
        "hw4_model": "Ridge(alpha=10) on same-date percentile-rank features + sector",
        "hw4_source": "Homeworks/HW4/Oscar_Sallusti (model.fit, features.build_panel)",
        "fit_date": f"{fit_date:%Y-%m-%d}",
        "training_rows": int(train_rows),
        "training_window": "signal dates from 2010-01-01 with 90-day exit before the fit date",
        "target": "percentile rank within the signal date of the 90-day forward return",
        "features": list(FEATURES),
        "nullable_features": nullable,
        "sectors": list(SECTORS),
        "prediction_range_last_signal_date": [float(np.min(scores)), float(np.max(scores))],
        "python_version": platform.python_version(),
        "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "joblib_version": joblib.__version__,
        "artifact_sha256": hashlib.sha256(
            (artifacts / "model.joblib").read_bytes()
        ).hexdigest(),
    }
    (artifacts / "model_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))
    print("example:", request, "->", expected)


if __name__ == "__main__":
    main()
