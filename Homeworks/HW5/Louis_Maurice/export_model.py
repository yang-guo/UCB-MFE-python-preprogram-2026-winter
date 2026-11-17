"""Rebuild and save the last fitted HW4 model (run once, in the HW4 environment).

This is NOT part of the service. It reuses the HW4 code unchanged
(Homeworks/HW4/Louis_Maurice/hw4.py) to reproduce the final quarterly refit of the
HW4 test run: the Ridge pipeline fitted at the 2026-06-30 signal date, which scored
the last quarter of the experiment (2026-06-30 to 2026-09-30).

From the repository root, after `uv sync --locked` (the HW4 environment):

    uv run --locked python Homeworks/HW5/Louis_Maurice/export_model.py

It needs equities_top2000.duckdb in Lectures/modeling/data/ (about 2 minutes).
It writes artifacts/model.joblib and artifacts/reference_example.json.
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "Homeworks/HW4/Louis_Maurice"))
# hw4 puts Lectures/modeling on sys.path, so the pipeline's QuantileClipper is pickled
# as course_data.QuantileClipper. The service ships an identical copy of course_data.py.
import hw4  # noqa: E402

from course_data import FEATURES, NUMERIC_FEATURES  # noqa: E402

FIT_DATE = pd.Timestamp("2026-06-30")  # last quarterly refit in hw4.quarterly_scores
MODEL_VERSION = "hw4-ridge-refit-2026-06-30"
REFERENCE_SYMBOL = "AAPL"


def main():
    panel, _, _ = hw4.build_panel()
    model = hw4.choose_alpha(panel)  # same development alpha search as HW4 (alpha=1000)

    # Same training rows and target clipping as hw4.quarterly_scores at FIT_DATE.
    train = panel.loc[
        panel.eligible
        & panel.date.ge(hw4.DEV_START)
        & panel.date.lt(FIT_DATE)
        & panel.label_end.lt(FIT_DATE)
    ]
    to_score = panel.loc[panel.eligible & panel.date.eq(FIT_DATE)].reset_index(drop=True)
    assert train.label_end.max() < to_score.date.min()
    fitted = clone(model).fit(train[FEATURES], hw4.clip_target(train.target, train.target))
    scores = fitted.predict(to_score[FEATURES])

    artifacts = HERE / "artifacts"
    artifacts.mkdir(exist_ok=True)
    bundle = {
        "pipeline": fitted,
        "model_version": MODEL_VERSION,
        "features": FEATURES,
        "sectors": sorted(train.sector.dropna().unique().tolist()),
        "alpha": float(fitted.named_steps["model"].alpha),
        "sklearn_version": sklearn.__version__,
        "train_rows": int(len(train)),
    }
    joblib.dump(bundle, artifacts / "model.joblib")

    # Reference row: a real 2026-06-30 signal row and the score HW4 code produced for it.
    row = to_score.loc[to_score.symbol.eq(REFERENCE_SYMBOL)].index[0]
    features = {
        name: (None if pd.isna(value) else float(value) if name in NUMERIC_FEATURES else value)
        for name, value in to_score.loc[row, FEATURES].items()
    }
    reference = {
        "source": "export_model.py: hw4 refit at 2026-06-30, predicted on that date's universe",
        "symbol": REFERENCE_SYMBOL,
        "signal_date": str(FIT_DATE.date()),
        "model_version": MODEL_VERSION,
        "features": features,
        "expected_prediction": float(scores[row]),
    }
    (artifacts / "reference_example.json").write_text(json.dumps(reference, indent=2) + "\n")
    (HERE / "example_request.json").write_text(json.dumps(features, indent=2) + "\n")

    # Ranges observed in training, used to justify the API bounds in app.py.
    print(train[NUMERIC_FEATURES].describe(percentiles=[0.001, 0.999]).T.to_string())
    print("missing share:\n", train[NUMERIC_FEATURES].isna().mean().round(3).to_string())
    print("sectors:", bundle["sectors"], "| alpha:", bundle["alpha"], "| rows:", len(train))
    print("reference:", reference)
    assert np.isfinite(scores).all()


if __name__ == "__main__":
    main()
