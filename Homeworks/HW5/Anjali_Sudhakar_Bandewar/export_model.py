"""One-time export of the fitted HW4 model. NOT used by the API or the Docker image.

HW4 fitted its pipelines in memory and never saved them, so this script reproduces the HW4
2015-fold HistGradientBoosting fit with HW4's own code and saves it:

1. Build the 2015 fold with HW4 ``make_fold`` (training rows: features before 2015-01-02 and
   exits before it; every 5th session) and fit HW4 ``hgb_pipeline`` with the selected
   hyperparameters. Fitting is deterministic (random_state=0, no early stopping).
2. Check the reproduction: the 2015 holdout metrics must equal the values HW4 recorded in
   ``results/models/selected_by_year.csv``.
3. Fit the identical pipeline using the QuantileClipper copy in ``hw4_model`` (so the pickle
   does not depend on HW4's training script/duckdb) and require bit-identical predictions on
   all 496,704 holdout rows.
4. Save the artifact and one real input row with its prediction from the HW4-code pipeline,
   used as the reference result in the tests.

Run from the repository root, in the HW4 training environment (Python 3.12.9,
scikit-learn 1.9.1, numpy 2.5.4, pandas 3.1.0, duckdb 2.0.0) with the HW4 data present:

    python Homeworks/HW5/Anjali_Sudhakar_Bandewar/export_model.py
"""

import json
import platform
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline

HW4_DIR = Path("Homeworks/HW4/Anjali_Sudhakar_Bandewar")
HW5_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(HW4_DIR / "src"))
sys.path.insert(0, str(HW5_DIR))

import train_baselines as hw4  # noqa: E402
from model_dataset import FEATURES, HOLDOUT_YEAR, load_dataset, make_fold  # noqa: E402

import hw4_model  # noqa: E402

# Selected in HW4 on mean 2012-2014 rank IC (results/models/tuning_summary.csv).
HGB_PARAMS = {"learning_rate": 0.05, "max_leaf_nodes": 15, "max_iter": 100}
REFERENCE_SYMBOL = "AAPL"


def serving_pipeline(learning_rate, max_leaf_nodes, max_iter):
    """Same steps and settings as HW4 hgb_pipeline, with hw4_model.QuantileClipper."""
    return Pipeline(
        [
            ("clip", hw4_model.QuantileClipper()),
            (
                "model",
                HistGradientBoostingRegressor(
                    learning_rate=learning_rate,
                    max_leaf_nodes=max_leaf_nodes,
                    max_iter=max_iter,
                    min_samples_leaf=1000,
                    early_stopping=False,
                    random_state=hw4.RANDOM_STATE,
                ),
            ),
        ]
    )


def main():
    assert FEATURES == hw4_model.FEATURES
    data, sessions = load_dataset()
    fold = make_fold(data, sessions, HOLDOUT_YEAR)
    X_train, y_train = fold.train[FEATURES], fold.train["target"]
    X_valid = fold.valid[FEATURES]
    print(f"2015 fold: fit_date={fold.fit_date.date()}, train_rows={len(X_train):,}")

    # 1-2. Reproduce HW4 with HW4 code and compare with the recorded holdout metrics.
    hw4_pipe = hw4.hgb_pipeline(**HGB_PARAMS).fit(X_train, y_train)
    hw4_scores = hw4_pipe.predict(X_valid)
    metrics = hw4.evaluate(fold.valid, hw4_scores)
    recorded = pd.read_csv(HW4_DIR / "results/models/selected_by_year.csv")
    recorded = recorded[(recorded["model"] == "hgb") & (recorded["year"] == HOLDOUT_YEAR)]
    assert recorded["params"].item() == str(HGB_PARAMS)
    for key in ("ic_mean", "decile_spread", "mse_rank", "r2_rank"):
        print(f"  {key}: reproduced={metrics[key]!r} recorded={recorded[key].item()!r}")
        assert np.isclose(metrics[key], recorded[key].item(), rtol=1e-12, atol=0)

    # 3. Serving pipeline must be bit-identical to the HW4-code pipeline.
    serve_pipe = serving_pipeline(**HGB_PARAMS).fit(X_train, y_train)
    assert np.array_equal(serve_pipe.predict(X_valid), hw4_scores)
    print(f"  serving pipeline identical on {len(X_valid):,} holdout rows")

    # 4. Save the artifact and a real reference row (first 2015 session, complete features).
    hw4_model.MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(serve_pipe, hw4_model.ARTIFACT_PATH)

    valid = fold.valid.reset_index(drop=True)
    idx = valid.index[
        (valid["symbol"] == REFERENCE_SYMBOL) & (valid["feature_date"] == fold.fit_date)
    ].item()
    row = valid.loc[idx, FEATURES]
    assert row.notna().all()
    reference = {
        "source": "HW4 train_baselines.hgb_pipeline fitted on HW4 make_fold(2015) training rows",
        "model_version": hw4_model.MODEL_VERSION,
        "symbol": REFERENCE_SYMBOL,
        "feature_date": str(fold.fit_date.date()),
        "input": {
            name: int(row[name]) if name == "universe_rank" else float(row[name])
            for name in FEATURES
        },
        "prediction": float(hw4_scores[idx]),
        "fit_date": str(fold.fit_date.date()),
        "train_rows": len(X_train),
        "hyperparameters": HGB_PARAMS,
        "reproduced_holdout_metrics": {
            k: float(metrics[k]) for k in ("ic_mean", "decile_spread", "mse_rank", "r2_rank")
        },
        "training_environment": {
            "python": platform.python_version(),
            "scikit-learn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "joblib": joblib.__version__,
        },
    }
    hw4_model.REFERENCE_PATH.write_text(json.dumps(reference, indent=2) + "\n")
    print(f"Saved {hw4_model.ARTIFACT_PATH} and {hw4_model.REFERENCE_PATH}")
    print(json.dumps(reference["input"]), "->", reference["prediction"])


if __name__ == "__main__":
    main()
