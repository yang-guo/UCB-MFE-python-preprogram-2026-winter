"""Save the fitted HW4 model for the HW5 service. Run once, in the HW4 environment.

    python export_model.py --hw4-dir ../../HW4/Romir_Kant

This is the only file here that needs the HW4 code and data. It repeats the
last refit of HW4's final test (final_test.py, fit date 2026-03-27) by calling
the same HW4 function, research.fit_predict, on the same rows. It writes:

    artifacts/model.joblib                the fitted Ridge, its version, the feature order
    artifacts/model_metadata.json         the same facts in a readable form
    artifacts/reference_predictions.json  HW4 inputs and the predictions HW4 gave them
    example_request.json                  the first reference input, as a request body

The reference predictions come from research.fit_predict itself, before any
API code touches the model. The tests compare the service against them.
"""

import argparse
import json
import platform
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

HERE = Path(__file__).resolve().parent
ARTIFACTS = HERE / "artifacts"
EXPECTED_TRAIN_ROWS = 1_580_288  # HW4 results/test_fit_schedule.csv, fit date 2026-03-27


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hw4-dir", type=Path, default=HERE.parents[1] / "HW4" / "Romir_Kant")
    parser.add_argument(
        "--panel", type=Path, help="default: <hw4-dir>/outputs/panel_equities_top2000.parquet"
    )
    parser.add_argument("--hw4-commit", default="93c2441", help="HW4 commit the code came from")
    args = parser.parse_args()
    sys.path.insert(0, str(args.hw4_dir))
    import research  # the HW4 module

    panel = pd.read_parquet(
        args.panel or args.hw4_dir / "outputs" / "panel_equities_top2000.parquet"
    )
    X = research.rank_features(panel)

    # Same fit date and rows as the last refit in HW4's final_test.py / walk_forward.
    march = panel.loc[panel.date.dt.month.eq(3) & panel.date.ge("2016-03-01"), "date"]
    fit_date = max(march.groupby(march.dt.year).max())
    train = panel.date.ge("2010-01-01") & panel.label_end.lt(fit_date) & panel.target.notna()
    test = panel.date.ge(fit_date)
    assert int(train.sum()) == EXPECTED_TRAIN_ROWS, (
        f"train rows {train.sum()} differ from HW4's final test"
    )
    preds, model = research.fit_predict(
        panel.loc[train], panel.loc[test], X.loc[train], X.loc[test], "ridge", "rank"
    )
    preds = pd.Series(preds, index=panel.index[test])
    features = list(research.FEATURES)
    assert list(model.feature_names_in_) == features

    version = f"hw4-ridge-rank-fit-{fit_date:%Y-%m-%d}"
    ARTIFACTS.mkdir(exist_ok=True)
    joblib.dump(
        {"model": model, "model_version": version, "features": features},
        ARTIFACTS / "model.joblib",
    )
    metadata = {
        "model_version": version,
        "hw4_commit": args.hw4_commit,
        "model": "sklearn.linear_model.Ridge(alpha=1000.0)",
        "target": "percentile rank of the 90-day return among that week's stocks, minus 0.5",
        "horizon": "next session's close to the first close at least 90 calendar days later",
        "features": features,
        "feature_units": "percentile rank among that week's eligible stocks, minus 0.5; missing = 0",
        "fit_date": f"{fit_date:%Y-%m-%d}",
        "train_rows": int(train.sum()),
        "last_training_outcome": f"{panel.loc[train, 'label_end'].max():%Y-%m-%d}",
        "python_version": platform.python_version(),
        "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "joblib_version": joblib.__version__,
    }
    (ARTIFACTS / "model_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    # Reference cases: two ordinary stocks and one with missing features (sent as 0).
    used = X.loc[test].fillna(0.0)  # the exact inputs fit_predict gave the model
    rows = panel.loc[test]
    last_date = rows.date.max()
    missing = X.loc[test].isna().any(axis=1) & rows.date.eq(fit_date)
    picks = [
        rows.index[rows.symbol.eq("AAPL") & rows.date.eq(fit_date)][0],
        rows.index[rows.symbol.eq("MSFT") & rows.date.eq(last_date)][0],
        missing.index[missing][0],
    ]
    cases = [
        {
            "symbol": panel.symbol[i],
            "signal_date": f"{panel.date[i]:%Y-%m-%d}",
            "missing_in_hw4": [f for f in features if pd.isna(X.loc[i, f])],
            "features": {f: float(used.loc[i, f]) for f in features},
            "hw4_prediction": float(preds[i]),
        }
        for i in picks
    ]
    (ARTIFACTS / "reference_predictions.json").write_text(json.dumps(cases, indent=2) + "\n")
    (HERE / "example_request.json").write_text(
        json.dumps(cases[0]["features"], indent=2) + "\n"
    )

    print(json.dumps(metadata, indent=2))
    for case in cases:
        print(
            case["symbol"],
            case["signal_date"],
            case["hw4_prediction"],
            "missing:",
            case["missing_in_hw4"],
        )
    print(
        "coefficients:",
        dict(zip(features, model.coef_.round(4))),
        "intercept:",
        round(float(model.intercept_), 6),
    )


if __name__ == "__main__":
    main()
