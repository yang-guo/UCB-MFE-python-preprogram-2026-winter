"""Reconstruct the frozen HW4 fit offline and verify predictions before export.

Run with the root HW4 environment. No HW4 files or results are modified.
Serving never imports this module, retrains, or accesses the database.
"""

import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from model_tools import FEATURES, MODEL_VERSION, ROOT, QuantileClipper

REPOSITORY = ROOT.parents[2]
HW4 = REPOSITORY / "Homeworks/HW4/Jing_Jian"


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main():
    sys.path.insert(0, str(HW4))
    import research

    specification = json.loads((HW4 / "results/frozen_strategy.json").read_text())
    expected_audit = json.loads((HW4 / "results/final_fit_audit.json").read_text())
    if digest(research.DATABASE) != specification["database_sha256"]:
        raise ValueError("Database differs from the frozen HW4 experiment")
    if specification["features"] != FEATURES:
        raise ValueError("Unexpected frozen feature definitions")
    amendment = json.loads((HW4 / "results/execution_amendment.json").read_text())
    if research.code_hashes() != amendment["amended_code_sha256"]:
        raise ValueError("HW4 source differs from the recorded execution amendment")

    # Includes the March 31 signal with the original final-fit label cutoff.
    panel, loader_warnings = research.prepare_monthly(
        research.DATABASE, "2016-04-01", "2016-03-30"
    )
    fit_before = pd.Timestamp(expected_audit["fit_before_signal"])
    eligible = research.screen_rows(panel, specification["screen"])
    train = panel.loc[
        eligible
        & panel.date.ge("2010-01-01")
        & panel.date.lt("2016-01-01")
        & panel.date.lt(fit_before)
        & panel.label_end.lt(fit_before)
        & panel.target.notna()
    ].copy()
    lower, upper = train.target.quantile([0.01, 0.99])
    audit = {
        "fit_before_signal": str(fit_before.date()),
        "training_rows": len(train),
        "latest_training_signal": str(train.date.max().date()),
        "latest_training_outcome": str(train.label_end.max().date()),
        "target_clip_lower": float(lower),
        "target_clip_upper": float(upper),
    }
    for key, value in audit.items():
        if isinstance(value, float):
            np.testing.assert_allclose(value, expected_audit[key], rtol=1e-12, atol=1e-12)
        elif value != expected_audit[key]:
            raise ValueError(f"Reconstructed fit differs on {key}: {value}")

    numeric = Pipeline(
        [
            ("clip", QuantileClipper()),
            (
                "impute",
                SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
            ),
            ("scale", StandardScaler()),
        ]
    )
    pipeline = Pipeline(
        [
            (
                "preprocess",
                ColumnTransformer([("numeric", numeric, FEATURES)], remainder="drop"),
            ),
            ("model", Ridge(alpha=specification["alpha"])),
        ]
    )
    pipeline.fit(train[FEATURES], train.target.clip(lower, upper))
    source_scores = HW4 / "outputs/full/final_scores.csv"
    scores = pd.read_csv(source_scores, parse_dates=["date"], float_precision="round_trip")
    old = scores.loc[scores.date.eq(fit_before)]
    decisions = panel.loc[eligible & panel.date.eq(fit_before)]
    comparison = old.merge(decisions, on=["date", "symbol"], validate="one_to_one")
    if len(comparison) != len(old) or not len(old):
        raise ValueError("Could not match all original first-signal predictions")
    actual = pipeline.predict(comparison[FEATURES])
    np.testing.assert_allclose(actual, comparison.prediction, rtol=1e-10, atol=1e-12)
    row = comparison.sort_values("symbol").iloc[0]
    example = {name: float(row[name]) for name in FEATURES}
    artifacts = ROOT / "artifacts"
    artifacts.mkdir(exist_ok=True)
    joblib.dump(
        {"pipeline": pipeline, "features": FEATURES, "model_version": MODEL_VERSION},
        artifacts / "model.joblib",
        compress=3,
    )
    write_json(ROOT / "example_request.json", example)
    write_json(
        artifacts / "reference_prediction.json",
        {
            "input": example,
            "prediction": float(row.prediction),
            "symbol": row.symbol,
            "signal_date": str(row.date.date()),
            "source": "HW4 outputs/full/final_scores.csv, recorded before the HW5 API existed",
            "source_sha256": digest(source_scores),
            "rtol": 1e-10,
            "atol": 1e-12,
        },
    )
    write_json(
        artifacts / "metadata.json",
        {
            "model_version": MODEL_VERSION,
            "features": FEATURES,
            "fit_audit": audit,
            "model_sha256": digest(artifacts / "model.joblib"),
            "database_sha256": specification["database_sha256"],
            "hw4_code_sha256": research.code_hashes(),
            "training_versions": {
                name: importlib.metadata.version(name)
                for name in [
                    "numpy",
                    "pandas",
                    "scikit-learn",
                    "scipy",
                    "joblib",
                    "threadpoolctl",
                ]
            },
            "reference_predictions_verified": len(comparison),
            "maximum_absolute_prediction_difference": float(
                np.max(np.abs(actual - comparison.prediction))
            ),
            "loader_warnings": loader_warnings,
            "horizon": "next-session adjusted-close entry to 90-calendar-day maturity, next session if needed",
        },
    )
    print(f"Exported {MODEL_VERSION}; verified {len(comparison)} original HW4 predictions.")


if __name__ == "__main__":
    main()
