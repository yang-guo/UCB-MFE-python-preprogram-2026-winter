"""Reproduce one HW4 fit, check saved forecasts, and export its fitted pipeline.

Run with the repository's HW4 Python environment. This is an offline build
step. The future API will load the result without this script or database.
"""

import copy
import hashlib
import json
import sys
from importlib.metadata import version
from pathlib import Path

import duckdb
import joblib
import numpy as np
import pandas as pd

from model_components import QuantileClipper

HERE = Path(__file__).resolve().parent
HW4 = HERE.parents[1] / "HW4" / "Mauricio_Gutierrez"
SOURCE = HW4 / "outputs" / "final_test"
MODEL_VERSION = "hw4-ridge-alpha100-2016-04-01"
ATOL = 1e-10
RTOL = 1e-8


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main():
    # Use the original HW4 functions so the training rules stay the same.
    sys.path.insert(0, str(HW4))
    from feature_engineering import clip_training_target, make_ridge_pipeline

    settings = json.loads((HW4 / "results" / "settings.json").read_text(encoding="utf-8"))
    features = settings["features"]
    first_signal = "2016-04-01"
    last_signal = "2016-12-31"
    columns = ", ".join(["date", "symbol", "target", "label_end"] + features)

    print("1. Read the existing HW4 feature cache (read-only).", flush=True)
    with duckdb.connect(str(SOURCE / "filtered_features.duckdb"), read_only=True) as con:
        train = con.execute(
            f"SELECT {columns} FROM D WHERE date >= ? AND date < ? "
            "AND label_end < ? AND isfinite(target) ORDER BY date, symbol",
            [settings["train_start"], first_signal, first_signal],
        ).fetchdf()
        signals = con.execute(
            f"SELECT {columns} FROM D WHERE date BETWEEN ? AND ? ORDER BY date, symbol",
            [first_signal, last_signal],
        ).fetchdf()
        saved = con.execute(
            "SELECT date, symbol, prediction FROM read_parquet(?) "
            "WHERE fit_date = ? ORDER BY date, symbol",
            [str(SOURCE / "forecasts.parquet"), first_signal],
        ).fetchdf()

    split = pd.read_csv(SOURCE / "training_splits.csv").set_index("fit_date").loc[first_signal]
    assert len(train) == int(split.training_rows), "Training row count differs from HW4."
    assert train.label_end.max() < pd.Timestamp(first_signal)
    assert signals.date.min() == pd.Timestamp(first_signal)
    target, clip = clip_training_target(train.target)
    np.testing.assert_allclose(
        [clip["target_clip_lower"], clip["target_clip_upper"]],
        [split.target_clip_lower, split.target_clip_upper],
        rtol=RTOL,
        atol=ATOL,
    )

    print(f"2. Fit the original HW4 pipeline on {len(train):,} rows.", flush=True)
    pipeline = make_ridge_pipeline(alpha=settings["alpha"], feature_names=features)
    pipeline.fit(train[features], target)
    training_rows = len(train)
    last_training_outcome = str(train.label_end.max().date())
    del train, target

    print("3. Compare predictions with the existing HW4 results.", flush=True)
    original_predictions = pipeline.predict(signals[features])
    comparison = signals[["date", "symbol"]].copy()
    comparison["reproduced"] = original_predictions
    comparison = comparison.merge(saved, on=["date", "symbol"], validate="one_to_one")
    assert len(comparison) == len(signals) == len(saved), "Forecast rows differ from HW4."
    np.testing.assert_allclose(
        comparison.reproduced, comparison.prediction, rtol=RTOL, atol=ATOL
    )

    # Keep the fitted values. Only move the custom class to an importable,
    # small HW5 module so loading does not require the HW4 project or DuckDB.
    numeric = pipeline.named_steps["preprocess"].named_transformers_["numeric"]
    fitted_clipper = numeric.named_steps["clip"]
    portable_clipper = QuantileClipper(**fitted_clipper.get_params())
    portable_clipper.__dict__.update(copy.deepcopy(fitted_clipper.__dict__))
    numeric.set_params(clip=portable_clipper)
    # ColumnTransformer also retains its unfitted constructor configuration.
    for _, transformer, _ in pipeline.named_steps["preprocess"].transformers:
        transformer.set_params(clip=QuantileClipper())
    np.testing.assert_allclose(
        pipeline.predict(signals[features]), original_predictions, rtol=RTOL, atol=ATOL
    )

    print("4. Save the fitted pipeline and a reference request.", flush=True)
    complete = np.isfinite(signals[features].to_numpy(dtype=float)).all(axis=1)
    row = signals.loc[complete].iloc[0]
    reference = saved.loc[(saved.date == row.date) & (saved.symbol == row.symbol)].iloc[0]
    example = {name: float(row[name]) for name in features}
    artifacts = HERE / "artifacts"
    artifacts.mkdir(exist_ok=True)
    model_path = artifacts / "model.joblib"
    joblib.dump(pipeline, model_path, compress=3)
    restored = joblib.load(model_path)
    restored_predictions = restored.predict(signals[features])
    np.testing.assert_allclose(restored_predictions, original_predictions, rtol=RTOL, atol=ATOL)
    write_json(HERE / "example_request.json", example)
    write_json(
        artifacts / "reference_prediction.json",
        {
            "source": "HW4 outputs/final_test/forecasts.parquet, saved before the HW5 API",
            "symbol": str(row.symbol),
            "signal_date": str(row.date.date()),
            "fit_date": first_signal,
            "prediction": float(reference.prediction),
            "model_version": MODEL_VERSION,
            "absolute_tolerance": ATOL,
            "relative_tolerance": RTOL,
        },
    )
    metadata = {
        "model_version": MODEL_VERSION,
        "fit_date": first_signal,
        "training_rows": training_rows,
        "last_training_outcome": last_training_outcome,
        "features": features,
        "alpha": settings["alpha"],
        "target": "Adjusted-price return from next-session entry to exit 90 calendar days later",
        "target_units": "decimal return: 0.05 means 5 percent",
        "preprocessing": [
            "quantile clipping",
            "median imputation and missing indicators",
            "scaling",
        ],
        "target_clipping": clip,
        "verified_forecasts": len(saved),
        "max_absolute_error_vs_hw4": float(
            np.abs(comparison.reproduced - comparison.prediction).max()
        ),
        "max_absolute_error_after_reload": float(
            np.abs(restored_predictions - original_predictions).max()
        ),
        "python_version": sys.version.split()[0],
        "package_versions": {
            name: version(name)
            for name in [
                "numpy",
                "pandas",
                "scikit-learn",
                "scipy",
                "joblib",
                "threadpoolctl",
                "duckdb",
            ]
        },
        "hw4_source_sha256": hashlib.sha256(
            (HW4 / "feature_engineering.py").read_bytes()
        ).hexdigest(),
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "artifact_bytes": model_path.stat().st_size,
    }
    write_json(artifacts / "model_metadata.json", metadata)
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
