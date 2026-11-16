"""Reproduce HW4's final fit once, then save its pipeline and a reference result.

The HW4 branch saved results, but not the fitted estimator. This script uses
that exact commit's source and the full bCourses database. The API never calls it.
"""

import argparse
import json
import subprocess
import sys
import tempfile
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

HERE = Path(__file__).resolve().parent
HW4_COMMIT = "1c2a664f5743d17cbf233becfcc16d68b8123aa4"
MODEL_VERSION = "hw4-ridge-alpha1000-1c2a664-v1"
TRAINING_VERSIONS = {
    "numpy": "2.3.5",
    "pandas": "2.3.3",
    "scikit-learn": "1.8.0",
    "scipy": "1.16.3",
    "joblib": "1.5.3",
    "threadpoolctl": "3.6.0",
    "duckdb": "1.5.6",
}


def hw4_file(path):
    return subprocess.check_output(["git", "show", f"{HW4_COMMIT}:{path}"], cwd=HERE)


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    args = parser.parse_args()
    database = args.database.expanduser().resolve()
    if not database.is_file():
        parser.error(f"Database not found: {database}")
    for package, expected in TRAINING_VERSIONS.items():
        if version(package) != expected:
            raise RuntimeError(f"Install {package}=={expected} before exporting")

    # Import the original data and pipeline helpers, rather than rewriting HW4.
    with tempfile.TemporaryDirectory(prefix="hw5-source-") as directory:
        source = Path(directory)
        for name, path in {
            "hw4_data.py": "Homeworks/HW4/Weitao_Su/hw4_data.py",
            "course_data.py": "Lectures/modeling/course_data.py",
        }.items():
            (source / name).write_bytes(hw4_file(path))
        sys.path.insert(0, str(source))
        import hw4_data

        from course_data import date_folds, make_pipeline

        saved = json.loads(hw4_file("Homeworks/HW4/Weitao_Su/results/summary.json"))
        settings = saved["settings"]
        features = hw4_data.FEATURES
        development = hw4_data.prepare_panel(database, "2016-03-31", settings)
        tuning = development.loc[
            development.date.between("2010-01-01", "2014-12-31")
            & development.label_end.lt("2015-01-01")
            & development.target.notna()
        ].reset_index(drop=True)
        search = GridSearchCV(
            make_pipeline(Ridge(), numeric_features=features, include_sector=False),
            {"model__alpha": [1, 100, 1000]},
            cv=date_folds(tuning, validation_years=(2013, 2014)),
            scoring="neg_root_mean_squared_error",
            n_jobs=1,
        )
        search.fit(tuning[features], tuning.target)
        cv_rmse = -search.cv_results_["mean_test_score"]
        np.testing.assert_allclose(
            cv_rmse, [row["RMSE"] for row in saved["model_selection"]], rtol=1e-8, atol=1e-10
        )
        assert search.best_params_["model__alpha"] == settings["alpha"]

        # This is the same final training selection and clone/fit as run_project.py.
        train = development.loc[
            development.date.between("2010-01-01", "2015-12-31")
            & development.label_end.lt("2016-03-31")
            & development.target.notna()
        ]
        assert str(train.label_end.max().date()) == settings["last_training_outcome"]
        pipeline = clone(search.best_estimator_).fit(train[features], train.target)

        # Record a real HW4 row and the original pipeline's result before API use.
        candidates = development.loc[
            development.symbol.eq("AAPL") & development[features].notna().all(axis=1)
        ]
        example = candidates.tail(1)
        if example.empty:
            raise ValueError("No complete AAPL example in the HW4 panel")
        request = {name: float(example.iloc[0][name]) for name in features}
        expected_prediction = float(pipeline.predict(example[features])[0])

        artifacts = HERE / "artifacts"
        artifacts.mkdir(exist_ok=True)
        bundle = {
            "pipeline": pipeline,
            "model_version": MODEL_VERSION,
            "feature_names": features,
            "training_versions": TRAINING_VERSIONS,
        }
        joblib.dump(bundle, artifacts / "model.joblib", compress=3)
        save_json(HERE / "example_request.json", request)
        save_json(
            artifacts / "reference_prediction.json",
            {
                "source": "Original HW4 pipeline.predict(example[FEATURES]) during export",
                "hw4_commit": HW4_COMMIT,
                "symbol": str(example.iloc[0].symbol),
                "signal_date": str(example.iloc[0].date.date()),
                "entry_date": str(example.iloc[0].entry_date.date()),
                "input": request,
                "prediction": expected_prediction,
                "model_version": MODEL_VERSION,
            },
        )
        print(f"Exported {len(train)} training rows; CV RMSE: {cv_rmse.tolist()}")
        print(f"Reference: {request}")
        print(f"Prediction: {expected_prediction}; model_version: {MODEL_VERSION}")


if __name__ == "__main__":
    main()
