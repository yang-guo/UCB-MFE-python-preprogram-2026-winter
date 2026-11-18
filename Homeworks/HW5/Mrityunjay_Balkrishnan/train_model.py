"""Recreate and export the HW4 walk-forward pipeline for the final fit date.

Run after HW4 source and the full course database are available in the repo.
The online service never calls this script, retrains, or opens the database.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone

ROOT = Path(__file__).resolve().parents[3]
HW4_FILE = ROOT / "Homeworks/HW4/Mrityunjay_Balkrishnan/hw4.py"
MODELING = ROOT / "Lectures/modeling"
DATABASE = MODELING / "data/equities_top2000.duckdb"
ARTIFACTS = Path(__file__).resolve().parent / "artifacts"
FIT_DATE = pd.Timestamp("2026-06-30")
MODEL_VERSION = "hw4-ridge-alpha1000-fit-2026-06-30"


def main() -> None:
    if not HW4_FILE.is_file():
        raise FileNotFoundError(f"Complete/merge HW4 before exporting: {HW4_FILE}")
    if not DATABASE.is_file():
        raise FileNotFoundError(f"Download the full HW4 database to: {DATABASE}")

    sys.path.insert(0, str(MODELING))
    spec = importlib.util.spec_from_file_location("hw4_submission", HW4_FILE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load HW4 implementation: {HW4_FILE}")
    hw4 = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = hw4
    spec.loader.exec_module(hw4)

    import course_data

    calendar, _benchmark, symbols, membership, _universe = hw4._read_reference_tables(DATABASE)
    excluded = {"BWNB", "SOFIW"}
    symbols = [symbol for symbol in symbols if symbol not in excluded]
    membership = membership.loc[~membership.symbol.isin(excluded)].copy()
    panel, _quality, _breaks = hw4.build_signal_panel(DATABASE, calendar, symbols, membership)

    validation_model, development, _validation_results = hw4.select_model(
        panel, course_data.FEATURES
    )
    alpha = float(validation_model.best_params_["model__alpha"])
    training = panel.loc[
        panel.date.ge(hw4.TRAIN_START)
        & panel.date.lt(FIT_DATE)
        & panel.label_end.lt(FIT_DATE)
        & panel.target.notna()
        & panel.eligible
    ].copy()
    if training.empty:
        raise ValueError("No mature HW4 labels exist for the final scheduled fit")

    pipeline = clone(validation_model.best_estimator_)
    pipeline.fit(training[course_data.FEATURES], training.target)
    scoring_rows = panel.loc[panel.date.eq(FIT_DATE) & panel.eligible].copy()
    if scoring_rows.empty:
        raise ValueError(f"No eligible HW4 signal rows exist for {FIT_DATE.date()}")
    row = (
        scoring_rows.assign(non_null=scoring_rows[course_data.FEATURES].notna().sum(axis=1))
        .sort_values(["non_null", "symbol"], ascending=[False, True])
        .iloc[0]
    )
    sample = {
        name: (None if pd.isna(row[name]) else float(row[name]))
        if name != "sector"
        else str(row[name])
        for name in course_data.FEATURES
    }
    sample_frame = pd.DataFrame([sample], columns=course_data.FEATURES)
    expected = float(np.asarray(pipeline.predict(sample_frame)).reshape(-1)[0])

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, ARTIFACTS / "hw4_ridge_pipeline.joblib", compress=3)
    (Path(__file__).resolve().parent / "example_request.json").write_text(
        json.dumps(sample, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    (ARTIFACTS / "reference_prediction.json").write_text(
        json.dumps(
            {
                "model_version": MODEL_VERSION,
                "source": "Direct pipeline prediction before the API wrapper",
                "signal_date": FIT_DATE.strftime("%Y-%m-%d"),
                "symbol_for_provenance": str(row.symbol),
                "prediction": expected,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    metadata = {
        "model_version": MODEL_VERSION,
        "fit_date": FIT_DATE.strftime("%Y-%m-%d"),
        "selected_ridge_alpha": alpha,
        "training_rows": int(len(training)),
        "training_start": training.date.min().strftime("%Y-%m-%d"),
        "latest_training_signal": training.date.max().strftime("%Y-%m-%d"),
        "latest_mature_label_exit": training.label_end.max().strftime("%Y-%m-%d"),
        "validation_years": [2012, 2013, 2014, 2015],
        "features_in_order": list(course_data.FEATURES),
        "sample_symbol_provenance": str(row.symbol),
        "sample_signal_date": FIT_DATE.strftime("%Y-%m-%d"),
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "scikit_learn_version": __import__("sklearn").__version__,
        "training_source": str(HW4_FILE.relative_to(ROOT)),
        "database_name": DATABASE.name,
        "database_sha256": hw4._file_sha256(DATABASE),
    }
    (ARTIFACTS / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))
    print(f"Direct reference prediction: {expected:.12g}")
    print(f"Development rows considered: {len(development):,}")


if __name__ == "__main__":
    main()
