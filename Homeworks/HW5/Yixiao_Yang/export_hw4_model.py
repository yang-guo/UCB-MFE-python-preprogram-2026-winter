"""Recreate and save the first quarterly fit from my completed HW4.

This is an offline export command, not part of API startup. It needs the
original HW4 folder and official bCourses database. Normal service use needs
only the small artifact already supplied with this assignment.
"""

from __future__ import annotations

import argparse
import ast
import gzip
import hashlib
import importlib
import importlib.metadata
import io
import json
import platform
import sys
import time
import zipfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from model_components import FEATURE_COLUMNS, MODEL_VERSION, TrainingQuantileClipper

HERE = Path(__file__).resolve().parent
HW4_COMMIT = "80c17f2fd6217926c18f7548d5d03dc4cfbf2c53"
HW4_URL = (
    "https://github.com/yang-guo/UCB-MFE-python-preprogram-2026-winter/blob/"
    f"{HW4_COMMIT}/Homeworks/HW4/Yixiao_Yang"
)
EXPECTED_DATABASE_SHA256 = "a028641d96c12e6f29d5dd34e3a6ebc059074dd38351648bf0d970b1dd4095b9"
EXPECTED_FROZEN_SHA256 = "fdb6fc0c8a176b72b00d8125e4350f98a32c1afd77f53b96929c96f70295ac82"
SOURCE_SHA256 = {
    "hw4.py": "290feeccf6a8dd2f31b8acd7d630e18c0952685ec7a6b278e79c423c7eb60f31",
    "data.py": "981712f7c771c6cd395734a634879d959c36c6cdb59690eaf01b55d4bcf3f3cb",
    "backtester.py": "4b23b947c217e7014c7e4cffcf6e07ac692e1674383d220841ddacb4ef852dcc",
}
ATOL = 1e-10
RTOL = 1e-10


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_supporting(hw4_dir: Path, name: str) -> bytes:
    """Read the recorded snapshot, not a later regenerated output directory."""
    with zipfile.ZipFile(hw4_dir / "supporting_evidence.zip") as archive:
        raw = archive.read(name)
        manifest = json.loads(archive.read("MANIFEST.json"))
    expected = manifest["files"][name]
    if len(raw) != expected["size_bytes"]:
        raise ValueError(f"Recorded evidence size changed: {name}")
    if hashlib.sha256(raw).hexdigest() != expected["sha256"]:
        raise ValueError(f"Recorded evidence checksum changed: {name}")
    return raw


def clipper_definition(path: Path) -> str:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "TrainingQuantileClipper"
    )
    return ast.dump(node, include_attributes=False)


def export(hw4_dir: Path, database: Path, output: Path, panel_cache: Path | None) -> dict:
    started = time.perf_counter()
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError("Use Python 3.12, as required for HW5.")
    for name, expected in SOURCE_SHA256.items():
        if file_sha256(hw4_dir / name) != expected:
            raise ValueError(f"Use the unchanged HW4 source at {HW4_COMMIT}: {name}")
    if file_sha256(database) != EXPECTED_DATABASE_SHA256:
        raise ValueError("The database is not the official HW4 file used in this experiment.")
    if clipper_definition(hw4_dir / "hw4.py") != clipper_definition(
        HERE / "model_components.py"
    ):
        raise ValueError("The portable clipper differs from the original HW4 class.")

    config = json.loads(read_supporting(hw4_dir, "outputs/run_config.json"))
    frozen_info = json.loads(read_supporting(hw4_dir, "verification/FROZEN_STRATEGY.json"))
    raw_forecasts = gzip.decompress(
        read_supporting(hw4_dir, "outputs/frozen_predictions.csv.gz")
    )
    if (
        hashlib.sha256(raw_forecasts).hexdigest() != EXPECTED_FROZEN_SHA256
        or frozen_info["predictions_sha256"] != EXPECTED_FROZEN_SHA256
    ):
        raise ValueError("The original frozen predictions failed verification.")
    if config["features"] != FEATURE_COLUMNS or config["selected_alpha"] != 1000.0:
        raise ValueError("Unexpected feature contract or selected alpha in HW4.")
    for name in ("numpy", "pandas", "scikit-learn", "duckdb"):
        if importlib.metadata.version(name) != config["packages"][name]:
            raise RuntimeError(f"Install {name}=={config['packages'][name]} before exporting.")

    # These imports happen only in this offline command, after source checks.
    sys.path.insert(0, str(hw4_dir))
    hw4 = importlib.import_module("hw4")
    hw4_data = importlib.import_module("data")
    if list(hw4_data.FEATURE_COLUMNS) != FEATURE_COLUMNS:
        raise ValueError("Feature order changed in the source code.")

    print("Preparing the authentic HW4 feature panel...", flush=True)
    prepared = hw4_data.prepare_data(database, batch_size=128)
    panel = prepared.panel
    if panel_cache is not None:
        panel_cache.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(panel, panel_cache, compress=3)
    del prepared

    quarter = "2016Q2"
    group = panel.loc[
        panel.eligible.fillna(False) & panel.entry_date.dt.to_period("Q").eq(pd.Period(quarter))
    ].sort_values(["entry_date", "symbol"])
    asof = group.signal_date.min()
    train = hw4.mature_training_rows(panel, asof)
    fit_audit = pd.read_csv(io.BytesIO(read_supporting(hw4_dir, "outputs/fit_audit.csv")))
    expected_fit = fit_audit.loc[fit_audit.entry_quarter.eq(quarter)].iloc[0]
    if (
        str(asof.date()) != expected_fit.fit_date
        or len(train) != int(expected_fit.training_rows)
        or len(group) != int(expected_fit.scored_rows)
        or not (train.label_end < asof).all()
    ):
        raise ValueError("The regenerated sample differs from the HW4 fit audit.")

    print(f"Fitting {quarter}: {len(train):,} rows, alpha=1000...", flush=True)
    pipeline = hw4.make_model(alpha=1000.0)
    pipeline.fit(hw4.feature_matrix(train), train.target)
    X_quarter = hw4.feature_matrix(group)
    before = pipeline.predict(X_quarter)
    frozen = pd.read_csv(
        io.BytesIO(raw_forecasts),
        parse_dates=["signal_date", "entry_date", "fit_date"],
        float_precision="round_trip",
    )
    original_quarter = frozen.loc[frozen.fit_date.eq(asof)]
    compared = group[["signal_date", "entry_date", "symbol"]].copy()
    compared["export_prediction"] = before
    compared = compared.merge(
        original_quarter,
        on=["signal_date", "entry_date", "symbol"],
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    if len(compared) != len(group) or not compared["_merge"].eq("both").all():
        raise ValueError("Quarterly keys differ from the saved HW4 forecasts.")
    np.testing.assert_allclose(compared.export_prediction, compared.score, atol=ATOL, rtol=RTOL)
    max_error = float(np.max(np.abs(compared.export_prediction - compared.score)))

    # The fitted numbers stay unchanged. Only the import location of the
    # identical custom class changes, so API loading does not import HW4.
    clipper = pipeline.named_steps["clip"]
    original_state = joblib.hash(clipper.__dict__)
    clipper.__class__ = TrainingQuantileClipper
    if joblib.hash(clipper.__dict__) != original_state:
        raise ValueError("The fitted clipper state changed during relocation.")
    np.testing.assert_array_equal(pipeline.predict(X_quarter), before)

    # This reference value comes from the prior HW4 CSV, never from the API
    # or the model reconstructed above.
    example = group.loc[
        group.symbol.eq("AAPL") & group.signal_date.eq(pd.Timestamp("2016-03-31"))
    ].iloc[0]
    example_input = {
        name: None if pd.isna(example[name]) else float(example[name])
        for name in FEATURE_COLUMNS
    }
    if example_input["filing_age_days"] is not None:
        example_input["filing_age_days"] = int(example_input["filing_age_days"])
    reference_row = original_quarter.loc[
        original_quarter.symbol.eq(example.symbol)
        & original_quarter.signal_date.eq(example.signal_date)
        & original_quarter.entry_date.eq(example.entry_date)
    ].iloc[0]
    serialization_packages = {
        name: importlib.metadata.version(name)
        for name in ("numpy", "pandas", "scikit-learn", "scipy", "joblib")
    }
    metadata = {
        "model_version": MODEL_VERSION,
        "features": FEATURE_COLUMNS,
        "model": "Ridge",
        "alpha": 1000.0,
        "fit_date": str(asof.date()),
        "entry_quarter": quarter,
        "training_rows": len(train),
        "scored_rows_verified": len(group),
        "first_training_signal": str(train.signal_date.min().date()),
        "last_training_signal": str(train.signal_date.max().date()),
        "last_training_label_end": str(train.label_end.max().date()),
        "target": (
            "Gross adjusted-price return fraction from the next market session's "
            "entry close to the first market session at or after entry + 90 calendar days."
        ),
        "preprocessing": [
            "Training-only 1st/99th percentile clipping",
            "Training median imputation with learned missing indicators; keep empty features",
            "Training StandardScaler",
        ],
        "missing_indicator_features": [
            FEATURE_COLUMNS[index]
            for index in pipeline.named_steps["impute"].indicator_.features_
        ],
        "hw4_source_commit": HW4_COMMIT,
        "hw4_source_url": HW4_URL,
        "hw4_source_sha256": SOURCE_SHA256,
        "model_components_sha256": file_sha256(HERE / "model_components.py"),
        "database_sha256": EXPECTED_DATABASE_SHA256,
        "frozen_predictions_sha256": EXPECTED_FROZEN_SHA256,
        "hw4_training_python": config["python"],
        "hw4_training_packages": config["packages"],
        "export_python": platform.python_version(),
        "serialization_packages": serialization_packages,
        "absolute_tolerance": ATOL,
        "relative_tolerance": RTOL,
        "maximum_absolute_difference_from_hw4": max_error,
        "export_method": (
            "Recreated the original 2016Q2 HW4 fit using its unchanged source, "
            "official database, selected alpha and training cutoff; checked every "
            "quarterly forecast against the previously saved HW4 CSV."
        ),
    }
    bundle = {
        "pipeline": pipeline,
        "model_version": MODEL_VERSION,
        "features": FEATURE_COLUMNS,
        "metadata": metadata,
    }
    artifact = output / "artifacts/model.joblib"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, artifact, compress=3)
    reloaded = joblib.load(artifact)
    np.testing.assert_array_equal(reloaded["pipeline"].predict(X_quarter), before)
    artifact_hash = file_sha256(artifact)
    write_json(
        output / "artifacts/model_metadata.json",
        {
            **metadata,
            "artifact_sha256": artifact_hash,
            "artifact_bytes": artifact.stat().st_size,
        },
    )
    write_json(output / "example_request.json", example_input)
    reference = {
        "input": example_input,
        "prediction": float(reference_row.score),
        "model_version": MODEL_VERSION,
        "absolute_tolerance": ATOL,
        "relative_tolerance": RTOL,
        "provenance": {
            "source": "Prior HW4 supporting_evidence.zip: outputs/frozen_predictions.csv.gz",
            "source_commit": HW4_COMMIT,
            "source_url": HW4_URL + "/supporting_evidence.zip",
            "uncompressed_forecasts_sha256": EXPECTED_FROZEN_SHA256,
            "symbol": str(example.symbol),
            "signal_date": str(example.signal_date.date()),
            "entry_date": str(example.entry_date.date()),
            "fit_date": str(reference_row.fit_date.date()),
            "feature_source": "Original data.prepare_data on the verified official HW4 database",
            "expected_value_source": "Existing frozen HW4 CSV, recorded before HW5 wrapping",
        },
    }
    write_json(output / "reference_prediction.json", reference)
    report = {
        "all_checks_passed": True,
        "model_version": MODEL_VERSION,
        "quarter": quarter,
        "fit_date": str(asof.date()),
        "training_rows": len(train),
        "frozen_forecasts_compared": len(compared),
        "max_absolute_forecast_difference": max_error,
        "absolute_tolerance": ATOL,
        "relative_tolerance": RTOL,
        "original_clipper_ast_matches": True,
        "clipper_fitted_state_unchanged": True,
        "relocated_class_predictions_bitwise_equal": True,
        "serialized_reload_predictions_bitwise_equal": True,
        "artifact_sha256": artifact_hash,
        "artifact_bytes": artifact.stat().st_size,
        "hw4_source_commit": HW4_COMMIT,
        "hw4_source_sha256": SOURCE_SHA256,
        "database_sha256": EXPECTED_DATABASE_SHA256,
        "frozen_predictions_sha256": EXPECTED_FROZEN_SHA256,
        "golden_reference_symbol": str(example.symbol),
        "golden_reference_prediction": float(reference_row.score),
        "reference_taken_from_prior_hw4": True,
        "runtime_seconds": round(time.perf_counter() - started, 3),
    }
    write_json(output / "evidence/model_export.json", report)
    print(json.dumps(report, indent=2), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hw4-dir", required=True, type=Path)
    parser.add_argument("--database", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=HERE)
    parser.add_argument(
        "--panel-cache",
        type=Path,
        help="Optional trusted local checkpoint, outside the submission",
    )
    args = parser.parse_args()
    export(
        args.hw4_dir.resolve(),
        args.database.resolve(),
        args.output.resolve(),
        args.panel_cache.resolve() if args.panel_cache else None,
    )


if __name__ == "__main__":
    main()
