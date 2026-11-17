"""Replay the original HW4 final test and save the pipeline it fits."""

import argparse
import hashlib
import importlib
import json
import math
import sys
from importlib.metadata import version
from pathlib import Path
from unittest.mock import patch

import joblib

from model_service import FEATURES, MODEL_PACKAGES, MODEL_VERSION

ROOT = Path(__file__).resolve().parent


def write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def export(hw4_dir, database):
    sys.path.insert(0, str(hw4_dir))
    hw4 = importlib.import_module("run")
    old_metrics = json.loads((hw4_dir / "results/test_metrics.json").read_text())
    locked = json.loads((hw4_dir / "results/development.json").read_text())
    assert tuple(hw4.FEATURES) == FEATURES and locked["selected"] == "ridge"

    prices, filings, members, benchmark = hw4.load(database, hw4.TEST_END)
    panel = hw4.build_panel(prices, filings, members)
    factory, original_predict = hw4.make_pipeline, hw4.predict
    fitted = []
    reference = {}

    def capture_pipeline(*steps):
        pipeline = factory(*steps)
        fitted.append(pipeline)
        return pipeline

    def capture_reference(train, rows, model):
        scores = original_predict(train, rows, model)
        first_day = rows.loc[rows.entry.eq(hw4.TEST_START)]
        sample = first_day.loc[first_day.symbol.eq("AAPL")].iloc[0]
        reference.update(
            symbol=sample.symbol,
            signal_date=str(sample.signal.date()),
            entry_date=str(sample.entry.date()),
            request={name: float(sample[name]) for name in FEATURES},
            expected_prediction=float(scores[rows.index.get_loc(sample.name)]),
            model_version=MODEL_VERSION,
        )
        return scores

    with (
        patch.object(hw4, "make_pipeline", capture_pipeline),
        patch.object(hw4, "predict", capture_reference),
    ):
        actual, _, _, _ = hw4.evaluate(
            prices, panel, benchmark, hw4.TEST_START, hw4.TEST_END, locked["selected"]
        )

    assert len(fitted) == 1
    for field in ("training_rows", "latest_training_exit", "scored_rows", "realized_labels"):
        assert actual[field] == old_metrics[field], field
    for portfolio in ("strategy", "benchmark"):
        for key, value in actual[portfolio].items():
            assert math.isclose(value, old_metrics[portfolio][key], rel_tol=1e-9), key
    assert math.isclose(
        actual["mean_weekly_rank_ic"], old_metrics["mean_weekly_rank_ic"], rel_tol=1e-9
    )

    artifact = ROOT / "artifacts/hw4_ridge.joblib"
    artifact.parent.mkdir(exist_ok=True)
    joblib.dump(fitted[0], artifact, compress=3)
    manifest = {
        "model_version": MODEL_VERSION,
        "features": list(FEATURES),
        "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        "packages": {name: version(name) for name in MODEL_PACKAGES},
        "hw4_source_sha256": hashlib.sha256((hw4_dir / "run.py").read_bytes()).hexdigest(),
        "training_rows": actual["training_rows"],
        "latest_training_exit": actual["latest_training_exit"],
    }
    write_json(artifact.with_suffix(".json"), manifest)
    write_json(ROOT / "reference_prediction.json", reference)
    write_json(ROOT / "example_request.json", reference["request"])
    print("HW4 final test replay matches the original training counts and portfolio metrics.")
    print(json.dumps({"reference": reference, "manifest": manifest}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hw4-dir", required=True, type=Path)
    parser.add_argument("--db", required=True, type=Path)
    options = parser.parse_args()
    export(options.hw4_dir.resolve(), options.db.resolve())
