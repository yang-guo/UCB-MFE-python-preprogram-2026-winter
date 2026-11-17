"""Reproduce the locked HW4 fit once, verify historical scores, and export it.

Research-only command; never imported or invoked by the HTTP service.
Run using the repository's HW4 Python environment from any directory.
"""

import hashlib
import json
import platform
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from preprocessing import QuantileClipper
from research.strategy.data import ROOT, build_dataset, course
from research.strategy.model import fit

HERE = Path(__file__).resolve().parent
MODEL_VERSION = "hw4-ridge-alpha10-fit20160401-v1"


def main():
    config = json.loads((HERE / "research/results/locked_config.json").read_text())
    original = json.loads((HERE / "research/results/run_metadata.json").read_text())
    with (ROOT / config["database"]).open("rb") as database:
        database_sha256 = hashlib.file_digest(database, "sha256").hexdigest()
    if database_sha256 != original["database_sha256"]:
        raise ValueError("Database differs from the original HW4 experiment")
    # Only the import path changes; the transformer implementation is unchanged.
    course.QuantileClipper = QuantileClipper
    panel, *_ = build_dataset(config)
    model, audit = fit(
        panel, pd.Timestamp("2016-04-01"), config["selected_alpha"], config["features"]
    )
    recorded = pd.read_csv(
        ROOT / "Homeworks/HW4/Prabhutosh_Kumar/HW4/outputs/results/predictions.csv"
    )
    # Historical predictions were saved by HW4 before any API implementation.
    actual = model.predict(recorded[config["features"]])
    np.testing.assert_allclose(actual, recorded.prediction, rtol=1e-9, atol=1e-10)
    reference_row = recorded.dropna(subset=config["features"]).iloc[0]
    request = {"features": [float(reference_row[name]) for name in config["features"]]}
    (HERE / "example_request.json").write_text(json.dumps(request, indent=2) + "\n")
    artifact_dir = HERE / "artifacts"
    artifact_dir.mkdir(exist_ok=True)
    artifact_path = artifact_dir / "model.joblib"
    joblib.dump(model, artifact_path, compress=3)
    versions = {
        name: version(name)
        for name in ["numpy", "pandas", "scikit-learn", "scipy", "joblib", "threadpoolctl"]
    }
    metadata = {
        "model_version": MODEL_VERSION,
        "features": config["features"],
        "holding_days": 90,
        "fit_audit": audit,
        "hw4_commit": "7f958c5cecf7dd066eb63704dfae7bcc4369068b",
        "database_sha256": database_sha256,
        "python": platform.python_version(),
        "packages": versions,
        "artifact_sha256": hashlib.sha256(artifact_path.read_bytes()).hexdigest(),
        "historical_rows_verified": len(recorded),
        "max_abs_prediction_difference": float(np.max(np.abs(actual - recorded.prediction))),
        "reference": {
            "source": "HW4 outputs/results/predictions.csv (saved before API wrapping)",
            "date": str(reference_row.date),
            "symbol": str(reference_row.symbol),
            "request": request,
            "prediction": float(reference_row.prediction),
            "rtol": 1e-9,
            "atol": 1e-10,
        },
    }
    (artifact_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
