"""Fit one HW4 ridge-rank pipeline and save it for the service.

Run from anywhere inside the repository, with the course environment that
matches ``requirements.txt``:

    python Homeworks/HW5/Suge_Li/train_artifact.py

The fit uses the Homework 4 month-end panel through 2016-04-15, the same
window as ``hw4.py --develop``. Rows are signals from 2010-01-01 through
2015-12-31 that pass the frozen screen (universe rank <= 400, unadjusted close
>= $5, dollar volume >= $1m) and whose 90-day labels end before 2016-04-01.
Extreme raw returns outside that sample's 1st–99th percentiles are dropped,
exactly as ``fit_ridge`` does. The frozen trading rule remains ``mom_roa``;
this file is the ``ridge_rank`` candidate from that experiment.

The service never runs this script and never opens the database.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from contract import FEATURE_NAMES, MODEL_VERSION
from preprocessing import make_ridge_pipeline

ROOT = Path(__file__).resolve().parent

PANEL_END = "2016-04-15"
SIGNAL_START = "2010-01-01"
SIGNAL_END = "2015-12-31"
LABEL_END_BEFORE = "2016-04-01"
RANK_CAP = 400


def find_repo() -> Path:
    for candidate in [ROOT, *ROOT.parents]:
        if (candidate / "Lectures" / "modeling" / "course_data.py").is_file():
            return candidate
    raise FileNotFoundError("could not find the course repository root")


def load_hw4_features():
    """Load ``features.py`` from the HW4 submission without checking it out."""
    repo = find_repo()
    os.chdir(repo)
    source = subprocess.check_output(
        ["git", "show", "hw_4_Suge_Li:Homeworks/HW4/Suge_Li/features.py"],
        cwd=repo,
    )
    module_path = Path(tempfile.mkdtemp(prefix="hw4-features-")) / "hw4_features.py"
    module_path.write_bytes(source)
    spec = importlib.util.spec_from_file_location("hw4_features", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load HW4 features.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def training_frame(panel: pd.DataFrame, features) -> pd.DataFrame:
    mask = (
        features.entry_screen(panel, RANK_CAP)
        & panel.date.ge(SIGNAL_START)
        & panel.date.le(SIGNAL_END)
        & panel.target.notna()
        & panel.rank_target.notna()
        & panel.label_end.lt(LABEL_END_BEFORE)
    )
    labeled = panel.loc[mask].reset_index(drop=True)
    if labeled.label_end.max() >= pd.Timestamp(LABEL_END_BEFORE):
        raise RuntimeError("a training label extends into the final test window")
    keep = features.in_fold_keep(labeled.target)
    return labeled.loc[keep].reset_index(drop=True)


def example_payload(fitted, kept: pd.DataFrame) -> tuple[dict[str, float], float, pd.Series]:
    complete = kept.loc[kept[FEATURE_NAMES].notna().all(axis=1)].sort_values(["date", "symbol"])
    if complete.empty:
        raise RuntimeError("training sample has no row with all 12 features")
    row = complete.iloc[len(complete) // 2]
    payload = {name: float(row[name]) for name in FEATURE_NAMES}
    payload = json.loads(json.dumps(payload))
    frame = pd.DataFrame([payload])[FEATURE_NAMES]
    prediction = float(fitted.predict(frame)[0])
    if not np.isfinite(prediction):
        raise RuntimeError("reference prediction is not finite")
    return payload, prediction, row


def learned_bounds(fitted) -> dict[str, dict[str, float]]:
    clipper = (
        fitted.named_steps["preprocess"].named_transformers_["numeric"].named_steps["clip"]
    )
    bounds = {}
    for name, lower, upper in zip(
        FEATURE_NAMES, clipper.lower_bounds_, clipper.upper_bounds_, strict=True
    ):
        bounds[name] = {"p01": float(lower), "p99": float(upper)}
    return bounds


def main() -> None:
    if sklearn.__version__ != "1.8.0":
        raise RuntimeError(
            f"refusing to save an artifact from scikit-learn {sklearn.__version__}"
        )
    repo = find_repo()
    database = repo / "Lectures" / "modeling" / "data" / "equities_top2000.duckdb"
    if not database.is_file():
        raise FileNotFoundError(database)
    features = load_hw4_features()
    print(f"Building the month-end panel through {PANEL_END}", flush=True)
    panel, _benchmark = features.batched_month_end_panel(database, end=PANEL_END)
    kept = training_frame(panel, features)
    print(f"Fitting Ridge on {len(kept):,} rows", flush=True)
    model = make_ridge_pipeline(FEATURE_NAMES)
    model.fit(kept[FEATURE_NAMES], kept["rank_target"])
    payload, prediction, row = example_payload(model, kept)
    artifact_dir = ROOT / "artifacts"
    artifact_dir.mkdir(exist_ok=True)
    bundle = {"pipeline": model, "model_version": MODEL_VERSION, "features": FEATURE_NAMES}
    joblib.dump(bundle, artifact_dir / "model.joblib")
    reference = {
        "model_version": MODEL_VERSION,
        "date": str(pd.Timestamp(row.date).date()),
        "symbol": str(row.symbol),
        "features": payload,
        "prediction": prediction,
        "source": (
            "Direct pipeline.predict on one finite training row after JSON "
            "round-trip, recorded before the API existed. The row is the "
            "middle date/symbol of the kept development sample."
        ),
    }
    (artifact_dir / "reference_prediction.json").write_text(
        json.dumps(reference, indent=2) + "\n"
    )
    (ROOT / "example_request.json").write_text(json.dumps(payload, indent=2) + "\n")
    metadata = {
        "model_version": MODEL_VERSION,
        "estimator": "Ridge(alpha=100)",
        "target": "same-date percentile rank of the 90-calendar-day forward return",
        "features": FEATURE_NAMES,
        "panel_end": PANEL_END,
        "signal_start": SIGNAL_START,
        "signal_end": SIGNAL_END,
        "label_end_before": LABEL_END_BEFORE,
        "rank_cap": RANK_CAP,
        "training_rows": int(len(kept)),
        "signal_min": str(pd.Timestamp(kept.date.min()).date()),
        "signal_max": str(pd.Timestamp(kept.date.max()).date()),
        "learned_clip_bounds": learned_bounds(model),
        "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "trading_rule_note": (
            "Homework 4 froze mom_roa for the backtest. This artifact is the "
            "ridge_rank candidate fit on the development labels."
        ),
    }
    (artifact_dir / "model_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved {MODEL_VERSION}", flush=True)
    print(
        f"Reference {reference['symbol']} {reference['date']} prediction={prediction:.8f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
