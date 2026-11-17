"""Export the frozen HW4 model and reference predictions for the HW5 service.

Run this ONCE with the HW4 (course) environment, not the HW5 service environment:
it imports the HW4 ``quantlab`` package and reads ``equities_top2000.duckdb``.

    # from the repository root, with the HW4 branch checked out in a worktree
    git worktree add ../hw4-src hw_4_Ian_Kwon
    .venv/Scripts/python Homeworks/HW5/Ian_Kwon/scripts/export_hw4_model.py \
        --hw4-dir ../hw4-src/Homeworks/HW4/Ian_Kwon \
        --database Lectures/modeling/data/equities_top2000.duckdb

What it does, using only HW4 code paths (nothing here re-implements the model):

1. Loads ``config/frozen_strategy.json`` and checks its SHA-256 against
   ``config/FROZEN.lock`` so the exported model is the submitted strategy.
2. Builds the HW4 panel in RESEARCH mode (data capped at 2016-03-31 by HW4's SQL
   firewall), so exporting the model never touches the final-test holdout.
3. Runs HW4's ``walk_forward_scores`` exactly as ``run_research`` does and keeps
   the estimator fitted at the last development refit.
4. Saves that estimator with joblib, plus metadata (feature order, versions,
   SHA-256) and a few (input, HW4 score) pairs that the HW5 tests replay.

The service never imports this script, the database, or the HW4 package.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

HW5_ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finite_or_none(value: float) -> float | None:
    """JSON has no NaN; HW4 used NaN for a missing rank, the API uses null."""
    return None if value != value else float(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--hw4-dir", type=Path, required=True, help="HW4 submission folder")
    parser.add_argument("--database", type=Path, default=None, help="equities_top2000.duckdb")
    parser.add_argument("--out-dir", type=Path, default=HW5_ROOT, help="HW5 submission folder")
    args = parser.parse_args()

    hw4 = args.hw4_dir.resolve()
    sys.path.insert(0, str(hw4))  # import the HW4 package, not the HW5 copy of models.py

    import joblib
    import numpy as np
    import pandas as pd
    import sklearn

    from quantlab import experiment, protocol
    from quantlab.panel import feature_columns
    from quantlab.protocol import Mode
    from quantlab.scoring import walk_forward_scores

    # 1) the submitted strategy, verified against its hash lock
    cfg = json.loads((hw4 / "config" / "frozen_strategy.json").read_text())
    lock = json.loads((hw4 / "config" / "FROZEN.lock").read_text())
    if protocol.digest(cfg) != lock["config_sha256"]:
        raise SystemExit("frozen_strategy.json does not match FROZEN.lock")
    experiment_id = lock["development_evidence"]["experiment_id"]

    # 2) research-mode panel: HW4 refuses to load prices after 2016-03-31 here
    database = args.database.resolve() if args.database else experiment.DEFAULT_DB
    market, _ref, _audit, panel = experiment.prepare(
        Mode.RESEARCH, database=database, panel_cfg=cfg["panel"]
    )
    columns = feature_columns(cfg["features"])  # e.g. "vol_63" -> "vol_63_csr"

    # 3) the same walk-forward call as HW4's run_research development backtest
    dev_start = pd.Timestamp(cfg["dev_backtest"]["start"])
    signal0 = market.calendar[market.calendar.searchsorted(dev_start) - 1]
    model_cfg = cfg["model"]
    scores, fit_audit = walk_forward_scores(
        panel,
        columns,
        model_cfg["kind"],
        model_cfg["params"],
        cfg["target"],
        signal0,
        protocol.DEVELOPMENT_END,
        cfg["train_start"],
        cfg["refit"],
        cfg["train_window_years"],
        Mode.RESEARCH,
    )
    last_fit = fit_audit.iloc[-1]
    model = last_fit["_model"]

    # 4a) the fitted estimator
    artifacts = args.out_dir / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    model_path = artifacts / "model.joblib"
    joblib.dump(model, model_path)

    # 4b) reference cases: rows scored by HW4 after the last refit, with their inputs
    scored = scores.loc[scores.fit_date.eq(last_fit["fit_date"])]
    rows = scored.merge(panel[["date", "symbol", *columns]], on=["date", "symbol"])
    last_day = rows.loc[rows.date.eq(rows.date.max())].sort_values("symbol")
    complete = last_day.dropna(subset=columns)
    picks = {
        "all features present (AAPL)": complete.loc[complete.symbol.eq("AAPL")].head(1),
        "a missing feature rank (null = median)": last_day.loc[
            last_day[columns].isna().any(axis=1)
        ].head(1),
        "highest HW4 score on the date": last_day.nlargest(1, "score"),
    }
    cases = []
    for label, row in picks.items():
        if row.empty:
            raise SystemExit(f"No row for reference case: {label}")
        row = row.iloc[0]
        x = row[columns].to_numpy(float).reshape(1, -1)
        recomputed = float(model.predict(x)[0])
        if not np.isclose(recomputed, row.score, rtol=0, atol=1e-12):
            raise SystemExit(f"Saved model does not reproduce the HW4 score for {label}")
        cases.append(
            {
                "case": label,
                "date": str(row.date.date()),
                "symbol": row.symbol,
                "features": {c: finite_or_none(row[c]) for c in columns},
                "hw4_score": float(row.score),
            }
        )

    reference = {
        "source": (
            "HW4 quantlab.scoring.walk_forward_scores, research mode, frozen config "
            f"{experiment_id}; scores of the last development refit "
            f"({last_fit['fit_date'].date()}) on signal date {cases[0]['date']}"
        ),
        "model_version": f"hw4-{experiment_id}",
        "cases": cases,
    }
    tests_data = args.out_dir / "tests" / "data"
    tests_data.mkdir(parents=True, exist_ok=True)
    (tests_data / "hw4_reference.json").write_text(json.dumps(reference, indent=2) + "\n")
    (args.out_dir / "example_request.json").write_text(
        json.dumps(cases[0]["features"], indent=2) + "\n"
    )

    # 4c) metadata the service checks before and after unpickling
    metadata = {
        "model_version": f"hw4-{experiment_id}",
        "hw4_experiment_id": experiment_id,
        "hw4_config_sha256": lock["config_sha256"],
        "model_class": f"{type(model).__module__}.{type(model).__qualname__}",
        "model_params": model_cfg["params"],
        "features": columns,
        "feature_registry_names": cfg["features"],
        "coef": [float(c) for c in model.coef_],
        "fit_date": str(last_fit["fit_date"].date()),
        "train_rows": int(last_fit["train_rows"]),
        "train_max_label_end": str(last_fit["train_max_label_end"].date()),
        "target": cfg["target"],
        "holding_calendar_days": cfg["panel"]["holding_days"],
        "input_range": [-0.5, 0.5],
        "artifact": "model.joblib",
        "artifact_sha256": sha256(model_path),
        "python_version": platform.python_version(),
        "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__,
        "joblib_version": joblib.__version__,
        "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_mode": "research (no final-test data loaded)",
    }
    (artifacts / "model_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"metadata": metadata, "reference": reference}, indent=2))


if __name__ == "__main__":
    main()
