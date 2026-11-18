"""Serialize the final HW4 model fit as a scikit-learn Pipeline (run once, offline).

Run this with the **HW4 training environment** (same library versions as
requirements.txt), from the HW5 submission folder:

    ../../HW4/Vasileios_Dimitropoulos/HW4/Scripts/python.exe scripts/export_hw4_model.py \
        --hw4-dir <folder containing hw4lib/ and outputs/ from my HW4 submission>

What it does:

1. Loads the HW4 feature panel (``outputs/panel.parquet``, built by HW4's
   notebook from the bCourses database). The database itself is not opened.
2. Rebuilds the last quarterly walk-forward fit of the HW4 test (fit date
   2026-07-06), using HW4's own ``training_rows`` and ``RidgeRanker("y_rank", 1000)``.
3. Fits ``Pipeline([RankFeatureDesign, Ridge(alpha=1000)])`` on the same rows
   and checks that it gives the same coefficients and the same scores as the
   HW4 code, and the same scores that the HW4 notebook recorded in
   ``outputs/test_predictions.parquet`` before this service existed.
4. Writes ``artifacts/model.joblib``, ``artifacts/model_metadata.json``,
   ``example_request.json`` and ``tests/hw4_reference_predictions.json``.

Ridge has a closed-form solution, so refitting on identical rows reproduces the
HW4 fit exactly (up to floating-point rounding). Nothing is tuned or changed here.
"""

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
import threadpoolctl
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

HW5_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HW5_ROOT))
from model_pipeline import (  # noqa: E402  (import after the sys.path setup above)
    ARTIFACT_PATH,
    INPUT_COLUMNS,
    METADATA_PATH,
    RANK_FEATURES,
    SECTORS,
    RankFeatureDesign,
    sha256_of,
)

FIT_DATE = pd.Timestamp("2026-07-06")  # last quarterly refit of the HW4 test
TRAIN_START = "2010-01-01"  # HW4 CONFIG["train_start"]
ALPHA = 1000.0  # HW4 frozen choice: "Ridge alpha=1000 rank target"
MODEL_VERSION = "hw4-ridge-a1000-rank-fit-2026-07-06"
TOLERANCE = 1e-9


def parse_args():
    default_hw4 = HW5_ROOT.parents[1] / "HW4" / "Vasileios_Dimitropoulos"
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--hw4-dir", type=Path, default=default_hw4, help="has hw4lib/")
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=None,
        help="HW4 outputs/ (default: <hw4-dir>/outputs)",
    )
    return parser.parse_args()


def request_from_row(row: pd.Series) -> dict:
    """A panel row -> the JSON request body (NaN ranks become null)."""
    body = {c: (None if pd.isna(row[c]) else float(row[c])) for c in RANK_FEATURES}
    body["sector"] = str(row["sector"])
    return body


def main():
    args = parse_args()
    outputs = args.outputs_dir or args.hw4_dir / "outputs"
    sys.path.insert(0, str(args.hw4_dir.resolve()))
    from hw4lib.model import RidgeRanker, training_rows  # my HW4 code, unchanged

    panel = pd.read_parquet(outputs / "panel.parquet")
    # Same restoration of the categorical dtype as hw4lib.pipeline.load_everything.
    panel["sector_cat"] = pd.Categorical(panel.sector_cat.astype(str), categories=SECTORS)
    panel = panel.loc[panel.date.ge(TRAIN_START)]  # as in HW4 notebook cell 18
    recorded = pd.read_parquet(outputs / "test_predictions.parquet")
    assert recorded.fit_date.max() == FIT_DATE, "FIT_DATE is not HW4's last refit"

    # 1. The HW4 fit, with the HW4 code.
    train = training_rows(panel, FIT_DATE, TRAIN_START)
    hw4_model = RidgeRanker("y_rank", ALPHA).fit(train)

    # 2. The same fit as a serializable sklearn Pipeline.
    pipeline = Pipeline([("design", RankFeatureDesign()), ("ridge", Ridge(alpha=ALPHA))])
    pipeline.fit(train[INPUT_COLUMNS], train["y_rank"])
    ridge = pipeline.named_steps["ridge"]
    coef_diff = float(np.max(np.abs(ridge.coef_ - hw4_model.model.coef_)))
    intercept_diff = abs(float(ridge.intercept_ - hw4_model.model.intercept_))
    assert coef_diff < TOLERANCE and intercept_diff < TOLERANCE, (coef_diff, intercept_diff)

    # 3. Scores for every row this fit scored in HW4, three ways.
    scored = panel.loc[panel.date.ge(FIT_DATE)].merge(
        recorded.loc[recorded.fit_date.eq(FIT_DATE), ["date", "symbol", "score"]],
        on=["date", "symbol"],
        validate="one_to_one",
    )
    assert len(scored) == int(recorded.fit_date.eq(FIT_DATE).sum())
    pipeline_scores = pipeline.predict(scored[INPUT_COLUMNS])
    vs_hw4_code = float(np.max(np.abs(pipeline_scores - hw4_model.predict(scored))))
    vs_recorded = float(np.max(np.abs(pipeline_scores - scored.score.to_numpy())))
    assert vs_hw4_code < TOLERANCE and vs_recorded < TOLERANCE, (vs_hw4_code, vs_recorded)
    print(f"Training rows: {len(train):,}; scored rows checked: {len(scored):,}")
    print(
        f"max |coef diff| {coef_diff:.2e}; max |score diff| vs HW4 code {vs_hw4_code:.2e}, "
        f"vs recorded HW4 predictions {vs_recorded:.2e}"
    )

    # 4. Reference cases from the last HW4 signal date (2026-09-28).
    last = scored.loc[scored.date.eq(scored.date.max())].set_index("symbol")
    n_missing = last[RANK_FEATURES].isna().sum(axis=1)
    complete = last.loc[n_missing.eq(0)]
    cases = {
        "complete_features": "AAPL" if "AAPL" in complete.index else complete.index[0],
        "most_missing_features": n_missing.idxmax(),
        "highest_hw4_score": last.score.idxmax(),
    }
    source = (
        "Homeworks/HW4/Vasileios_Dimitropoulos/outputs/test_predictions.parquet, column "
        "'score', written by hw4_strategy.ipynb (cell 18, quarterly walk-forward of "
        "RidgeRanker('y_rank', 1000)) on the full bCourses database before HW5 existed."
    )
    references = {
        "model_version": MODEL_VERSION,
        "source": source,
        "tolerance_abs": TOLERANCE,
        "cases": [
            {
                "case": case,
                "symbol": symbol,
                "signal_date": str(last.loc[symbol, "date"].date()),
                "fit_date": str(FIT_DATE.date()),
                "missing_features": int(n_missing[symbol]),
                "request": request_from_row(last.loc[symbol]),
                "expected_prediction": float(last.loc[symbol, "score"]),
            }
            for case, symbol in cases.items()
        ],
    }
    (HW5_ROOT / "tests" / "hw4_reference_predictions.json").write_text(
        json.dumps(references, indent=2) + "\n", encoding="utf-8"
    )
    (HW5_ROOT / "example_request.json").write_text(
        json.dumps(references["cases"][0]["request"], indent=2) + "\n", encoding="utf-8"
    )

    # 5. The artifact and its metadata (hash is checked by the service before loading).
    ARTIFACT_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(pipeline, ARTIFACT_PATH)
    metadata = {
        "model_version": MODEL_VERSION,
        "description": "My HW4 model: Ridge(alpha=1000) on same-date percentile ranks of 28 "
        "features + sector, rank target, last quarterly walk-forward refit of the HW4 test.",
        "fit_date": str(FIT_DATE.date()),
        "training_rows": len(train),
        "training_signal_dates": [str(train.date.min().date()), str(train.date.max().date())],
        "last_training_label_end": str(train.label_end.max().date()),
        "scores_signals_from": str(FIT_DATE.date()),
        "input_columns": INPUT_COLUMNS,
        "sectors": SECTORS,
        "target": "Same-date percentile rank of the 90-calendar-day forward return among "
        "eligible stocks, minus 0.5 (range -0.5..+0.5).",
        "prediction_units": "Forecast of that centred rank (dimensionless). Higher means a "
        "higher expected 90-day return relative to the same-date universe. HW4 buys the 10 "
        "highest scores each week.",
        "export_checks": {
            "max_abs_coef_diff_vs_hw4_code": coef_diff,
            "max_abs_score_diff_vs_hw4_code": vs_hw4_code,
            "max_abs_score_diff_vs_recorded_hw4_predictions": vs_recorded,
            "rows_checked": len(scored),
        },
        "library_versions": {
            "python": platform.python_version(),
            "scikit-learn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "joblib": joblib.__version__,
            "threadpoolctl": threadpoolctl.__version__,
        },
        "artifact_file": "artifacts/model.joblib",
        "artifact_sha256": sha256_of(ARTIFACT_PATH),
        "exported_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(
        f"Wrote {ARTIFACT_PATH.relative_to(HW5_ROOT)} ({ARTIFACT_PATH.stat().st_size:,} bytes)"
    )
    print(f"Reference cases: {cases}")


if __name__ == "__main__":
    main()
