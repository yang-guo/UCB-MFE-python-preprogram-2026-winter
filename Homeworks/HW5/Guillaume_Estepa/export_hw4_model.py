"""Export the HW4 model that this service serves. Run once; the service never imports this file.

It re-runs my HW4 walk-forward code unchanged, keeps the Ridge fitted on 2026-01-05 (the last
fit of the HW4 test, which scored every signal date from 2026-01-05 to 2026-09-28), puts HW4's
missing-value rule in front of it, checks the result against HW4's own numbers, and writes:

    artifacts/hw4_ridge_rank.joblib   the fitted pipeline, its version and its feature order
    artifacts/model_metadata.json     provenance, readable without unpickling anything
    tests/data/hw4_reference.json     real HW4 inputs with the predictions HW4 made for them
    example_request.json              the first reference input, as a request body

It needs my HW4 folder (pull request #248), the bCourses database in Lectures/modeling/data/
and the course environment HW4 was trained in. From the repository root:

    uv sync --locked
    uv run python Homeworks/HW5/Guillaume_Estepa/export_hw4_model.py
"""

import hashlib
import json
import platform
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
import threadpoolctl
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

HERE = Path(__file__).resolve().parent
HW4_DIR = HERE.parents[1] / "HW4" / "Guillaume_Estepa"
HW4_SOURCE = "Homeworks/HW4/Guillaume_Estepa, commit 68d7d59 (pull request #248)"

MODEL_VERSION = "hw4-ridge_rank-2026-01-05"
SERVED_FIT = pd.Timestamp("2026-01-05")  # last walk-forward fit of the HW4 test

# Rows of the HW4 panel kept as reference cases: (signal date, symbol, why this row).
REFERENCE_ROWS = [
    ("2026-09-28", "NVDA", "Bought at the next close (HW4 lot 6299). No missing feature."),
    ("2026-01-20", "HAS", "Bought at the next close (HW4 lot 5907). Four missing features."),
    ("2026-09-28", "AMLX", "Lowest prediction of that date, not bought. Two missing features."),
]


def sha256(path: Path) -> str:
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def require(condition: bool, message: str) -> None:
    """Stop the export rather than write an artifact that does not match HW4."""
    if not condition:
        raise SystemExit(f"Export stopped: {message}")


def json_number(rank: np.float32) -> float | None:
    """A float32 rank as the shortest decimal that reads back to the same float32; NaN -> None."""
    if np.isnan(rank):
        return None
    short = float(str(rank))
    require(np.float32(short) == rank, f"{rank!r} does not survive a round trip through JSON")
    return short


def write_json(path: Path, content) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content, indent=2) + "\n")


def main():
    # Reuse the HW4 modules exactly as submitted (they import each other by bare name).
    sys.path.insert(0, str(HW4_DIR))
    from data import FULL_DB, load_market_data
    from features import FEATURES, build_panel, weekly_grid
    from model import MODELS, TRAIN_START, refit_dates, walk_forward_predict
    from run_final import TEST_END, TEST_START
    from strategy import build_schedule

    # ---- 1. Check the database, then rebuild the HW4 panel and walk-forward fits with the
    # same steps as HW4 run_final.main.
    database_sha256 = sha256(FULL_DB)
    manifest = json.loads((FULL_DB.parent / "manifest.json").read_text())
    course_sha256 = manifest["files"][FULL_DB.name]["sha256"]
    require(database_sha256 == course_sha256, f"{FULL_DB.name} is not the course database")
    config = json.loads((HW4_DIR / "results" / "chosen_config.json").read_text())
    md = load_market_data(FULL_DB, end=TEST_END)
    sessions = md.sessions
    first_entry = sessions[sessions.searchsorted(pd.Timestamp(TEST_START))]
    first_signal = sessions[sessions.get_loc(first_entry) - 1]
    schedule = build_schedule(sessions, first_signal, TEST_END, config["n_tranches"])
    panel = build_panel(md, weekly_grid(sessions).union(pd.DatetimeIndex(list(schedule))))
    fits = refit_dates(panel.date, first_signal)
    panel["prediction"], fit_log, models = walk_forward_predict(panel, config["model"], fits)

    # ---- 2. Prove this re-run is the HW4 experiment: same fit log, same coefficients.
    date_columns = ["fit_date", "first_train_signal", "last_train_signal", "last_label_end"]
    submitted_log = pd.read_csv(HW4_DIR / "results" / "fit_log.csv", parse_dates=date_columns)
    fit_log_matches = fit_log.reset_index(drop=True).equals(submitted_log)
    require(fit_log_matches, "the fit log differs from HW4 results/fit_log.csv")
    require(fit_log.fit_date.max() == SERVED_FIT, f"the last fit is not {SERVED_FIT.date()}")
    ridge = models[SERVED_FIT]
    submitted = pd.read_csv(HW4_DIR / "results" / "ridge_coefficients.csv", index_col=0)
    require(list(submitted.index) == FEATURES, "the feature order differs from HW4 results")
    hw4_coefficients = submitted[str(SERVED_FIT.date())].to_numpy()
    coefficient_gap = float(np.abs(ridge.coef_ - hw4_coefficients).max())
    # The HW4 file keeps five decimals, so half a unit of the fifth is the most to expect.
    require(coefficient_gap < 5.1e-6, f"coefficients differ from HW4 by {coefficient_gap}")

    # ---- 3. Put HW4's missing-value rule in front of the fitted Ridge.
    # HW4 `model.design_matrix` gave the Ridge a float32 matrix in FEATURES order with missing
    # ranks set to 0.0, the middle rank. Nothing in that step is learned from data, so the
    # imputer is "fitted" on the training rows only to record that it expects 24 columns.
    def rank_matrix(frame):
        return frame[FEATURES].to_numpy(dtype=np.float32)  # NaN = missing

    target = MODELS[config["model"]][1]
    labelled = panel[target].notna() & (panel.date >= pd.Timestamp(TRAIN_START))
    train = panel.loc[labelled & (panel.label_end < SERVED_FIT)]
    served_log = fit_log.set_index("fit_date").loc[SERVED_FIT]
    require(len(train) == served_log.train_rows, "training rows differ from the HW4 fit log")
    imputer = SimpleImputer(strategy="constant", fill_value=0.0, keep_empty_features=True)
    imputer.fit(rank_matrix(train))
    pipeline = Pipeline([("missing_rank_to_middle", imputer), ("ridge", ridge)])

    # ---- 4. The pipeline must reproduce every prediction HW4 made with this fit.
    scored = panel.loc[panel.date >= SERVED_FIT].reset_index(drop=True)
    require(len(scored) == served_log.predicted_rows, "scored rows differ from the HW4 fit log")
    inputs, hw4_predictions = rank_matrix(scored), scored.prediction.to_numpy()
    batch_matches = bool(np.array_equal(pipeline.predict(inputs), hw4_predictions))
    require(batch_matches, "the pipeline does not reproduce the HW4 predictions")
    # The service scores one row at a time; HW4 scored them all in one batch. Measure the gap.
    one_by_one = np.array([pipeline.predict(inputs[i : i + 1])[0] for i in range(len(inputs))])
    single_row_gap = float(np.abs(one_by_one - hw4_predictions).max())

    # ---- 5. Reference cases: real HW4 inputs with the prediction HW4 itself produced.
    cases = []
    for signal_date, symbol, note in REFERENCE_ROWS:
        (row,) = scored.index[(scored.date == signal_date) & (scored.symbol == symbol)]
        features = {name: json_number(rank) for name, rank in zip(FEATURES, inputs[row])}
        cases.append(
            {
                "signal_date": signal_date,
                "symbol": symbol,
                "note": note,
                "features": features,
                "hw4_prediction": float(hw4_predictions[row]),
            }
        )

    # ---- 6. Write the artifact, its provenance, the reference cases and the example request.
    model_path = HERE / "artifacts" / "hw4_ridge_rank.joblib"
    model_path.parent.mkdir(exist_ok=True)
    bundle = {"pipeline": pipeline, "model_version": MODEL_VERSION, "features": list(FEATURES)}
    joblib.dump(bundle, model_path)

    missing_anywhere = panel[FEATURES].isna().sum()  # over all 11 fits' rows, not only this one
    clip_low, clip_high = train[target].quantile([0.01, 0.99])  # as HW4 walk_forward_predict
    metadata = {
        "model_version": MODEL_VERSION,
        "artifact": model_path.name,
        "artifact_sha256": sha256(model_path),
        "hw4_source": HW4_SOURCE,
        "hw4_config": config,
        "estimator": repr(ridge),
        "database": {"file": FULL_DB.name, "sha256": database_sha256},
        "fit": {
            "fit_date": str(SERVED_FIT.date()),
            "train_rows": int(served_log.train_rows),
            "first_train_signal": str(served_log.first_train_signal.date()),
            "last_train_signal": str(served_log.last_train_signal.date()),
            "last_label_end": str(served_log.last_label_end.date()),
            "scored_rows": int(served_log.predicted_rows),
            "first_scored_signal": str(scored.date.min().date()),
            "last_scored_signal": str(scored.date.max().date()),
        },
        "input": "24 same-date cross-sectional percentile ranks minus 0.5, float32, in this order",
        "features": list(FEATURES),
        "never_missing_features": [name for name in FEATURES if missing_anywhere[name] == 0],
        "missing_share_in_training": train[FEATURES].isna().mean().round(5).to_dict(),
        "preprocessing": "missing rank -> 0.0, the middle rank (HW4 model.design_matrix)",
        "target": "90-calendar-day forward return as a same-date percentile rank minus 0.5",
        "target_clipped_to_training_quantiles_1_99": [float(clip_low), float(clip_high)],
        "predictions_of_this_fit": {
            "mean": float(hw4_predictions.mean()),
            "std": float(hw4_predictions.std()),
            "min": float(hw4_predictions.min()),
            "max": float(hw4_predictions.max()),
        },
        "checks": {
            "fit_log_equals_hw4_results": fit_log_matches,
            "max_abs_coefficient_gap_to_hw4_csv": coefficient_gap,
            "rows_compared_with_hw4_predictions": len(scored),
            "batch_predictions_identical_to_hw4": batch_matches,
            "max_abs_gap_single_row_vs_hw4": single_row_gap,
        },
        "versions": {
            "python": platform.python_version(),
            "scikit-learn": sklearn.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "joblib": joblib.__version__,
            "threadpoolctl": threadpoolctl.__version__,
            "pandas (export only)": pd.__version__,
        },
    }
    write_json(model_path.with_name("model_metadata.json"), metadata)
    reference = {
        "source": "panel['prediction'] from HW4 model.walk_forward_predict, before any API existed",
        "model_version": MODEL_VERSION,
        "cases": cases,
    }
    write_json(HERE / "tests" / "data" / "hw4_reference.json", reference)
    write_json(HERE / "example_request.json", cases[0]["features"])

    print(f"Exported {MODEL_VERSION}: {model_path.stat().st_size} bytes")
    print(f"  identical to HW4 on {len(scored):,} rows; row by row: {single_row_gap:.2e}")
    for case in cases:
        print(f"  {case['signal_date']} {case['symbol']:5s} HW4: {case['hw4_prediction']}")


if __name__ == "__main__":
    main()
