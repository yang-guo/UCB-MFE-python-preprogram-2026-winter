"""One-off export of my HW4 final-test model for the HW5 service (Ananya Das).

Run once, from this folder, with the repository-root environment (it has DuckDB and the
HW4 dependencies); the HW5 service environment does not need either:

    ../../../.venv/bin/python export_model.py            # rebuild, save, then verify
    .venv/bin/python export_model.py --verify            # re-check the saved files only

What it does:
1. Imports my unchanged ../../HW4/Ananya_Das/hw4.py and repeats the model steps of
   `hw4.py final` (run_final): check the frozen settings, load data through the test end,
   build the month-end panel, compute the refit schedule, and fit with HW4's own `fit`.
   It keeps only the LAST refit (2026-01-02), the model that was live at the end of the
   final test. `fit` depends only on the panel and the fit date, so fitting that one date
   gives the same model run_final fitted.
2. Saves the fitted pipeline (imputer + Ridge) with joblib, plus metadata derived from the
   fitted model and HW4's settings. On a rerun it never overwrites artifacts/model.joblib:
   it checks the rebuilt model is identical and leaves the model and metadata untouched.
3. Records real test-period observations, exactly as HW4's ModelStrategy scored them, and
   the HW4 pipeline's prediction for each at full precision: one complete reference plus
   one example per missing-value group (price, financial, log_cap) where real data has one.
4. Re-loads the saved files in a fresh process (without hw4.py) and checks that the saved
   model reproduces those predictions within 1e-12.

The final-test date lock (hw4.RUN_FINAL_TEST) is lifted at runtime for this process only.
The HW4 final test already used this data once; nothing about the model is changed or tuned.
The FastAPI service never imports this file or hw4.py.
"""

import argparse
import hashlib
import importlib
import json
import platform
import resource
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
import threadpoolctl

HW5_DIR = Path(__file__).resolve().parent
HW4_DIR = HW5_DIR.parents[1] / "HW4" / "Ananya_Das"
ARTIFACT_DIR = HW5_DIR / "artifacts"
MODEL_FILE = ARTIFACT_DIR / "model.joblib"
METADATA_FILE = ARTIFACT_DIR / "model_metadata.json"
REFERENCE_FILE = ARTIFACT_DIR / "reference_prediction.json"
TOLERANCE = 1e-12  # saved model vs the HW4 pipeline; both are the same float64 computation


# Rebuild the HW4 final-test model


def import_hw4():
    """Import my unchanged hw4.py without writing bytecode into the HW4 folder."""
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(HW4_DIR))
    return importlib.import_module("hw4")


def rebuild_final_refit(hw4):
    """Repeat run_final()'s model-building steps and return the last refit."""
    hw4.RUN_FINAL_TEST = True  # lifts the date lock for this process only; hw4.py is unchanged
    try:
        frozen = hw4.check_frozen()  # refuses if any setting differs from frozen_config.json
        _, _, _, calendar, builder, load_notes = hw4.load(hw4.TEST_END)
        panel = builder.panel(hw4.month_end_sessions(calendar, hw4.DEV_START, hw4.TEST_END))
        first_trade = calendar[calendar >= hw4.TEST_START][0]
        fit_dates = hw4.refit_dates(calendar, first_trade, hw4.TEST_END)
        fit_date = fit_dates[-1]
        model = hw4.fit(panel, fit_date, frozen["ridge_alpha"])
    finally:
        hw4.RUN_FINAL_TEST = False

    # The schedule must be the one my final test recorded, or this is not the same model.
    listed = ", ".join(str(d.date()) for d in fit_dates)
    if f"Models refit at: {listed}." not in (HW4_DIR / "final_results.md").read_text():
        raise SystemExit(f"Refit dates {listed} do not match HW4's final_results.md")

    rows = hw4.training_rows(panel, fit_date)
    assert len(rows) == model.n_train_, "training rows differ from the fitted model"
    assert model.named_steps["ridge"].alpha == frozen["ridge_alpha"]
    return SimpleNamespace(
        frozen=frozen,
        builder=builder,
        panel=panel,
        fit_dates=fit_dates,
        fit_date=fit_date,
        model=model,
        rows=rows,
        load_notes=load_notes,
    )


# Reference observations from the real test period


def as_feature_dict(row, features):
    """Feature values in model order; NaN (missing in HW4) becomes JSON null."""
    return {name: (None if pd.isna(row[name]) else float(row[name])) for name in features}


def scored_day(hw4, run, signal_date):
    """One signal date's features exactly as HW4's ModelStrategy saw them, and their scores."""
    day = run.panel.loc[run.panel.date.eq(signal_date)].set_index("symbol").sort_index()
    # The panel rows are exactly what ModelStrategy.pick() scored on that date.
    strategy_view = run.builder.features_on(signal_date)[hw4.FEATURES].sort_index()
    pd.testing.assert_frame_equal(day[hw4.FEATURES], strategy_view, check_names=False)
    batch = pd.Series(run.model.predict(day[hw4.FEATURES]), index=day.index)
    return day, batch


def record(hw4, run, day, batch, symbol, signal_date, category):
    """Score one stock alone with the HW4 pipeline and keep the prediction exactly."""
    single = float(run.model.predict(day.loc[[symbol], hw4.FEATURES])[0])
    gap = abs(single - float(batch[symbol]))
    assert gap <= TOLERANCE, f"{symbol}: one-row and cross-section predictions differ by {gap}"
    row = day.loc[symbol]
    # The refit HW4's final test used on this date (as in ModelStrategy.model_for).
    hw4_refit = max(d for d in run.fit_dates if d <= signal_date)
    return {
        "category": category,
        "symbol": symbol,
        "signal_date": str(signal_date.date()),
        "missing_features": [name for name in hw4.FEATURES if pd.isna(row[name])],
        "hw4_backtest_refit": str(hw4_refit.date()),
        "is_hw4_backtest_score": bool(hw4_refit == run.fit_date),
        "features": as_feature_dict(row, hw4.FEATURES),
        "prediction": single,
        "prediction_hex": single.hex(),
        "cross_section_prediction": float(batch[symbol]),
    }


def pick_references(hw4, run):
    """Real final-test observations scored by the served pipeline.

    Main reference: the first stock (alphabetically) with all eight features on the last
    test-period signal date in the panel. Then, for each missing-value category, the first
    stock with only that group missing, on the newest final-test month-end where one exists.
    A category with no real observation is reported in `not_found`, never made up.
    """
    signal_date = run.panel.date.max()
    assert signal_date > run.fit_date, "the reference must come after the fit date"
    day, batch = scored_day(hw4, run, signal_date)
    complete = day[hw4.FEATURES].notna().all(axis=1)
    if not complete.any():
        raise SystemExit(f"No stock on {signal_date.date()} has all eight features")
    main = record(hw4, run, day, batch, day.index[complete][0], signal_date, "complete")

    groups = {
        "missing_price_feature": list(hw4.PRICE_FEATURES),
        "missing_financial_feature": list(hw4.FINANCIAL_FEATURES),
        "missing_log_cap": ["log_cap"],
    }
    test_rows = run.panel.loc[run.panel.date.ge(run.fit_dates[0])]  # final-test signal dates
    extra, not_found = [], []
    for category, columns in groups.items():
        others = [name for name in hw4.FEATURES if name not in columns]

        def only_this_group_missing(frame, columns=columns, others=others):
            return frame[columns].isna().any(axis=1) & frame[others].notna().all(axis=1)

        hits = test_rows.loc[only_this_group_missing(test_rows)]
        if hits.empty:
            not_found.append(
                {
                    "category": category,
                    "searched": (
                        f"{test_rows.date.nunique()} final-test month-ends, "
                        f"{test_rows.date.min().date()} to {test_rows.date.max().date()} "
                        f"({len(test_rows):,} stock-dates)"
                    ),
                    "result": f"no observation with only {', '.join(columns)} missing",
                    "any_missing_in_test_period": int(
                        test_rows[columns].isna().any(axis=1).sum()
                    ),
                }
            )
            continue
        hit_date = hits.date.max()
        hit_day, hit_batch = scored_day(hw4, run, hit_date)
        symbol = hit_day.index[only_this_group_missing(hit_day)][0]
        extra.append(record(hw4, run, hit_day, hit_batch, symbol, hit_date, category))
    return signal_date, len(day), main, extra, not_found


# Metadata derived from the fitted model and HW4's settings


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def hw4_commit():
    """The last commit that changed my HW4 folder (PR #296), if git is available."""
    try:
        out = subprocess.run(
            ["git", "log", "-1", "--format=%H", "--", "."],
            cwd=HW4_DIR,
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def json_params(estimator):
    """Plain estimator parameters; NaN (SimpleImputer's missing_values) becomes "NaN"."""
    params = {}
    for key, value in estimator.get_params().items():
        if isinstance(value, float) and not np.isfinite(value):
            params[key] = str(value)
        elif value is None or isinstance(value, (bool, int, float, str)):
            params[key] = value
    return params


def iso_date(value):
    return str(pd.Timestamp(value).date())


def build_metadata(hw4, run, model_version, signal_date, n_scored):
    model, rows, frozen = run.model, run.rows, run.frozen
    imputer, ridge = model.named_steps["impute"], model.named_steps["ridge"]
    features = list(hw4.FEATURES)
    return {
        "model_version": model_version,
        "description": (
            "My HW4 stock-return model: the last refit of the HW4 final test "
            "(`hw4.py final`), rebuilt with HW4's own functions and frozen settings."
        ),
        "model": {
            "type": "sklearn.pipeline.Pipeline",
            "steps": [
                {"name": name, "class": type(step).__name__, "params": json_params(step)}
                for name, step in model.steps
            ],
            "ridge_coefficients": dict(zip(features, map(float, ridge.coef_))),
            "ridge_intercept": float(ridge.intercept_),
        },
        "features": {
            "order": features,
            "price_features": list(hw4.PRICE_FEATURES),
            "financial_features": list(hw4.FINANCIAL_FEATURES),
            "units": (
                "Each feature is a percentile rank among that signal date's eligible stocks "
                "(pandas rank(pct=True)), so values lie in (0, 1]. Raw definitions: mom_21 = "
                "21-session return; mom_252_21 = return from 252 to 21 sessions ago; vol_63 = "
                "annualized std of 63 daily returns; drawdown_252 = price / 252-session high "
                "- 1; operating_roa, accruals_to_assets, debt_to_assets = latest quarterly "
                "filing, usable from filing date + 1 day, blank after 550 days; log_cap = log "
                "of the quarter's market-cap proxy."
            ),
            "training_min": rows[features].min().to_dict(),
            "training_max": rows[features].max().to_dict(),
            "training_missing_share": rows[features].isna().mean().round(6).to_dict(),
        },
        "missing_values": {
            "in_hw4": "A feature is NaN when its raw value is unavailable for that stock/date.",
            "pipeline": (
                f"SimpleImputer(strategy={imputer.strategy!r}, fill_value={imputer.fill_value}): "
                "a missing rank becomes the middle rank. Nothing is learned in preprocessing."
            ),
        },
        "training": {
            "fit_date": iso_date(model.fit_date_),
            "train_start": frozen["train_start"],
            "first_signal_date": iso_date(rows.date.min()),
            "last_signal_date": iso_date(rows.date.max()),
            "last_target_end": iso_date(model.train_label_end_),
            "rule": "month-end rows from train_start whose 90-day target ended before fit_date",
            "n_rows": int(model.n_train_),
            "n_signal_dates": int(rows.date.nunique()),
            "n_symbols": int(rows.symbol.nunique()),
            "data_loaded_through": iso_date(hw4.TEST_END),
            "database": hw4.FULL_DB.name,
            "loader_warnings": run.load_notes,
            "final_test_refit_dates": [iso_date(d) for d in run.fit_dates],
        },
        "target": {
            "definition": frozen["target"],
            "holding_days": hw4.HOLDING_DAYS,
            "entry": "next session's close after the signal date",
            "exit": "first close on or after entry + 90 calendar days",
            "training_cap_quantile": frozen["target_winsor"],
        },
        "output": {
            "name": "prediction",
            "units": "fraction (0.01 = 1 percentage point)",
            "interpretation": (
                "Predicted 90-day return minus that date's median eligible stock. HW4 used it "
                "only to rank stocks; with alpha = 1e7 predictions stay close to the intercept."
            ),
        },
        "reference_data": {
            "signal_date": iso_date(signal_date),
            "stocks_scored_that_day": n_scored,
            "file": REFERENCE_FILE.name,
        },
        "hw4_config": frozen,
        "provenance": {
            "hw4_folder": "Homeworks/HW4/Ananya_Das",
            "hw4_commit": hw4_commit(),
            "hw4_py_sha256": sha256(HW4_DIR / "hw4.py"),
            "frozen_config_sha256": sha256(HW4_DIR / "frozen_config.json"),
            "export_script": Path(__file__).name,
            "exported_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
        "environment": {
            "python": platform.python_version(),
            "scikit-learn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "joblib": joblib.__version__,
            "threadpoolctl": threadpoolctl.__version__,
        },
    }


def write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def check_same_model(saved, bundle):
    """Stop unless the saved artifact is exactly the model rebuilt from HW4."""
    old, new = saved["pipeline"], bundle["pipeline"]
    differences = [
        name
        for name, same in [
            ("model_version", saved["model_version"] == bundle["model_version"]),
            ("features", list(saved["features"]) == list(bundle["features"])),
            ("steps", [n for n, _ in old.steps] == [n for n, _ in new.steps]),
            (
                "params",
                [json_params(s) for _, s in old.steps]
                == [json_params(s) for _, s in new.steps],
            ),
            ("coefficients", np.array_equal(old[-1].coef_, new[-1].coef_)),
            ("intercept", old[-1].intercept_ == new[-1].intercept_),
            ("fit_date", old.fit_date_ == new.fit_date_),
            ("n_train", old.n_train_ == new.n_train_),
            ("train_label_end", old.train_label_end_ == new.train_label_end_),
        ]
        if not same
    ]
    if differences:
        raise SystemExit(
            f"The saved {MODEL_FILE.name} differs from the rebuilt model ({differences}); "
            "refusing to overwrite it."
        )


def check_main_reference_unchanged(main):
    """The complete reference recorded first must stay exactly as it was."""
    if not REFERENCE_FILE.exists():
        return
    old = json.loads(REFERENCE_FILE.read_text())["reference"]
    changed = [key for key in old if old[key] != main.get(key)]
    if changed:
        raise SystemExit(
            f"The main reference would change ({changed}); refusing to rewrite it."
        )


def export():
    hw4 = import_hw4()
    print("Rebuilding the HW4 final-test model (loads data through 2026-09-30)...", flush=True)
    run = rebuild_final_refit(hw4)
    model_version = f"hw4-ridge-final-refit-{run.fit_date.date()}"
    signal_date, n_scored, main, extra, not_found = pick_references(hw4, run)

    ARTIFACT_DIR.mkdir(exist_ok=True)
    bundle = {"pipeline": run.model, "model_version": model_version, "features": hw4.FEATURES}
    if MODEL_FILE.exists():
        # Never replace a saved artifact: a rerun must rebuild exactly the same model.
        check_same_model(joblib.load(MODEL_FILE), bundle)
        print(
            f"{MODEL_FILE.name} already holds this exact model; it and the metadata are kept."
        )
    else:
        joblib.dump(bundle, MODEL_FILE)
        write_json(
            METADATA_FILE, build_metadata(hw4, run, model_version, signal_date, n_scored)
        )
        print(f"Saved {MODEL_FILE.name} and {METADATA_FILE.name}")
    check_main_reference_unchanged(main)
    write_json(
        REFERENCE_FILE,
        {
            "model_version": model_version,
            "feature_order": list(hw4.FEATURES),
            "how_recorded": (
                "HW4's ModelStrategy features for the signal date (builder.features_on), scored "
                "by the pipeline from hw4.fit on a one-row DataFrame with HW4's column names, "
                "before any API code. cross_section_prediction is the same stock scored with "
                "the whole day, as HW4's strategy did. Missing features are null (NaN in HW4). "
                "is_hw4_backtest_score is false when HW4's backtest scored that date with an "
                "earlier yearly refit (hw4_backtest_refit); the features are still real HW4 "
                "test-period inputs and the prediction is the served pipeline's."
            ),
            "tolerance": TOLERANCE,
            "reference": main,
            "additional_examples": extra,
            "not_found": not_found,
        },
    )

    peak_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9  # bytes on macOS
    print(f"Saved {REFERENCE_FILE.name}: 1 reference, {len(extra)} additional example(s)")
    for gap in not_found:
        print(f"  not found: {gap['category']}: {gap['result']}")
    print(f"Model version: {model_version}; fit date {run.fit_date.date()}")
    print(
        f"Training rows: {run.model.n_train_:,} on {run.rows.date.nunique()} month-ends, "
        f"{run.rows.date.min().date()} to {run.rows.date.max().date()}; "
        f"last target ended {run.model.train_label_end_.date()}"
    )
    print(f"Peak memory (resident): {peak_gb:.2f} GB")


# Fresh-process verification (no hw4.py, no database)


def verify():
    bundle = joblib.load(MODEL_FILE)
    metadata = json.loads(METADATA_FILE.read_text())
    reference = json.loads(REFERENCE_FILE.read_text())
    assert "hw4" not in sys.modules, "verification must not depend on hw4.py"

    pipeline, features = bundle["pipeline"], bundle["features"]
    assert features == metadata["features"]["order"] == reference["feature_order"]
    assert list(pipeline.feature_names_in_) == features
    assert bundle["model_version"] == metadata["model_version"] == reference["model_version"]
    assert sklearn.__version__ == metadata["environment"]["scikit-learn"], "sklearn mismatch"

    print(f"Verifying {bundle['model_version']} with {sys.executable}")
    ok = True
    for example in [reference["reference"], *reference["additional_examples"]]:
        expected = example["prediction"]
        assert expected == float.fromhex(example["prediction_hex"])
        nulls = [name for name in features if example["features"][name] is None]
        assert nulls == example.get("missing_features", nulls), "null pattern mismatch"
        row = pd.DataFrame([example["features"]], columns=features, dtype=float)  # null -> NaN
        loaded = float(pipeline.predict(row)[0])
        diff = abs(loaded - expected)
        ok = ok and diff <= TOLERANCE
        print(
            f"  [{example.get('category', 'complete')}] nulls={nulls or 'none'}\n"
            f"  {example['symbol']} {example['signal_date']}: HW4 {expected!r} | "
            f"saved artifact {loaded!r} | abs diff {diff:.3e} | "
            f"{'PASS' if diff <= TOLERANCE else 'FAIL'} (<= {TOLERANCE:g})"
        )
    for path in (MODEL_FILE, METADATA_FILE, REFERENCE_FILE):
        print(f"  readable: {path.relative_to(HW5_DIR)} ({path.stat().st_size:,} bytes)")
    if not ok:
        raise SystemExit("Saved artifact does not reproduce the HW4 predictions")
    print("Verification passed.")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--verify", action="store_true", help="only check the saved files")
    if parser.parse_args().verify:
        verify()
        return
    export()
    # A separate interpreter proves the artifact loads on its own, without hw4.py in memory.
    subprocess.run([sys.executable, str(Path(__file__).resolve()), "--verify"], check=True)


if __name__ == "__main__":
    main()
