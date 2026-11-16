"""Export the fitted HW4 model as a servable artifact.  Run once; the service never retrains.

Which model: HW4 refit its Ridge model at the first signal of every quarter.  The
last fit of the HW4 final test was on **2026-07-02** (trained only on outcomes
that finished by 2026-06-29) and it scored the 2026-Q3 signals.  That is the
model served here: it is the most recent model HW4 produced.

This script re-runs HW4's own code (no new modelling code), then checks that the
re-created model reproduces the scores HW4 recorded during its final run
(``outputs/test_predictions.csv.gz``).  Only then does it save the artifact.

Needs the HW4 code and its database (not needed to *serve*):
    HW4_DIR=<path to Homeworks/HW4/Nirmit_Ghughu on branch hw_4_Nirmit_Ghughu> \
        python export_model.py
"""

import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.pipeline import Pipeline

from hw4_model import INPUT_COLUMNS, NUMERIC_FEATURES, HW4RidgeDesign

HERE = Path(__file__).resolve().parent
HW4_DIR = Path(os.environ.get("HW4_DIR", HERE.parents[1] / "HW4" / "Nirmit_Ghughu")).resolve()
FIT_DATE = pd.Timestamp("2026-07-02")
MODEL_VERSION = "hw4-ridge-rank-2026-07-02"


def main():
    sys.path.insert(0, str(HW4_DIR))
    from data import FULL_DB, build_panel  # HW4 modules
    from model import _design, add_rank_features, fit, training_rows

    panel, _ = build_panel(FULL_DB, end="2026-09-30")
    ranked = add_rank_features(panel)
    train = training_rows(ranked, FIT_DATE)  # same call HW4's walk-forward makes
    ridge = fit(train, "ridge", "rank")
    assert train.label_end.max() < FIT_DATE

    # 1) The re-created model must equal the one HW4 used: compare with HW4's recorded scores.
    block = ranked.loc[ranked.date >= FIT_DATE]
    recorded = pd.read_csv(
        HW4_DIR / "outputs" / "test_predictions.csv.gz", parse_dates=["date"]
    )
    recorded = recorded.loc[recorded.date >= FIT_DATE, ["date", "symbol", "score"]]
    check = block[["date", "symbol"]].assign(score_now=ridge.predict(_design(block, "ridge")))
    check = check.merge(recorded, on=["date", "symbol"], validate="one_to_one")
    max_diff = float((check.score_now - check.score).abs().max())
    print(
        f"re-created vs HW4-recorded scores: {len(check):,} rows, max |diff| = {max_diff:.2e}"
    )
    assert len(check) == len(block) and max_diff < 1e-6

    # 2) Wrap HW4's design step + the fitted Ridge as one sklearn pipeline (no refit).
    sectors = [
        c
        for c in ridge.feature_names_in_
        if c not in NUMERIC_FEATURES and not c.endswith("_missing")
    ]
    design = HW4RidgeDesign(numeric_features=NUMERIC_FEATURES, sectors=sectors).fit()
    assert list(design.get_feature_names_out()) == list(ridge.feature_names_in_)
    pipeline = Pipeline([("design", design), ("ridge", ridge)])
    direct = pipeline.predict(block[INPUT_COLUMNS])
    assert np.allclose(
        direct,
        check.set_index(["date", "symbol"])
        .loc[list(zip(block.date, block.symbol)), "score"]
        .to_numpy(),
        atol=1e-6,
    )

    artifacts = HERE / "artifacts"
    artifacts.mkdir(exist_ok=True)
    bundle = {
        "pipeline": pipeline,
        "model_version": MODEL_VERSION,
        "sectors": sectors,
        "numeric_features": NUMERIC_FEATURES,
    }
    joblib.dump(bundle, artifacts / "hw4_ridge.joblib")
    meta = {
        "model_version": MODEL_VERSION,
        "model": "sklearn Ridge(alpha=1000), HW4 rank target, quarterly walk-forward refit",
        "fit_date": str(FIT_DATE.date()),
        "training_rows": int(len(train)),
        "training_signals": [str(train.date.min().date()), str(train.date.max().date())],
        "latest_training_label_end": str(train.label_end.max().date()),
        "target": "within-date percentile rank of the 90-calendar-day return minus 0.5",
        "horizon": "entry at next session close, exit first close on/after entry + 90 days",
        "features": NUMERIC_FEATURES + ["sector"],
        "sectors": sectors,
        "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "max_abs_diff_vs_hw4_recorded_scores": max_diff,
    }
    (artifacts / "model_metadata.json").write_text(json.dumps(meta, indent=2) + "\n")

    # 3) Reference examples recorded from HW4 (last signal date of the test).
    last = block.loc[block.date == block.date.max()].merge(recorded, on=["date", "symbol"])
    has_missing = last[NUMERIC_FEATURES].isna().any(axis=1)
    picks = pd.concat(
        [
            last.nlargest(1, "score"),  # the model's top pick
            last.loc[has_missing].nsmallest(1, "score"),  # a row with missing features
            last.iloc[[(last.score - last.score.median()).abs().argmin()]],  # a typical stock
        ]
    )

    def payload(row):
        out = {c: (None if pd.isna(row[c]) else float(row[c])) for c in NUMERIC_FEATURES}
        out["sector"] = row["sector"]
        return out

    refs = [
        {
            "source": "HW4 outputs/test_predictions.csv.gz (recorded during HW4 run_final.py)",
            "signal_date": str(r.date.date()),
            "symbol": r.symbol,
            "hw4_recorded_score": float(r.score),
            "request": payload(r),
        }
        for _, r in picks.iterrows()
    ]
    (HERE / "tests" / "reference_examples.json").write_text(json.dumps(refs, indent=2) + "\n")
    (HERE / "example_request.json").write_text(json.dumps(refs[0]["request"], indent=2) + "\n")
    print(
        f"saved {MODEL_VERSION}: {len(sectors)} sectors, references {[r['symbol'] for r in refs]}"
    )


if __name__ == "__main__":
    main()
