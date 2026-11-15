"""Rebuild and serialize the fitted HW4 LightGBM model that this service serves.

Run this ONCE, in the HW4 training environment, from a checkout that contains
Homeworks/HW4/Shashank_Mishra (the HW4 code, its cached feature panel and the
course database). The service itself never runs this file, never retrains and
never touches the database: it only loads the files written to ``artifacts/``.

    python export_hw4_model.py                       # defaults to ../../HW4/Shashank_Mishra
    python export_hw4_model.py --hw4-dir PATH --db PATH/equities_top2000.duckdb

What it does, using HW4's own functions (nothing is re-implemented here):

1. Recomputes the HW4 final-test refit schedule (first signal on/after March 31
   of each year) and takes the LAST refit, the one HW4 used for its
   2026 predictions.
2. Fits ``run_hw4.FITTED["lgb_excess"]`` on ``hw4.model.training_rows`` for that
   date, exactly as ``run_hw4.stage_test`` did.
3. Checks that the refit reproduces the 2026 LightGBM feature importances saved
   in HW4's ``results/results.csv`` (so this is the same model, not a new one).
4. Records reference inputs and HW4 predictions for real stocks on the fit date,
   computed by HW4's ``RankLGBM.predict`` BEFORE any API code is involved.
5. Saves the fitted ``LGBMRegressor`` with its feature order and metadata.
"""

import argparse
import hashlib
import json
import platform
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ARTIFACTS = HERE / "artifacts"
DEFAULT_HW4 = HERE.parent.parent / "HW4" / "Shashank_Mishra"
SERVED_MODEL = "lgb_excess"
REFERENCE_SYMBOLS = 8  # how many real stock rows to record as reference predictions


def library_versions():
    import lightgbm
    import scipy
    import sklearn

    return {
        "python": platform.python_version(),
        "lightgbm": lightgbm.__version__,
        "scikit-learn": sklearn.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "joblib": joblib.__version__,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hw4-dir", type=Path, default=DEFAULT_HW4)
    ap.add_argument("--db", default=None, help="path to equities_top2000.duckdb")
    args = ap.parse_args()

    sys.path.insert(0, str(args.hw4_dir.resolve()))
    warnings.filterwarnings("ignore", category=UserWarning)
    import run_hw4 as R  # noqa: E402  (HW4 code, imported from its own folder)
    from hw4 import data
    from hw4 import model as M

    db = data.find_database(args.db)
    md, panel = R.load_everything(db)
    sectors = sorted(panel.sector.unique())

    # 1. The HW4 final-test refit schedule (copied logic: run_hw4.stage_test).
    sig = R.signal_dates(md, R.TEST_START - pd.Timedelta(days=1), R.TEST_END)
    fit_dates = sorted(
        {
            sig[sig >= pd.Timestamp(f"{y}-03-31")][0]
            for y in range(2016, 2027)
            if (sig >= pd.Timestamp(f"{y}-03-31")).any()
        }
    )
    fit_date = fit_dates[-1]

    # 2. Fit exactly as HW4 did.
    train = M.training_rows(panel, fit_date)
    assert train.label_end.max() < fit_date, "training label not yet observed"
    wrapper = R.FITTED[SERVED_MODEL](R.FEATURES, sectors).fit(train)
    estimator = wrapper.model_
    columns = M.model_columns(R.FEATURES, use_sector=False)
    assert list(estimator.feature_name_) == columns

    # 3. Same model as HW4? Compare with the importances HW4 saved for this refit year.
    results = pd.read_csv(args.hw4_dir / "results" / "results.csv")
    saved = results.loc[
        results.section.eq("test_lightgbm_feature_importance")
        & results.metric.astype(str).eq(str(fit_date.year))
    ].set_index("name")["value"]
    saved = saved.astype(float)
    importance = wrapper.importance()
    max_diff = float((importance.reindex(saved.index) - saved).abs().max())
    assert max_diff < 1e-5, f"refit does not reproduce HW4 importances (max diff {max_diff})"

    # 4. Reference predictions from HW4 code, on real eligible stocks on the fit date.
    rows = panel.loc[panel.date.eq(fit_date) & panel.eligible].copy()
    rows["hw4_prediction"] = wrapper.predict(rows)
    rows = rows.sort_values("symbol")
    # The whole cross-section of that date, for the full-day test (blank cell = missing).
    day = rows[["symbol", *columns, "hw4_prediction"]].rename(
        columns=dict(zip(columns, R.FEATURES))
    )
    day.to_csv(HERE / "tests" / f"reference_day_{fit_date.date()}.csv", index=False)
    has_missing = rows[columns].isna().any(axis=1)
    picks = pd.concat(
        [
            rows.loc[~has_missing].iloc[
                :: max(1, (~has_missing).sum() // (REFERENCE_SYMBOLS - 2))
            ],
            rows.loc[has_missing].head(2),  # missing values; head() below kept 1 (ACI)
        ]
    ).head(REFERENCE_SYMBOLS)
    reference = []
    for _, r in picks.iterrows():
        feats = {f: (None if pd.isna(r[f"r_{f}"]) else float(r[f"r_{f}"])) for f in R.FEATURES}
        reference.append(
            {
                "symbol": r.symbol,
                "signal_date": str(r.date.date()),
                "features": feats,
                "hw4_prediction": float(r.hw4_prediction),
            }
        )

    # Contract evidence: ranges and missingness of each input in the training rows.
    stats = {
        f: {
            "train_min": float(train[c].min()),
            "train_max": float(train[c].max()),
            "train_missing_share": float(train[c].isna().mean()),
        }
        for f, c in zip(R.FEATURES, columns)
    }

    model_version = f"hw4-{SERVED_MODEL}-fit-{fit_date.date()}"
    ARTIFACTS.mkdir(exist_ok=True)
    bundle = {
        "estimator": estimator,
        "model_version": model_version,
        "features": list(R.FEATURES),
        "columns": columns,
        "versions": library_versions(),
    }
    path = ARTIFACTS / "model.joblib"
    joblib.dump(bundle, path)

    metadata = {
        "model_version": model_version,
        "source": "Homeworks/HW4/Shashank_Mishra run_hw4.FITTED['lgb_excess'] (final-test comparison)",
        "fit_date": str(fit_date.date()),
        "train_rows": int(len(train)),
        "train_first_date": str(train.date.min().date()),
        "last_label_end": str(train.label_end.max().date()),
        "used_for_hw4_signals": f"{fit_date.date()} to {R.TEST_END.date()}",
        "features_in_order": list(R.FEATURES),
        "model_columns": columns,
        "target": "90-day forward return (next-session close entry, first close on/after "
        "entry + 90 calendar days), winsorized 1%/99% within the signal date, minus "
        "that date's mean. Fraction, e.g. 0.01 = 1 percentage point above the average stock.",
        "inputs": "Each feature is its percentile rank (0, 1] among that date's eligible "
        "top-2000 stocks (pandas rank(pct=True)); null = missing, routed natively by LightGBM.",
        "params": {k: v for k, v in estimator.get_params().items() if k != "n_jobs"},
        "importance_check_max_abs_diff_vs_hw4_results_csv": max_diff,
        "feature_stats": stats,
        "versions": library_versions(),
        "artifact_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    (ARTIFACTS / "model_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (ARTIFACTS / "reference_predictions.json").write_text(
        json.dumps(reference, indent=2) + "\n"
    )

    # The first reference stock doubles as the committed example request.
    example = reference[0]["features"]
    (HERE / "example_request.json").write_text(json.dumps(example, indent=2) + "\n")
    print(
        f"saved {model_version}: {len(train):,} training rows, importance diff {max_diff:.2e}"
    )
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
