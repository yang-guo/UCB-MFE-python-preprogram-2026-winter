"""HW4 stage 2: chronological validation inside the development period.

Inputs: the stage 1 panel (outputs/panel_stage1/). Only eligible rows with
signal dates 2010-01-01 to 2015-12-31 are used; nothing after the 2016-03-31
data cutoff exists in that panel.

Folds (expanding window, one per calendar year 2012-2015):
  * the fold's model is fitted just before the validation year's first signal,
  * training rows: signals from 2010-01-01 whose outcome (label_end) ended
    strictly before that fit date,
  * validation rows: every stock on every date of that year (all stocks on a
    date stay together, so a date is never split between train and validation).

Model: the lecture's Ridge pipeline (course_data.make_pipeline) on the
lecture's FEATURES. Alpha is chosen by mean fold RMSE from [1, 100, 1000],
and compared with a training-mean baseline (DummyRegressor).

Reused code: course_data.date_folds, make_pipeline, prediction_metrics and
FEATURES are imported unchanged from Lectures/modeling/course_data.py.

Run from the repository root:
    .venv/Scripts/python -B Homeworks/HW4/Ryan_Beauchamp/stage2_validation.py

Credits: date_folds, make_pipeline, prediction_metrics and FEATURES come from
Lectures/modeling/course_data.py (imported, unmodified). Written with AI
assistance (Claude Code) and reviewed by the author.
"""

import sys
import time
import warnings
from pathlib import Path

sys.dont_write_bytecode = True  # keep Lectures/ free of __pycache__

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "Lectures" / "modeling"))
import course_data as cd  # noqa: E402

PANEL_DIR = HERE / "outputs" / "panel_stage1"
OUTPUT_DIR = HERE / "outputs"
DEV_START, DEV_END = "2010-01-01", "2015-12-31"
DATA_END = pd.Timestamp("2016-03-31")
VALIDATION_YEARS = (2012, 2013, 2014, 2015)
ALPHAS = [1, 100, 1000]


def load_development_rows():
    """Eligible development rows only (dated membership + 63 trailing returns)."""
    columns = ["date", "symbol", "eligible", "target", "entry_date", "label_end"] + cd.FEATURES
    frame = pd.read_parquet(PANEL_DIR, columns=columns,
                            filters=[("date", ">=", pd.Timestamp(DEV_START)),
                                     ("date", "<=", pd.Timestamp(DEV_END)),
                                     ("eligible", "==", True)])
    frame = frame.sort_values(["date", "symbol"]).reset_index(drop=True)
    assert frame.date.max() <= pd.Timestamp(DEV_END)
    assert frame.label_end.max() <= DATA_END, "an outcome ends after the data cutoff"
    return frame


def describe_folds(frame, splits):
    """One row per fold, plus the look-ahead asserts."""
    rows = []
    for year, (tr, va) in zip(VALIDATION_YEARS, splits):
        train, valid = frame.iloc[tr], frame.iloc[va]
        fit_date = valid.date.min()  # model is fitted before this signal
        # No training outcome may end on or after the first validation signal.
        assert train.label_end.max() < fit_date
        assert train.date.max() < fit_date
        assert valid.date.dt.year.eq(year).all()
        # All stocks on a validation date are validated together.
        assert not set(train.date).intersection(valid.date)
        available = frame.loc[frame.date.isin(valid.date.unique()) & frame.target.notna()]
        assert len(available) == len(valid), "a validation date lost some stocks"
        # Training dates where some stocks were left out because their exit was delayed.
        per_date = frame.loc[frame.date.isin(train.date.unique()) & frame.target.notna()].groupby("date").size()
        partial_dates = int((train.groupby("date").size() != per_date).sum())
        rows.append({
            "validation_year": year,
            "fit_date": fit_date.date(),
            "train_rows": len(train),
            "train_dates": train.date.nunique(),
            "first_train_signal": train.date.min().date(),
            "last_train_signal": train.date.max().date(),
            "last_train_label_end": train.label_end.max().date(),
            "partially_matured_train_dates": partial_dates,
            "validation_first_signal": valid.date.min().date(),
            "validation_last_signal": valid.date.max().date(),
            "validation_last_label_end": valid.label_end.max().date(),
            "validation_rows": len(valid),
            "validation_dates": valid.date.nunique(),
        })
    return pd.DataFrame(rows)


def fit_and_score(model, frame, tr, va):
    train, valid = frame.iloc[tr], frame.iloc[va]
    fitted = clone(model).fit(train[cd.FEATURES], train.target)
    with warnings.catch_warnings():
        # Constant predictions make the daily rank IC undefined; that NaN is expected.
        warnings.simplefilter("ignore", RuntimeWarning)
        return cd.prediction_metrics(valid, fitted.predict(valid[cd.FEATURES]))


def main():
    start = time.perf_counter()
    timings = {}

    t = time.perf_counter()
    frame = load_development_rows()
    timings["load development rows"] = time.perf_counter() - t
    print(f"Development rows (eligible, {DEV_START} to {DEV_END}): {len(frame):,}; "
          f"with target: {frame.target.notna().sum():,}")

    t = time.perf_counter()
    splits = cd.date_folds(frame, validation_years=VALIDATION_YEARS, train_start=DEV_START)
    folds = describe_folds(frame, splits)
    timings["build folds + asserts"] = time.perf_counter() - t
    print("\nFolds (all look-ahead asserts passed):")
    print(folds.set_index("validation_year").T.to_string())

    # --- Alpha search: mean RMSE across the four folds --------------------------
    t = time.perf_counter()
    results = []
    for alpha in ALPHAS:
        for year, (tr, va) in zip(VALIDATION_YEARS, splits):
            m = fit_and_score(cd.make_pipeline(Ridge(alpha=alpha)), frame, tr, va)
            results.append({"model": f"Ridge(alpha={alpha})", "alpha": alpha, "validation_year": year, **m})
    timings["Ridge fits (3 alphas x 4 folds)"] = time.perf_counter() - t

    t = time.perf_counter()
    for year, (tr, va) in zip(VALIDATION_YEARS, splits):
        m = fit_and_score(DummyRegressor(strategy="mean"), frame, tr, va)
        results.append({"model": "Training mean", "alpha": np.nan, "validation_year": year, **m})
    timings["training-mean baseline"] = time.perf_counter() - t

    metrics = pd.DataFrame(results)
    metrics["IC_dates"] = metrics.IC_dates.astype(int)
    search = (metrics.loc[metrics.alpha.notna()].groupby("alpha")
              [["RMSE", "MAE", "R2", "mean_daily_rank_IC"]].mean())
    best_alpha = int(search.RMSE.idxmin())
    print("\nAlpha search (mean over folds):")
    print(search.to_string(float_format=lambda x: f"{x:.5f}"))
    print(f"Chosen alpha by mean fold RMSE: {best_alpha}")

    chosen = metrics.loc[metrics.alpha.eq(best_alpha) | metrics.model.eq("Training mean")]
    table = chosen.pivot(index="validation_year", columns="model",
                         values=["RMSE", "MAE", "R2", "mean_daily_rank_IC"])
    table.loc["mean"] = table.mean()
    print("\nPer-fold metrics: chosen Ridge vs training mean")
    print(table.to_string(float_format=lambda x: f"{x:.5f}"))

    folds.to_csv(OUTPUT_DIR / "stage2_folds.csv", index=False)
    metrics.to_csv(OUTPUT_DIR / "stage2_metrics_all.csv", index=False)
    search.to_csv(OUTPUT_DIR / "stage2_alpha_search.csv")
    table.to_csv(OUTPUT_DIR / "stage2_metrics_chosen.csv")

    timings["total wall clock"] = time.perf_counter() - start
    print("\nTime per step (seconds):")
    print(pd.Series(timings).round(1).to_string())


if __name__ == "__main__":
    main()
