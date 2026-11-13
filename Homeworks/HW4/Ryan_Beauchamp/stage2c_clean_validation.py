"""HW4 stage 2c: rerun the stage 2 folds on causally clean rows.

Changes from stage 2:
  * Rows must pass the `clean` flag (measures from stage2b_clean_flag.py):
      - largest |daily log move| over the trailing 252 sessions <= ln 5, and
      - supplied close >= $1.
    Thresholds were chosen from exclusion counts only (stage2b output), never
    from validation or return results. The flag applies to training AND
    validation rows, because it will also gate new entries in the backtest.
  * Each fold's training targets are clipped at that fold's own 1st/99th
    percentiles (learned from training rows only).
  * Validation metrics are reported on raw targets and on targets clipped at
    the same training bounds. Rank IC is the headline metric.
  * Top-5% bucket: on each date, the mean actual return of the top 5% of
    predictions vs all other rows, averaged over dates.
  * Extra model: Ridge trained on each date's cross-sectional percentile rank
    of the target (centred at 0). Its predictions are on a rank scale, so only
    rank-based metrics are comparable with the return models.

Alpha is chosen by mean fold RMSE on clipped validation targets (raw RMSE can
still be dominated by a single surviving glitch); the raw-RMSE choice is
printed alongside. The rank model's alpha is chosen by RMSE against the
validation date-rank target.

Run from the repository root:
    .venv/Scripts/python -B Homeworks/HW4/Ryan_Beauchamp/stage2c_clean_validation.py

Credits: date_folds, make_pipeline and FEATURES come from
Lectures/modeling/course_data.py (imported, unmodified). Written with AI
assistance (Claude Code) and reviewed by the author.
"""

import sys
import time
import warnings
from pathlib import Path

sys.dont_write_bytecode = True

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import stage2_validation as s2  # noqa: E402  (also puts course_data on the path)
import course_data as cd  # noqa: E402

MEASURES = HERE / "outputs" / "clean_measures" / "measures.parquet"
OUTPUT_DIR = HERE / "outputs"
MAX_LOG_MOVE = np.log(5)
MIN_CLOSE = 1.0
ALPHAS = [1, 100, 1000]
TOP_SHARE = 0.05


def load_clean_rows():
    frame = s2.load_development_rows()
    measures = pd.read_parquet(MEASURES, columns=["symbol", "date", "close", "max_abs_log_move_252"])
    frame = frame.merge(measures, on=["symbol", "date"], how="left", validate="one_to_one")
    broken = frame.max_abs_log_move_252.gt(MAX_LOG_MOVE) | frame.close.lt(MIN_CLOSE)
    print(f"Eligible development rows: {len(frame):,}; failing the clean flag: {broken.sum():,} "
          f"({broken.mean():.2%}); stocks affected: {frame.loc[broken, 'symbol'].nunique()}")
    return frame.loc[~broken].reset_index(drop=True)


def date_rank(frame, values):
    """Cross-sectional percentile rank per date, centred at 0 (range about -0.5..0.5)."""
    return values.groupby(frame.date.to_numpy()).rank(pct=True).to_numpy() - 0.5


def score(valid, prediction, low, high, rank_model=False):
    """Validation metrics for one fold."""
    out = valid[["date", "target"]].copy()
    out["prediction"] = prediction
    out["clipped"] = out.target.clip(low, high)
    result = {}
    if not rank_model:
        for name, column in [("raw", "target"), ("clipped", "clipped")]:
            error = out.prediction - out[column]
            result[f"RMSE_{name}"] = float(np.sqrt((error ** 2).mean()))
            result[f"MAE_{name}"] = float(error.abs().mean())
    by_date = out.groupby("date")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ic = by_date.apply(lambda x: x.prediction.corr(x.target, method="spearman")
                           if x.prediction.nunique() > 1 else np.nan, include_groups=False)
    result["rank_IC"] = float(ic.mean())
    result["IC_t_stat"] = float(ic.mean() / ic.std() * np.sqrt(ic.notna().sum())) if ic.notna().sum() > 1 else np.nan
    if out.prediction.nunique() > 1:
        cut = by_date.prediction.transform(lambda p: p.quantile(1 - TOP_SHARE))
        out["top"] = out.prediction.ge(cut)
        for name, column in [("raw", "target"), ("clipped", "clipped")]:
            means = out.groupby(["date", "top"])[column].mean().unstack()
            result[f"top5_mean_{name}"] = float(means[True].mean())
            result[f"rest_mean_{name}"] = float(means[False].mean())
            result[f"top5_minus_rest_{name}"] = float((means[True] - means[False]).mean())
    return result


def run_fold(frame, tr, va, model, kind):
    train, valid = frame.iloc[tr], frame.iloc[va]
    low, high = train.target.quantile([0.01, 0.99])  # learned on training rows only
    if kind == "rank":
        y = date_rank(train, train.target)
    else:
        y = train.target.clip(low, high)
    fitted = clone(model).fit(train[cd.FEATURES], y)
    prediction = fitted.predict(valid[cd.FEATURES])
    result = score(valid, prediction, low, high, rank_model=(kind == "rank"))
    result["clip_low"], result["clip_high"] = float(low), float(high)
    if kind == "rank":
        rank_error = prediction - date_rank(valid, valid.target)
        result["RMSE_vs_date_rank"] = float(np.sqrt((rank_error ** 2).mean()))
    return result


def main():
    start = time.perf_counter()
    timings = {}

    t = time.perf_counter()
    frame = load_clean_rows()
    splits = cd.date_folds(frame, validation_years=s2.VALIDATION_YEARS, train_start=s2.DEV_START)
    folds = s2.describe_folds(frame, splits)
    timings["load + folds + asserts"] = time.perf_counter() - t
    print("\nFolds on clean rows (all look-ahead asserts passed):")
    print(folds.set_index("validation_year").T.to_string())

    runs = [("Training mean (clipped y)", np.nan, DummyRegressor(strategy="mean"), "clipped")]
    runs += [(f"Ridge clipped y (alpha={a})", a, cd.make_pipeline(Ridge(alpha=a)), "clipped") for a in ALPHAS]
    runs += [(f"Ridge date-rank y (alpha={a})", a, cd.make_pipeline(Ridge(alpha=a)), "rank") for a in ALPHAS]

    t = time.perf_counter()
    rows = []
    for name, alpha, model, kind in runs:
        for year, (tr, va) in zip(s2.VALIDATION_YEARS, splits):
            rows.append({"model": name, "kind": kind, "alpha": alpha, "validation_year": year,
                         **run_fold(frame, tr, va, model, kind)})
        print(f"  fitted {name}")
    timings["fits (1 + 2 x 3 alphas, 4 folds each)"] = time.perf_counter() - t
    metrics = pd.DataFrame(rows)

    ridge = metrics.loc[metrics.kind.eq("clipped") & metrics.alpha.notna()]
    search = ridge.groupby("alpha")[["RMSE_clipped", "RMSE_raw", "MAE_clipped", "rank_IC"]].mean()
    best = int(search.RMSE_clipped.idxmin())
    rank_search = (metrics.loc[metrics.kind.eq("rank")].groupby("alpha")
                   [["RMSE_vs_date_rank", "rank_IC"]].mean())
    best_rank = int(rank_search.RMSE_vs_date_rank.idxmin())
    print("\nAlpha search, Ridge on clipped targets (mean over folds):")
    print(search.to_string(float_format=lambda x: f"{x:.5f}"))
    print(f"Chosen by clipped RMSE: alpha={best}; by raw RMSE it would be alpha={int(search.RMSE_raw.idxmin())}")
    print("\nAlpha search, Ridge on date-rank targets (mean over folds):")
    print(rank_search.to_string(float_format=lambda x: f"{x:.5f}"))
    print(f"Chosen by date-rank RMSE: alpha={best_rank}")

    chosen = metrics.loc[metrics.model.isin(["Training mean (clipped y)",
                                             f"Ridge clipped y (alpha={best})",
                                             f"Ridge date-rank y (alpha={best_rank})"])]
    columns = ["RMSE_raw", "RMSE_clipped", "MAE_raw", "MAE_clipped", "rank_IC", "IC_t_stat",
               "top5_mean_raw", "rest_mean_raw", "top5_minus_rest_raw", "top5_minus_rest_clipped"]
    for name, group in chosen.groupby("model", sort=False):
        table = group.set_index("validation_year")[columns]
        table.loc["mean"] = table.mean()
        print(f"\n{name}")
        print(table.to_string(float_format=lambda x: f"{x:.4f}"))

    clip_bounds = metrics.loc[metrics.model.eq("Training mean (clipped y)"),
                              ["validation_year", "clip_low", "clip_high"]]
    print("\nTraining-target clip bounds per fold:")
    print(clip_bounds.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))

    folds.to_csv(OUTPUT_DIR / "stage2c_folds.csv", index=False)
    metrics.to_csv(OUTPUT_DIR / "stage2c_metrics_all.csv", index=False)
    search.to_csv(OUTPUT_DIR / "stage2c_alpha_search_clipped.csv")
    rank_search.to_csv(OUTPUT_DIR / "stage2c_alpha_search_rank.csv")

    timings["total wall clock"] = time.perf_counter() - start
    print("\nTime per step (seconds):")
    print(pd.Series(timings).round(1).to_string())


if __name__ == "__main__":
    main()
