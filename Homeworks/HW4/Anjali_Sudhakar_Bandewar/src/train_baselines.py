"""Tune Ridge and HistGradientBoosting baselines on expanding-window folds.

Hyperparameters are chosen on the 2012-2014 folds only. The 2015 development
holdout is scored once, after selection. Nothing from the final test period
(2016-04-01 onward) is used.
"""
import itertools
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted, validate_data

from model_dataset import (
    FEATURES,
    HOLDOUT_YEAR,
    TUNING_YEARS,
    describe_fold,
    load_dataset,
    make_fold,
)

OUTPUT_DIR = Path("Homeworks/HW4/Anjali_Sudhakar_Bandewar/results/models")

RANDOM_STATE = 0
TOP_N = 50

RIDGE_GRID = [{"alpha": a} for a in (0.1, 1.0, 10.0, 100.0, 1000.0)]
HGB_GRID = [
    {"learning_rate": lr, "max_leaf_nodes": leaves, "max_iter": iters}
    for lr, leaves, iters in itertools.product((0.05, 0.1), (15, 31), (100, 300))
]


class QuantileClipper(TransformerMixin, OneToOneFeatureMixin, BaseEstimator):
    """Clip each column to tail quantiles learned on training rows only.

    Adapted from QuantileClipper in Lectures/modeling/course_data.py.
    Missing values are preserved for the imputer or the tree model.
    """

    def __init__(self, lower=0.01, upper=0.99):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        a = validate_data(self, X, reset=True, dtype=float, ensure_all_finite="allow-nan")
        self.lower_bounds_ = np.nanquantile(a, self.lower, axis=0)
        self.upper_bounds_ = np.nanquantile(a, self.upper, axis=0)
        return self

    def transform(self, X):
        check_is_fitted(self, "lower_bounds_")
        a = validate_data(self, X, reset=False, dtype=float, ensure_all_finite="allow-nan")
        return np.clip(a, self.lower_bounds_, self.upper_bounds_)


def ridge_pipeline(alpha):
    return Pipeline([
        ("clip", QuantileClipper()),
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
        ("model", Ridge(alpha=alpha)),
    ])


def hgb_pipeline(learning_rate, max_leaf_nodes, max_iter):
    return Pipeline([
        ("clip", QuantileClipper()),
        # early_stopping must stay off: its internal split is random rows,
        # which would mix dates and overlapping 90-day labels.
        ("model", HistGradientBoostingRegressor(
            learning_rate=learning_rate,
            max_leaf_nodes=max_leaf_nodes,
            max_iter=max_iter,
            min_samples_leaf=1000,
            early_stopping=False,
            random_state=RANDOM_STATE,
        )),
    ])


MODELS = {
    "ridge": (ridge_pipeline, RIDGE_GRID),
    "hgb": (hgb_pipeline, HGB_GRID),
}


def momentum_scores(valid):
    """Momentum-only baseline: rank by momentum_12m, no fitting.

    Missing momentum takes that date's cross-sectional median (same-date data).
    """
    m = valid["momentum_12m"]
    return m.fillna(valid.groupby("feature_date")["momentum_12m"].transform("median")).to_numpy()


def evaluate(valid, scores):
    """Daily cross-sectional metrics, averaged over validation sessions."""
    v = valid[["feature_date", "forward_return", "target"]].copy()
    v["score"] = scores
    by_date = v.groupby("feature_date")

    # Spearman = Pearson correlation of within-date ranks.
    v["score_rank"] = by_date["score"].rank()
    v["ret_rank"] = by_date["forward_return"].rank()
    ic = v.groupby("feature_date")[["score_rank", "ret_rank"]].corr().xs(
        "score_rank", level=1)["ret_rank"]

    v["decile"] = by_date["score"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 10, labels=False))
    dec = v.groupby(["feature_date", "decile"])["forward_return"].mean().unstack()
    spread = dec[9] - dec[0]

    v["date_mean"] = by_date["forward_return"].transform("mean")
    v["score_order"] = by_date["score"].rank(method="first", ascending=False)
    top = v[v["score_order"] <= TOP_N].groupby("feature_date")
    top_excess = top["forward_return"].mean() - top["date_mean"].first()

    return {
        "ic_mean": ic.mean(),
        "ic_std": ic.std(),
        "ic_ir": ic.mean() / ic.std(),
        "ic_pct_positive": (ic > 0).mean(),
        "top_decile_ret": dec[9].mean(),
        "bottom_decile_ret": dec[0].mean(),
        "decile_spread": spread.mean(),
        "top50_excess": top_excess.mean(),
        "mse_rank": mean_squared_error(v["target"], v["score"]),
        "r2_rank": r2_score(v["target"], v["score"]),
        "valid_dates": len(ic),
    }


def fit_and_score(model_name, params, fold):
    builder, _ = MODELS[model_name]
    pipe = builder(**params)
    start = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        # Fitted on this fold's eligible training rows only.
        pipe.fit(fold.train[FEATURES], fold.train["target"])
        scores = pipe.predict(fold.valid[FEATURES])
    result = evaluate(fold.valid, scores)
    result.update(
        model=model_name,
        params=str(params),
        year=fold.year,
        fit_seconds=round(time.perf_counter() - start, 2),
        warnings="; ".join(sorted({str(w.message)[:80] for w in caught})),
    )
    return result


def momentum_result(fold):
    result = evaluate(fold.valid, momentum_scores(fold.valid))
    # A raw momentum value is not on the rank-target scale.
    result.update(model="momentum_12m", params="-", year=fold.year,
                  fit_seconds=0.0, warnings="", mse_rank=np.nan, r2_rank=np.nan)
    return result


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    data, sessions = load_dataset()
    print(f"Loaded {len(data):,} usable labelled rows in {time.perf_counter() - t0:.1f}s")

    tuning_folds = [make_fold(data, sessions, y) for y in TUNING_YEARS]
    holdout_fold = make_fold(data, sessions, HOLDOUT_YEAR)
    folds_table = pd.DataFrame([describe_fold(f) for f in (*tuning_folds, holdout_fold)])
    folds_table.to_csv(OUTPUT_DIR / "folds.csv", index=False)
    print("\nFOLDS")
    print(folds_table.T.to_string(header=False))

    # ---- Tuning on 2012-2014 only ----
    rows = []
    for fold in tuning_folds:
        rows.append(momentum_result(fold))
        for name, (_, grid) in MODELS.items():
            for params in grid:
                rows.append(fit_and_score(name, params, fold))
                print(f"  {fold.year} {name} {params}: IC={rows[-1]['ic_mean']:.4f} "
                      f"({rows[-1]['fit_seconds']}s)")
    tuning = pd.DataFrame(rows)
    tuning.to_csv(OUTPUT_DIR / "tuning_results.csv", index=False)

    summary = (tuning.groupby(["model", "params"])
               .agg(ic_mean=("ic_mean", "mean"), ic_min_year=("ic_mean", "min"),
                    ic_ir=("ic_ir", "mean"), decile_spread=("decile_spread", "mean"),
                    top50_excess=("top50_excess", "mean"),
                    fit_seconds=("fit_seconds", "sum"))
               .reset_index().sort_values("ic_mean", ascending=False))
    summary.to_csv(OUTPUT_DIR / "tuning_summary.csv", index=False)
    print("\nTUNING SUMMARY (mean over 2012-2014 folds)")
    print(summary.to_string(index=False))

    # Selection uses the mean 2012-2014 rank IC only.
    best = {}
    for name, (_, grid) in MODELS.items():
        top = summary[summary["model"] == name].iloc[0]
        best[name] = {str(p): p for p in grid}[top["params"]]
    print("\nSELECTED:", best)

    by_year = tuning[
        (tuning["model"] == "momentum_12m")
        | tuning.apply(lambda r: r["model"] in best and r["params"] == str(best[r["model"]]), axis=1)
    ]

    # ---- 2015 development holdout: scored once with the selected settings ----
    holdout_rows = [momentum_result(holdout_fold)]
    for name, params in best.items():
        holdout_rows.append(fit_and_score(name, params, holdout_fold))
    holdout = pd.DataFrame(holdout_rows)

    report = pd.concat([by_year, holdout], ignore_index=True)
    report.to_csv(OUTPUT_DIR / "selected_by_year.csv", index=False)
    cols = ["model", "year", "ic_mean", "ic_std", "ic_ir", "ic_pct_positive",
            "top_decile_ret", "bottom_decile_ret", "decile_spread", "top50_excess",
            "r2_rank", "fit_seconds"]
    print("\nSELECTED MODELS BY YEAR (2015 = untouched holdout)")
    print(report[cols].sort_values(["model", "year"]).to_string(index=False))

    all_fits = pd.concat([tuning, holdout], ignore_index=True)
    warned = all_fits.loc[all_fits["warnings"] != "", ["model", "params", "year", "warnings"]]
    print(f"\nFits with warnings: {len(warned)}")
    if len(warned):
        print(warned.to_string(index=False))
    print(f"\nTotal runtime: {time.perf_counter() - t0:.1f}s; outputs in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
