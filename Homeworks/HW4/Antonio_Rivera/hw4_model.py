"""HW4 module 3: chronological validation and model choice (development only).

Reused code (cited): Lectures/modeling/course_data.py
    date_folds        expanding folds; drops training rows whose label ends
                      on/after the first validation signal (purge) and asserts it
    make_pipeline     clip -> impute -> scale (+ sector one-hot) -> model;
                      all preprocessing is fitted inside each fold on train rows only
    prediction_metrics  RMSE, MAE, R2, mean daily Spearman rank IC

Rules fixed BEFORE seeing results (see README):
    grid       4 feature sets x 2 models = 8 candidates (F4 added in module 4)
    folds      validation years 2012, 2013, 2014, 2015; training from 2010-01-01
    metric     mean daily rank IC across the 4 folds
    selection  best mean IC; any simpler candidate within 1 standard error of it wins
    retrain    (test period) refit at each calendar-quarter start, expanding window
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

import hw4_data as hd
import hw4_features as hf

cd = hd.cd

DEV_START, DEV_END = "2010-01-01", "2015-12-31"
VALIDATION_YEARS = (2012, 2013, 2014, 2015)

FEATURE_SETS = {
    "F1_baseline": cd.NUMERIC_FEATURES,   # 12 course features
    "F2_plus_AB": hf.RAW_NUMERIC,         # + gross_profitability, asset_growth
    "F3_ranks": hf.RANK_FEATURES,         # same-date percentile ranks (D2)
    # Added 2026-11-10 (module 4): grid grows 6 -> 8; selection rule unchanged,
    # F4 ranks as the most complex set.
    "F4_plus_4ab": hf.RAW_NUMERIC + hf.MODEL_FINANCIAL_4A + hf.PRICE_4B,
}  # every set also gets one-hot sector inside make_pipeline


def make_model(name):
    if name == "ridge":
        return Ridge(alpha=100)
    if name == "hgb":
        # early_stopping=False: fixed 200 iterations as pre-registered; the
        # automatic early stop would carve a RANDOM (non-chronological) split.
        return HistGradientBoostingRegressor(max_depth=3, max_iter=200, learning_rate=0.05,
                                             early_stopping=False, random_state=0)
    raise ValueError(name)


# Simplest first: Ridge before boosting; fewer/rawer features before more.
SIMPLICITY_ORDER = [(m, f) for m in ("ridge", "hgb") for f in FEATURE_SETS]


def add_target_rank(frame):
    """Training target = per-date percentile rank of the 90-day return, in (0, 1].

    Changed after the top2000 audit (not after seeing model results): the source
    contains absurd labels (AAM +14,270,000% in 90 days) that would dominate any
    squared-error fit. Ranks keep only the ordering, which is what the strategy
    uses. Each date's ranks use only that date's targets, and folds keep whole
    dates together, so no information crosses a train/validation boundary.
    """
    out = frame.copy()
    out["target_rank"] = out.groupby("date").target.rank(pct=True)
    return out


def dev_frame(panel):
    """Eligible rows with signal dates inside development, plus target_rank. Positional index."""
    keep = panel.eligible & panel.date.between(DEV_START, DEV_END)
    return add_target_rank(panel.loc[keep].reset_index(drop=True))


def top_quintile_excess(valid, pred):
    """Mean over dates of (avg target of top 20% by prediction - avg target of all).

    Targets are winsorized per date at 1%/99% so a bad print cannot dominate.
    """
    v = valid[["date", "target"]].assign(pred=np.asarray(pred))
    g = v.groupby("date").target
    v["target"] = v.target.clip(g.transform("quantile", 0.01), g.transform("quantile", 0.99))

    def one_date(g):
        k = max(1, int(round(0.2 * len(g))))
        return g.nlargest(k, "pred").target.mean() - g.target.mean()

    return v.groupby("date").apply(one_date, include_groups=False).mean()


def run_grid(frame, years=VALIDATION_YEARS):
    """Fit every candidate on every fold. Returns one row per (model, features, fold)."""
    folds = cd.date_folds(frame, validation_years=years, train_start=DEV_START)
    rows = []
    for model_name, fs_name in SIMPLICITY_ORDER:
        for year, (tr, va) in zip(years, folds):
            pipe = cd.make_pipeline(make_model(model_name), FEATURE_SETS[fs_name])
            train, valid = frame.iloc[tr], frame.iloc[va]
            pipe.fit(train, train.target_rank)        # rank target (see add_target_rank)
            pred = pipe.predict(valid)
            m = cd.prediction_metrics(valid, pred)
            rows.append({"model": model_name, "features": fs_name, "fold": year,
                         "n_train": len(tr), "n_valid": len(va),
                         "train_last_label_end": train.label_end.max().date(),
                         "valid_first_signal": valid.date.min().date(),
                         "rank_IC": m.mean_daily_rank_IC, "IC_dates": int(m.IC_dates),
                         "top20_excess": top_quintile_excess(valid, pred)})
    return pd.DataFrame(rows)


def summarize(results):
    """Mean and standard error of rank IC across folds, in simplicity order."""
    g = results.groupby(["model", "features"], sort=False)
    s = g.agg(mean_IC=("rank_IC", "mean"), sd_IC=("rank_IC", "std"), folds=("rank_IC", "size"),
              min_IC=("rank_IC", "min"), mean_top20_excess=("top20_excess", "mean"))
    s["se_IC"] = s.sd_IC / np.sqrt(s.folds)
    return s.reset_index()


def select(summary):
    """Pre-registered rule: simplest candidate whose mean IC >= best mean IC - best SE."""
    best = summary.loc[summary.mean_IC.idxmax()]
    bar = best.mean_IC - best.se_IC
    order = {k: i for i, k in enumerate(SIMPLICITY_ORDER)}
    ok = summary.loc[summary.mean_IC >= bar].copy()
    ok["rank"] = [order[(m, f)] for m, f in zip(ok.model, ok.features)]
    pick = ok.sort_values("rank").iloc[0]
    return {"model": pick.model, "features": pick.features, "mean_IC": pick.mean_IC,
            "best": (best.model, best.features), "bar": bar}


if __name__ == "__main__":
    # Usage: hw4_model.py [class|dev]   (reads outputs/panel_<name>.parquet from hw4_build_cache.py)
    import sys
    import hw4_build_cache as hc
    name = sys.argv[1] if len(sys.argv) > 1 else "class"
    assert hc.TARGETS[name][1] == DEV_END, "validation only runs on development caches"
    feats = sorted({c for fs in FEATURE_SETS.values() for c in fs})
    panel = pd.read_parquet(hc.cache_path(name), columns=[
        "date", "symbol", "eligible", "target", "label_end", "sector"] + feats)
    frame = dev_frame(panel)
    del panel
    results = run_grid(frame)
    out = hc.OUT / f"validation_{name}.csv"
    results.to_csv(out, index=False)
    pd.set_option("display.width", 160)
    print(results.round(4).to_string(index=False))
    summary = summarize(results)
    print(summary.round(4).to_string(index=False))
    print("SELECTED:", select(summary))
