"""Models, walk-forward fitting, and prediction diagnostics for HW4.

Every fit at date F uses only rows with
  date >= TRAIN_START  and  label_end < F
so each training outcome (the 90-day exit close) was observed before the fit.
Features are same-date percentile ranks, so the only learned preprocessing is
the Ridge imputer (fit on training rows inside the sklearn pipeline).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from features import FEATURES

TRAIN_START = pd.Timestamp("2010-01-01")
SECTORS = ["Basic Materials", "Communication Services", "Consumer Cyclical",
           "Consumer Defensive", "Energy", "Financial Services", "Healthcare",
           "Industrials", "Real Estate", "Technology", "Utilities", "Unknown"]


def design(panel: pd.DataFrame) -> pd.DataFrame:
    X = panel[FEATURES].copy()
    X["sector"] = pd.Categorical(panel.sector, categories=SECTORS)
    return X


def make_model(kind: str):
    if kind == "hgb":
        # Gradient-boosted trees: capture non-linear effects and interactions
        # (e.g. momentum behaving differently for small, volatile stocks), handle
        # missing values natively, and are robust to the noisy target. Heavy
        # regularisation because 90-day returns are mostly noise.
        return HistGradientBoostingRegressor(
            learning_rate=0.05, max_iter=200, max_leaf_nodes=15, min_samples_leaf=2000,
            l2_regularization=1.0, categorical_features="from_dtype", random_state=0)
    if kind == "hgb_small":
        # Shallower, shorter boosting run: fewer chances to fit noise.
        return HistGradientBoostingRegressor(
            learning_rate=0.05, max_iter=100, max_leaf_nodes=7, min_samples_leaf=5000,
            l2_regularization=1.0, categorical_features="from_dtype", random_state=0)
    if kind == "ridge":
        pre = ColumnTransformer([
            ("num", SimpleImputer(strategy="constant", fill_value=0.5, add_indicator=True), FEATURES),
            ("sec", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["sector"]),
        ])
        return Pipeline([("pre", pre), ("ridge", Ridge(alpha=10.0))])
    raise ValueError(kind)


def training_mask(panel: pd.DataFrame, fit_date) -> pd.Series:
    fit_date = pd.Timestamp(fit_date)
    return (panel.date >= TRAIN_START) & (panel.label_end < fit_date) & panel.target.notna()


def fit(panel: pd.DataFrame, fit_date, kind: str):
    rows = panel[training_mask(panel, fit_date)]
    assert rows.label_end.max() < pd.Timestamp(fit_date)
    model = make_model(kind)
    X = design(rows)
    if kind == "ridge":
        X["sector"] = X.sector.astype(str)
    # Re-rank returns among the training rows only, so a row's target never
    # depends on a same-date peer whose exit is after the fit date.
    y = rows.groupby("date").ret_90.rank(pct=True)
    model.fit(X, y)
    return model, len(rows)


def predict(model, panel: pd.DataFrame, kind: str) -> np.ndarray:
    X = design(panel)
    if kind == "ridge":
        X["sector"] = X.sector.astype(str)
    return model.predict(X)


def walk_forward(panel: pd.DataFrame, fit_dates, end, kind: str) -> pd.DataFrame:
    """Predict rows dated in [fit_i, fit_{i+1}) with the model fitted at fit_i."""
    fit_dates = [pd.Timestamp(d) for d in fit_dates]
    bounds = fit_dates[1:] + [pd.Timestamp(end) + pd.Timedelta(days=1)]
    out, log = [], []
    for start, stop in zip(fit_dates, bounds):
        block = panel[(panel.date >= start) & (panel.date < stop)]
        if block.empty:
            continue
        if kind == "momentum":  # no fitting: classic 12-1 momentum rank
            score, n = block.mom_12_1.fillna(0.5).to_numpy(), 0
        elif kind == "size":    # no fitting: biggest companies first (index-like baseline)
            score, n = block.log_mcap.fillna(0.0).to_numpy(), 0
        else:
            model, n = fit(panel, start, kind)
            score = predict(model, block, kind)
        out.append(block[["date", "symbol", "sector", "dollar_vol_raw", "universe_rank",
                          "ret_90", "target", "label_end"]].assign(score=score))
        log.append(dict(fit_date=start, predict_until=stop, train_rows=n,
                        predicted_rows=len(block)))
    return pd.concat(out, ignore_index=True), pd.DataFrame(log)


def prediction_report(pred: pd.DataFrame, top_n: int = 30, max_rank=None) -> pd.DataFrame:
    """Per-year ranking quality on completed labels.

    rank_IC: mean same-date Spearman correlation of score and 90-day return.
    top_minus_all: mean 90-day return of the top ``top_n`` scores minus the
    equal-weight average of all eligible stocks that date.
    """
    p = pred[pred.ret_90.notna()].copy()
    if max_rank is not None:
        p = p[p.universe_rank <= max_rank]

    def by_date(g):
        top = g.nlargest(top_n, "score")
        return pd.Series({"ic": g.score.corr(g.ret_90, method="spearman"),
                          "top": top.ret_90.mean(), "all": g.ret_90.mean(),
                          "top_decile": g[g.score >= g.score.quantile(0.9)].ret_90.mean(),
                          "bottom_decile": g[g.score <= g.score.quantile(0.1)].ret_90.mean()})

    d = p.groupby("date").apply(by_date, include_groups=False)
    d["year"] = d.index.year
    rep = d.groupby("year").agg(rank_IC=("ic", "mean"), IC_std=("ic", "std"),
                                top_mean=("top", "mean"), all_mean=("all", "mean"),
                                top_decile=("top_decile", "mean"),
                                bottom_decile=("bottom_decile", "mean"), dates=("ic", "size"))
    rep["top_minus_all"] = rep.top_mean - rep.all_mean
    rep["decile_spread"] = rep.top_decile - rep.bottom_decile
    total = d.agg({"ic": ["mean", "std", "size"]}).ic
    rep.loc["all"] = [total["mean"], total["std"], d.top.mean(), d["all"].mean(),
                      d.top_decile.mean(), d.bottom_decile.mean(), total["size"],
                      d.top.mean() - d["all"].mean(), d.top_decile.mean() - d.bottom_decile.mean()]
    return rep
