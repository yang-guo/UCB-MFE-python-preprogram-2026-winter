"""Models, training targets, and walk-forward fitting for HW4.

The walk-forward rule (train only on rows whose label ended strictly before
the fit date, refit on a fixed schedule, audit each fit) follows
Lectures/Lecture 8/2_research_workflow.ipynb. Pipelines follow the sklearn
pattern of Lectures/modeling/course_data.make_pipeline.
"""

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from hw4_data import FEATURES, NUMERIC_FEATURES

TRAIN_START = pd.Timestamp("2010-01-01")


def make_model(kind, alpha=100.0):
    """Ridge (linear baseline) or histogram gradient boosting (nonlinear)."""
    sector = ("sector", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["sector"])
    if kind == "ridge":
        numeric = Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
                            ("scale", StandardScaler())])
        return Pipeline([("preprocess", ColumnTransformer([("numeric", numeric, NUMERIC_FEATURES), sector])),
                         ("model", Ridge(alpha=alpha))])
    if kind == "hgb":
        # Missing values are handled natively; early stopping is off because its random
        # holdout would mix rows from the same dates and overlapping horizons.
        return Pipeline([("preprocess", ColumnTransformer([("numeric", "passthrough", NUMERIC_FEATURES), sector])),
                         ("model", HistGradientBoostingRegressor(
                             learning_rate=0.05, max_iter=300, max_leaf_nodes=15, min_samples_leaf=500,
                             l2_regularization=1.0, early_stopping=False, random_state=0))])
    raise ValueError(f"Unknown model kind {kind!r}")


def training_target(train, target="rank", lower=0.01, upper=0.99):
    """Transform the realized 90-day return before fitting.

    "rank": percentile of the 90-day return among the same date's members.
    "excess": return minus the same-date member mean, winsorized at quantiles
    estimated on the training rows only.
    """
    if target == "rank":
        return train.groupby("date").target.rank(pct=True)
    if target == "excess":
        excess = train.target - train.groupby("date").target.transform("mean")
        return excess.clip(excess.quantile(lower), excess.quantile(upper))
    raise ValueError(f"Unknown target {target!r}")


def fit_dates(signal_dates, start, end):
    """Fit at the start, then at the first signal date of each later calendar quarter."""
    signal_dates = pd.DatetimeIndex(signal_dates)
    scored = signal_dates[(signal_dates >= start) & (signal_dates <= end)]
    first_in_quarter = scored.to_series().groupby(scored.to_period("Q")).min()
    first_entry_quarter = (pd.Timestamp(start) + pd.Timedelta(days=1)).to_period("Q")
    refits = first_in_quarter[first_in_quarter.index > first_entry_quarter]
    return pd.DatetimeIndex([pd.Timestamp(start)]).append(pd.DatetimeIndex(refits.to_numpy())).unique()


def walk_forward(panel, model, start, end, target="rank", fit_schedule=None):
    """Score signals in [start, end] with copies of `model` refitted on matured labels only."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    schedule = fit_schedule if fit_schedule is not None else fit_dates(panel.date.unique(), start, end)
    parts, audit, models = [], [], []
    for k, fit_date in enumerate(schedule):
        next_fit = schedule[k + 1] if k + 1 < len(schedule) else end + pd.Timedelta(days=1)
        train = panel.loc[panel.date.ge(TRAIN_START) & panel.date.lt(fit_date)
                          & panel.label_end.lt(fit_date) & panel.target.notna()]
        score = panel.loc[panel.date.ge(fit_date) & panel.date.lt(next_fit) & panel.date.le(end)]
        fitted = clone(model).fit(train[FEATURES], training_target(train, target))
        models.append(fitted)
        predicted = score[["date", "symbol"]].copy()
        predicted["score"] = fitted.predict(score[FEATURES])
        predicted["fit_date"] = fit_date
        parts.append(predicted)
        assert train.label_end.max() < fit_date <= score.date.min()
        audit.append({"fit_date": fit_date, "train_rows": len(train),
                      "first_train_signal": train.date.min(), "last_train_signal": train.date.max(),
                      "last_train_label_end": train.label_end.max(),
                      "first_scored_signal": score.date.min(), "last_scored_signal": score.date.max(),
                      "scored_rows": len(score)})
    scores = pd.concat(parts, ignore_index=True)
    assert not scores.duplicated(["date", "symbol"]).any()
    return scores, pd.DataFrame(audit), models


def rank_ic(frame, score="score", target="target"):
    """Spearman correlation of score and realized target on each signal date."""
    rows = frame.loc[frame[target].notna()]
    return rows.groupby("date").apply(lambda g: g[score].corr(g[target], method="spearman"),
                                      include_groups=False)


def prediction_report(frame, score="score", top_n=25, max_rank=None):
    """Rank IC and the average realized excess return of the top-scored names."""
    rows = frame.loc[frame.target.notna()].copy()
    if max_rank is not None:
        rows = rows.loc[rows.universe_rank.le(max_rank)]
    rows["excess"] = rows.target - rows.groupby("date").target.transform("mean")
    ic = rank_ic(rows, score)
    top = rows.sort_values(score, ascending=False).groupby("date").head(top_n)
    weekly_top = top.groupby("date").excess.mean()
    return pd.Series({"mean_rank_IC": ic.mean(), "naive_IC_t_stat": ic.mean() / ic.std() * np.sqrt(len(ic)),
                      "share_IC_positive": ic.gt(0).mean(), f"top{top_n}_mean_excess_90d": weekly_top.mean(),
                      f"top{top_n}_median_excess_90d": top.excess.median(), "signal_dates": len(ic)})
