"""
Part 3: Candidate models, chronological validation, and rules for retraining

Workflow (same pattern as Lecture 7 notebook 3 and Lecture 8 notebook 2):
- ``candidates``: sklearn pipelines built with the course's ``make_pipeline``
- ``date_folds``: expanding-window folds inside 2010-2015. Each fold trains only
on rows whose 90-day outcome finished before the validation year's first signal.
- ``walk_forward``: rules for retraining. Refit at chosen dates on every outcome
completed before that date, then score the signals until the next refit.

Reused code: 
- ``date_folds``
- ``make_pipeline``
- ``prediction_metrics`` come from ``Lectures/modeling/course_data.py``.
the candidate/refit loops follow the lecture notebooks. 
My additions: rank-feature and excess return variants, the train-only target clipping, 
and top-N portfolio metrics.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from data import experiment
from features import NUMERIC_FEATURES, RANK_FEATURES
from course_data import date_folds, make_pipeline, prediction_metrics

# first training signal 2010-01-01
TRAIN_START = experiment["development_start"]
# 4 expanding-window folds
VALIDATION_YEARS = (2012, 2013, 2014, 2015)
# stocks bought per month in the backtest
TOP_N = 30

# screen for investibility
# raw close >= $5 to avoid penny stocks
MIN_PRICE = 5.0
# good tradeable size with median $ traded per day over 63 sessions >= $1M   
MIN_DOLLAR_VOLUME = 1000000
# cap 90-day returns at +200% to avoid price error 
EVAL_RETURN_CAP = 2.0


def boosted_trees():
    """Small, heavily regularized trees: same settings as the Lecture 7 example."""
    return HistGradientBoostingRegressor(max_iter=60, max_leaf_nodes=7, min_samples_leaf=100,
                                         l2_regularization=10, early_stopping=False, random_state=42)

# Five signals.
# +1: higher is better.
# -1: lower is better.
FACTOR_SIGNS = {
    # profitable firms (Novy-Marx 2013; Fama-French RMW)
    "rank_operating_roa": +1,
    # firms that generate cash
    "rank_fcf_to_assets": +1,
    # close to the 1-year high (George & Hwang 2004)
    "rank_drawdown_252": +1,
    # 12-month momentum, skipping the last month (Jegadeesh & Titman 1993)
    "rank_momentum_252_21": +1,
    # low volatility (Ang et al. 2006; Frazzini & Pedersen 2014)
    "rank_volatility_63": -1,
}
FACTOR_FEATURES = list(FACTOR_SIGNS)


class FactorScore(RegressorMixin, BaseEstimator):
    """
    Score each stock by averaging its 5 factor ranks.

    How the score works:
    - Each rank is between 0 (worst that month) and 1 (best that month).
    - For volatility, lower is better, so its rank is flipped to 1 - rank.
    - The score is the average of the 5 ranks. A missing rank is left out of
      the average. A higher score means a more profitable stock.
    """
    def fit(self, X, y=None):
        # nothing to learn but kept so the code can call .fit().
        self.is_fitted_ = True
        return self

    def predict(self, X):
        # keep each rank as is or flip it to 1 - rank
        signed = pd.DataFrame({c: X[c] if s > 0 else 1 - X[c] for c, s in FACTOR_SIGNS.items()})
        # average across the 5 columns, one score per stock
        # If all 5 are missing, use 0.5 (a middle score), so the stock is never accidently a top pick
        return signed.mean(axis=1).fillna(0.5).to_numpy()


# Each candidate = (pipeline, numeric feature columns, target column).
# raw features  : values as measured (course baseline)
# rank features : same date percentile ranking from Part 2
# target        : raw 90-day return
candidates = {
    "Ridge, raw features, raw target (course baseline)":
        (make_pipeline(Ridge(alpha=100), NUMERIC_FEATURES), NUMERIC_FEATURES, "target"),
    "Ridge, rank features, excess target":
        (make_pipeline(Ridge(alpha=100), RANK_FEATURES), RANK_FEATURES, "target_excess"),
    "Ridge (alpha=1000), rank features, excess target":
        (make_pipeline(Ridge(alpha=1000), RANK_FEATURES), RANK_FEATURES, "target_excess"),
    "Boosted trees, raw features, excess target":
        (make_pipeline(boosted_trees(), NUMERIC_FEATURES), NUMERIC_FEATURES, "target_excess"),
    "Boosted trees, rank features, excess target":
        (make_pipeline(boosted_trees(), RANK_FEATURES), RANK_FEATURES, "target_excess"),
    "Ridge, 5 factor ranks, excess target":
        (make_pipeline(Ridge(alpha=100), FACTOR_FEATURES), FACTOR_FEATURES, "target_excess"),
    "Factor score (fixed weights, no fitting)":
        (FactorScore(), FACTOR_FEATURES, "target_excess"),
}


def investable(panel):
    """
    Keep only rows that pass the price and liquidity check

    ``target_excess`` is recomputed against the eligible stocks' average on the same date,
    so we can see how one performs aginst others that we could have bought
    """
    keep = panel.close.ge(MIN_PRICE) & panel.log_dollar_volume_63.ge(np.log(MIN_DOLLAR_VOLUME))
    out = panel.loc[keep].copy()
    out["target_excess"] = out.target - out.groupby("date").target.transform("mean")
    return out.reset_index(drop=True)


# Frozen choice for winner model after validation. Fixed weights with no
# retraining needed in the test period.
FINAL_MODEL = "Factor score (fixed weights, no fitting)"


def development_rows(panel):
    """Takes full feature table and returns only stocks that passed the checks with signal dates 2010-2015"""
    dev = investable(panel)
    return dev.loc[dev.date.between(TRAIN_START, experiment["development_end"])].reset_index(drop=True)


def fit_model(pipeline, train, features, target):
    """Fit a copy of the pipeline. Learned preprocessing on training data only."""
    y = train[target]
    low, high = y.quantile([0.01, 0.99])
    return clone(pipeline).fit(train[features + ["sector"]], y.clip(low, high))


def top_n_metrics(frame, scores, n=TOP_N):
    """
    How much the n highest scored stock beat the average stock
    and how much the top 10% beat the bottom 10% 
    """
    # exclude extreme returns
    capped = frame.target.clip(upper=EVAL_RETURN_CAP)
    f = pd.DataFrame({"date": frame.date.to_numpy(), "score": np.asarray(scores),
                      "target_excess": (capped - capped.groupby(frame.date).transform("mean")).to_numpy()})
    # rank within each date
    by_date = f.groupby("date")
    pct = by_date.score.rank(pct=True)
    top_n = f.loc[by_date.score.rank(ascending=False, method="first").le(n)]
    return pd.Series({
        # average of the month and then average across the monthly numbers
        f"top{n}_excess": top_n.groupby("date").target_excess.mean().mean(),
        # top 10% Vs. bottom 10%
        "top_decile_minus_bottom": (f.loc[pct > 0.9].groupby("date").target_excess.mean()
                                    - f.loc[pct <= 0.1].groupby("date").target_excess.mean()).mean(),
    })


def validate(dev, candidates=candidates, validation_years=VALIDATION_YEARS):
    """Runs the model competition. Score every candidate on every fold. Returns (per-fold table, folds)."""
    folds = date_folds(dev, validation_years=validation_years, train_start=TRAIN_START)
    rows = []
    # loop over models 
    for name, (pipeline, features, target) in candidates.items():
        # loop over years
        for year, (tr, va) in zip(validation_years, folds):
            fitted = fit_model(pipeline, dev.iloc[tr], features, target)
            prediction = fitted.predict(dev.iloc[va][features + ["sector"]])
            # Get rank IC and compare across targets.
            ic = prediction_metrics(dev.iloc[va], prediction)
            rows.append({"model": name, "validation_year": year,
                         "rank_IC": ic["mean_daily_rank_IC"],
                         **top_n_metrics(dev.iloc[va], prediction).to_dict()})
    return pd.DataFrame(rows), folds


def summarize(results):
    """Average over folds with how many folds had a positive rank IC."""
    summary = results.groupby("model", sort=False).agg(
        mean_rank_IC=("rank_IC", "mean"), IC_std=("rank_IC", "std"),
        folds_IC_positive=("rank_IC", lambda s: f"{(s > 0).sum()}/{len(s)}"),
        mean_top30_excess=(f"top{TOP_N}_excess", "mean"),
        mean_decile_spread=("top_decile_minus_bottom", "mean"))
    return summary.sort_values("mean_rank_IC", ascending=False)


def yearly_fit_dates(panel, start, end):
    """Retraining rule: refit at `start` and then at the first signal of every later year."""
    signals = pd.DatetimeIndex(np.sort(panel.date.unique()))
    signals = signals[(signals >= start) & (signals <= end)]
    firsts = pd.Series(signals, index=signals).groupby(signals.year).min()
    return pd.DatetimeIndex(sorted({signals[0], *firsts[firsts > signals[0]]}))


def walk_forward(panel, pipeline, features, target, fit_dates, end):
    """Refit at each fit date. Score every signal until the next fit date.

    At each fit date it trains the model on everything that was known by then,
    uses that model to score stocks until the next fit date, and repeats.
    """
    parts, audit = [], []
    # asof - as of the date
    for i, asof in enumerate(fit_dates):
        # get the next fit date
        next_asof = fit_dates[i + 1] if i + 1 < len(fit_dates) else pd.Timestamp(end) + pd.Timedelta(days=1)
        # Picks the training rows:
        # signal on or after 2010-01-0
        # signal before the fit date
        # 90-day result had finished before the fit date
        # result exists
        train = panel.loc[panel.date.ge(TRAIN_START) & panel.date.lt(asof)
                          & panel.label_end.lt(asof) & panel[target].notna()]
        to_score = panel.loc[panel.date.ge(asof) & panel.date.lt(next_asof)]
        # safety check 
        assert train.label_end.max() < to_score.date.min(), "training outcome not finished at fit time"
        # train and score
        model = fit_model(pipeline, train, features, target)
        scored = to_score[["date", "symbol"]].copy()
        scored["score"] = model.predict(to_score[features + ["sector"]])
        scored["fit_date"] = asof
        parts.append(scored)
        # add rows to the audit table
        audit.append({"fit_date": asof.date(), "training_rows": len(train),
                      "last_finished_outcome": train.label_end.max().date(),
                      "scores_signals": f"{to_score.date.min().date()} .. {to_score.date.max().date()}",
                      "scored_rows": len(to_score)})
    return pd.concat(parts, ignore_index=True), pd.DataFrame(audit)
