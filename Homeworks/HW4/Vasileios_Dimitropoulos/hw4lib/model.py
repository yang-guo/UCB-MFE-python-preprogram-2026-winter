"""Cross-sectional preprocessing, models and walk-forward fitting.

Preprocessing is **cross-sectional**: each numeric feature is converted to its
percentile rank among the eligible stocks on the same signal date.  This uses
only same-date information (no learned statistics, so nothing can leak from a
training period into the test), removes the effect of bad units/currencies
in vendor fundamentals and makes 2010 and 2025 feature scales comparable.

Targets: the model predicts each stock's 90-calendar-day return *relative to
the other eligible stocks on the same date*.  Two versions are compared in
validation:

* ``rank``   -- percentile rank of the 90-day return on its date, minus 0.5;
* ``excess`` -- 90-day return minus the same-date median, winsorised at the
  same-date 1st/99th percentiles (a direct return forecast in decimals).

The walk-forward rule, "train only on rows whose label ended strictly before
the fit date", follows ``date_folds`` / the quarterly refit loop in Lecture 8.2.
"""

import numpy as np
import pandas as pd
from lightgbm import LGBMRanker as _LGBMRank, LGBMRegressor
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge

from .features import NUMERIC_FEATURES

SECTORS = ["Basic Materials", "Communication Services", "Consumer Cyclical", "Consumer Defensive",
           "Energy", "Financial Services", "Healthcare", "Industrials", "Real Estate",
           "Technology", "Utilities", "Unknown"]


def add_model_columns(panel):
    """Rank features per date, build both target versions.  Pure same-date ops."""
    p = panel.copy()
    g = p.groupby("date")
    ranks = g[NUMERIC_FEATURES].rank(pct=True)
    p[[f"r_{c}" for c in NUMERIC_FEATURES]] = ranks.to_numpy()
    p["sector_cat"] = pd.Categorical(p.sector, categories=SECTORS)
    # Targets (future information -- used only for evaluation; every fit recomputes
    # them inside its own training rows, see ``training_rows``).
    p["y_rank"], p["y_excess"] = make_targets(p)
    return p


RANK_FEATURES = [f"r_{c}" for c in NUMERIC_FEATURES]
MODEL_INPUTS = RANK_FEATURES + ["sector_cat"]


class GBMModel:
    """LightGBM on ranked features (+ sector as a categorical).

    ``objective="regression"`` fits the chosen target with squared error.
    ``objective="lambdarank"`` treats each signal date as one query and
    optimises the ordering of the *top* of the list (NDCG@k), using the
    90-day return's same-date quantile bucket (0..``n_bins``-1) as relevance.
    """

    def __init__(self, target="y_rank", objective="regression", n_bins=10, top_k=50, **params):
        self.target, self.objective, self.n_bins, self.top_k = target, objective, n_bins, top_k
        defaults = dict(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=2000,
                        subsample=0.7, subsample_freq=1, colsample_bytree=0.7, reg_lambda=10.0,
                        random_state=0, n_jobs=-1, verbose=-1)
        defaults.update(params)
        self.params = defaults

    def fit(self, rows):
        if self.objective == "lambdarank":
            rows = rows.sort_values(["date", "symbol"])
            rel = np.floor((rows["y_rank"] + 0.5) * self.n_bins).clip(0, self.n_bins - 1).astype(int)
            groups = rows.groupby("date", sort=True).size().to_numpy()
            self.model = _LGBMRank(objective="lambdarank", label_gain=list(range(self.n_bins)),
                                   lambdarank_truncation_level=self.top_k,
                                   **self.params)
            self.model.fit(rows[MODEL_INPUTS], rel, group=groups)
        else:
            self.model = LGBMRegressor(**self.params).fit(rows[MODEL_INPUTS], rows[self.target])
        return self

    def predict(self, rows):
        return self.model.predict(rows[MODEL_INPUTS])


class RidgeRanker:
    """Linear baseline: ranked features (missing -> 0.5, the median) + sector dummies."""

    def __init__(self, target="y_rank", alpha=100.0):
        self.target, self.alpha = target, alpha

    @staticmethod
    def _design(rows):
        x = rows[RANK_FEATURES].fillna(0.5) - 0.5
        missing = rows[RANK_FEATURES].isna().astype(float).add_suffix("_missing")
        dummies = pd.get_dummies(rows.sector_cat, prefix="sector", dtype=float)
        return pd.concat([x, missing, dummies], axis=1).to_numpy()

    def fit(self, rows):
        self.model = Ridge(alpha=self.alpha).fit(self._design(rows), rows[self.target])
        return self

    def predict(self, rows):
        return self.model.predict(self._design(rows))


class MomentumScore:
    """No fitting: rank by 12-1 month momentum (a classic baseline)."""
    target = "y_rank"

    def fit(self, rows):
        return self

    def predict(self, rows):
        return rows["r_mom_252_21"].fillna(0.5).to_numpy()


def make_targets(rows):
    """Same-date relative targets computed ONLY from the rows passed in."""
    g = rows.groupby("date").target
    y_rank = g.rank(pct=True) - 0.5
    lo, hi = g.transform(lambda x: x.quantile(0.01)), g.transform(lambda x: x.quantile(0.99))
    y_excess = rows.target.clip(lo, hi) - g.transform("median")
    return y_rank, y_excess


def training_rows(panel, fit_date, train_start="2010-01-01"):
    """Rows usable at ``fit_date``: signal >= train_start and outcome already known.

    The targets are recomputed *within* these rows, so a same-date peer whose
    outcome ends on/after the fit date (e.g. a delayed exit) can never influence
    another row's rank target.
    """
    fit_date = pd.Timestamp(fit_date)
    rows = panel.loc[panel.date.ge(train_start) & panel.date.lt(fit_date)
                     & panel.label_end.lt(fit_date) & panel.target.notna()].copy()
    rows["y_rank"], rows["y_excess"] = make_targets(rows)
    return rows


def walk_forward(panel, make_model, fit_dates, score_end, train_start="2010-01-01"):
    """Fit at each fit date; score signals from that date until the next fit.

    Returns (predictions, fit_audit).  Scoring rows do NOT require a known
    target: censored stocks are still scored (we know their features).
    """
    fit_dates = pd.DatetimeIndex(fit_dates)
    preds, audit = [], []
    for i, fit_date in enumerate(fit_dates):
        stop = fit_dates[i + 1] if i + 1 < len(fit_dates) else pd.Timestamp(score_end) + pd.Timedelta(days=1)
        train = training_rows(panel, fit_date, train_start)
        score = panel.loc[panel.date.ge(fit_date) & panel.date.lt(stop)]
        if not len(score):
            continue
        model = make_model().fit(train)
        out = score[["date", "symbol"]].copy()
        out["score"] = model.predict(score)
        out["fit_date"] = fit_date
        preds.append(out)
        audit.append({"fit_date": fit_date.date(), "training_rows": len(train),
                      "first_training_signal": train.date.min().date() if len(train) else None,
                      "last_training_signal": train.date.max().date() if len(train) else None,
                      "last_training_label_end": train.label_end.max().date() if len(train) else None,
                      "first_scored_signal": out.date.min().date(), "last_scored_signal": out.date.max().date(),
                      "scored_rows": len(out)})
        assert len(train) == 0 or train.label_end.max() < out.date.min(), "look-ahead in training labels"
    return pd.concat(preds, ignore_index=True), pd.DataFrame(audit)


def prediction_metrics(panel, preds):
    """Forecast quality on rows whose outcome is known.

    * mean_IC: average same-date Spearman correlation between score and the raw
      90-day return (ranking skill, what a top-N strategy needs);
    * IC_t: mean/std*sqrt(n) -- optimistic because weekly 90-day labels overlap;
    * top_decile_excess: average 90-day return of the top-10% scores minus the
      same-date average of all eligible stocks.
    """
    m = panel[["date", "symbol", "target"]].merge(preds, on=["date", "symbol"])
    m = m.loc[m.target.notna()]
    ic = m.groupby("date").apply(lambda x: spearmanr(x.score, x.target).statistic, include_groups=False)
    m["decile"] = m.groupby("date").score.rank(pct=True)
    mean_by_date = m.groupby("date").target.transform("mean")
    top = (m.target - mean_by_date)[m.decile > 0.9].groupby(m.date).mean()
    bottom = (m.target - mean_by_date)[m.decile <= 0.1].groupby(m.date).mean()
    return pd.Series({"dates": ic.notna().sum(), "mean_IC": ic.mean(), "IC_t": ic.mean() / ic.std() * np.sqrt(ic.notna().sum()),
                      "IC_hit_rate": (ic > 0).mean(), "top_decile_excess": top.mean(),
                      "bottom_decile_excess": bottom.mean()})
