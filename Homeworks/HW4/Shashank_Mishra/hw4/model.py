"""LightGBM ranking-style regressor, chronological folds, and metrics."""

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

TRAIN_START = pd.Timestamp("2010-01-01")

BASE_PARAMS = dict(
    objective="regression",       # squared error on the cross-sectional rank target
    learning_rate=0.03,
    num_leaves=15,
    min_child_samples=1000,       # very noisy target: each leaf must average >=1000 stock-weeks
    subsample=0.7, subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=10.0,
    max_bin=63,                    # inputs are percentiles; 63 bins is plenty
    n_estimators=300,
    n_jobs=-1, random_state=7, verbose=-1,
)


def model_columns(features, use_sector=True):
    return [f"r_{c}" for c in features] + (["sector"] if use_sector else [])


class RankLGBM:
    """LightGBM regressor on a same-date-normalised target.

    target="rank":   (90-day return percentile within the date) - 0.5
    target="excess": winsorised 90-day return minus the date's mean
    """

    def __init__(self, features, sectors=(), use_sector=False, target="rank", **params):
        self.features = list(features)
        self.use_sector = use_sector
        self.target_col = {"rank": "target_rank", "excess": "target_excess"}[target]
        self.sector_dtype = pd.CategoricalDtype(sorted(sectors))
        self.params = {**BASE_PARAMS, **params}

    def _X(self, frame):
        X = frame[model_columns(self.features, self.use_sector)].copy()
        if self.use_sector:
            X["sector"] = X.sector.astype(self.sector_dtype)
        return X

    def fit(self, frame):
        self.model_ = lgb.LGBMRegressor(**self.params)
        y = frame[self.target_col] - (0.5 if self.target_col == "target_rank" else 0.0)
        self.model_.fit(self._X(frame), y,
                        categorical_feature=["sector"] if self.use_sector else "auto")
        return self

    def predict(self, frame):
        return self.model_.predict(self._X(frame))

    def importance(self):
        gain = self.model_.booster_.feature_importance("gain")
        return pd.Series(gain / gain.sum(), index=model_columns(self.features, self.use_sector)).sort_values(ascending=False)


class RankRidge:
    """Linear baseline on the same ranked inputs (missing -> median rank 0.5)."""

    def __init__(self, features, alpha=10.0):
        self.features = list(features)
        self.alpha = alpha

    def _X(self, frame):
        return frame[[f"r_{c}" for c in self.features]].fillna(0.5).to_numpy()

    def fit(self, frame):
        self.model_ = Ridge(alpha=self.alpha).fit(self._X(frame), frame.target_rank - 0.5)
        return self

    def predict(self, frame):
        return self.model_.predict(self._X(frame))


def training_rows(panel, fit_date, train_start=TRAIN_START):
    """Rows whose 90-day outcome was fully observed before the fit date."""
    return panel.loc[panel.date.ge(train_start) & panel.target_rank.notna() & panel.label_end.lt(fit_date)]


def yearly_folds(panel, years):
    """Expanding-window folds: fit just before the first signal of each year."""
    folds = []
    for y in years:
        valid = panel.loc[panel.date.dt.year.eq(y) & panel.target_rank.notna()]
        fit_date = valid.date.min()
        train = training_rows(panel, fit_date)
        assert train.label_end.max() < valid.date.min()
        folds.append((y, fit_date, train, valid))
    return folds


def daily_ic(frame, score_col="score", target="target"):
    return frame.groupby("date").apply(
        lambda g: g[score_col].corr(g[target], method="spearman"), include_groups=False)


def prediction_metrics(frame, score_col="score", top_n=30):
    ic = daily_ic(frame, score_col)
    tops = frame.sort_values(score_col, ascending=False).groupby("date").head(top_n)
    decile = frame.groupby("date")[score_col].rank(pct=True).ge(0.9)
    return pd.Series({
        "mean_rank_IC": ic.mean(),
        "IC_t_stat": ic.mean() / ic.std() * np.sqrt(len(ic)),   # overlapping labels: optimistic
        "pct_dates_IC>0": (ic > 0).mean(),
        "top_decile_minus_all_90d": frame.loc[decile, "target"].mean() - frame.target.mean(),
        f"top{top_n}_median_minus_all_median_90d": tops.groupby("date").target.median().mean()
        - frame.groupby("date").target.median().mean(),
        "dates": len(ic),
    })
