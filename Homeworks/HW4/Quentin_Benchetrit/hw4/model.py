"""Return models and the walk-forward loop that fits and scores them."""

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline

from .features import FEATURES

FIRST_TRAINING_SIGNAL = "2010-01-01"

# Direction in which each characteristic is expected to line up with future
# returns, taken from the published anomalies and not from this sample.
PRIOR_SIGNS = {
    # momentum, with the usual one-month reversal
    "mom_12_1": +1, "mom_6_1": +1, "high_252": +1, "ret_21": -1,
    # low risk
    "vol_63": -1, "vol_252": -1, "idio_vol": -1, "beta": -1, "max_ret_21": -1,
    # profitability and earnings quality
    "roa": +1, "roe": +1, "gross_profitability": +1, "operating_margin": +1,
    "asset_turnover": +1, "accruals": -1,
    # change in fundamentals, investment and issuance
    "revenue_acceleration": +1, "income_change": +1, "margin_change": +1,
    "asset_growth": -1, "share_growth": -1,
    # value
    "earnings_yield": +1, "book_to_price": +1, "sales_to_price": +1,
    "cash_flow_yield": +1, "fcf_yield": +1, "ebitda_to_ev": +1,
}
RISK_FEATURES = ["vol_63", "vol_252", "idio_vol", "beta", "max_ret_21"]


def ridge(alpha=1000.0, positive=False):
    # Inputs are centred ranks, so a missing one is placed in the middle.
    return make_pipeline(SimpleImputer(strategy="constant", fill_value=0.0),
                         Ridge(alpha=alpha, positive=positive))


def boosted_trees(monotone=None, **overrides):
    settings = dict(max_depth=4, learning_rate=0.05, max_iter=300, min_samples_leaf=500,
                    l2_regularization=1.0, max_features=0.5, random_state=0)
    settings.update(overrides)
    return HistGradientBoostingRegressor(monotonic_cst=monotone, **settings)


def shrunk_beta(rows):
    """One-year beta pulled a third of the way to 1; 1 when there is no history."""
    return (0.67 * rows.raw_beta.fillna(1.0) + 0.33).clip(0, 3)


class ReturnModel:
    """Forecast the part of the 90-day return that stock selection can earn.

    target="alpha":  return minus beta times the benchmark's move over the
                     same 90 days. The market term is common to all candidates
                     and I do not try to forecast it.
    target="median": return minus the median stock's return on that date.

    With `signs`, each input is flipped so that a positive coefficient agrees
    with the prior direction; pair it with ridge(positive=True) to keep the
    regression from betting against the prior. The clipping bounds of the
    target are learned from the training rows of each fit.
    """

    def __init__(self, estimator, features=FEATURES, signs=None, target="alpha", clip=(0.01, 0.99)):
        self.estimator = estimator
        self.features = list(features)
        self.direction = np.array([signs[f] if signs else 1 for f in self.features], dtype="float32")
        self.target = target
        self.clip = clip

    def _inputs(self, rows):
        return (rows[self.features].to_numpy() - 0.5) * self.direction

    def _target(self, rows):
        if self.target == "alpha":
            return rows.target - shrunk_beta(rows) * rows.benchmark_target
        return rows.target - rows.groupby("date").target.transform("median")

    def fit(self, rows):
        y = self._target(rows)
        self.bounds_ = tuple(y.quantile(list(self.clip)))
        self.estimator_ = clone(self.estimator).fit(self._inputs(rows), y.clip(*self.bounds_))
        return self

    def predict(self, rows):
        return self.estimator_.predict(self._inputs(rows))

    def weights(self):
        """Coefficients of a linear estimator, in return per unit of rank."""
        return pd.Series(self.estimator_[-1].coef_ * self.direction, index=self.features)


def training_rows(panel, fit_date):
    """Rows whose whole holding period ended before the fit date."""
    return panel.loc[panel.eligible & panel.target.notna()
                     & (panel.date >= FIRST_TRAINING_SIGNAL) & (panel.label_end < fit_date)]


def quarter_ends(calendar, start, end):
    """Last session of each quarter in [start, end]: the dates on which we refit."""
    days = pd.Series(calendar, index=calendar)
    last = days.groupby(calendar.to_period("Q")).max()
    return pd.DatetimeIndex(last[(last >= start) & (last <= end)])


def walk_forward(panel, model, fit_dates, score_until):
    """Refit after the close of each fit date and score signals until the next.

    Returns the predictions and one audit row per fit showing that every
    training outcome ended before the first signal the model scored.
    """
    fit_dates = pd.DatetimeIndex(fit_dates)
    limits = list(fit_dates[1:]) + [pd.Timestamp(score_until) + pd.Timedelta(days=1)]
    predictions, audit = [], []
    for fit_date, next_fit in zip(fit_dates, limits):
        train = training_rows(panel, fit_date)
        score = panel.loc[panel.eligible & (panel.date >= fit_date) & (panel.date < next_fit)]
        if score.empty:
            continue
        fitted = model.fit(train)
        out = score[["date", "symbol"]].copy()
        out["prediction"] = fitted.predict(score)
        out["fit_date"] = fit_date
        predictions.append(out)
        audit.append({"fit_date": fit_date, "training_rows": len(train),
                      "first_training_signal": train.date.min(),
                      "last_training_signal": train.date.max(),
                      "last_label_end": train.label_end.max(),
                      "first_scored_signal": score.date.min(),
                      "last_scored_signal": score.date.max()})
        assert train.label_end.max() < score.date.min()
    return pd.concat(predictions, ignore_index=True), pd.DataFrame(audit)


def forecast_report(predictions, panel, top=50):
    """Rank IC by year, and how the `top` highest forecasts did against the index."""
    scored = predictions.merge(
        panel[["date", "symbol", "target", "benchmark_target", "raw_beta"]], on=["date", "symbol"])
    scored = scored.loc[scored.target.notna()]
    ic = scored.groupby("date").apply(
        lambda x: x.prediction.corr(x.target, method="spearman"), include_groups=False)
    best = scored.sort_values("prediction", ascending=False).groupby("date").head(top).groupby("date")
    versus_index = best.target.mean() - best.benchmark_target.first()
    report = {"rank_ic": ic.mean()}
    report.update({f"ic_{year}": value for year, value in ic.groupby(ic.index.year).mean().items()})
    report[f"top{top}_vs_index"] = versus_index.mean()
    report.update({f"vs_index_{year}": value
                   for year, value in versus_index.groupby(versus_index.index.year).mean().items()})
    report[f"top{top}_beta"] = best.raw_beta.mean().mean()
    return pd.Series(report)
