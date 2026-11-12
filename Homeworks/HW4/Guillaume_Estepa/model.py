"""Models and the walk-forward (expanding window) fitting loop."""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

from features import FEATURES

TRAIN_START = "2010-01-01"


# name -> (estimator factory, target column). Compared during development.
#
# * ridge: linear model on ranked features, a transparent baseline.
# * hgb:   shallow, heavily regularised gradient-boosted trees, which can pick
#          up non-linear effects and interactions and accept missing values.
# * target "target":      90-day return minus the same-date median return
#                         (winsorised at the training 1%/99% quantiles);
#   target "target_rank": the same return as a same-date percentile rank,
#                         which is insensitive to extreme (or bad) prices.
def _ridge():
    return Ridge(alpha=1000.0)


def _hgb():
    return HistGradientBoostingRegressor(
        max_iter=200, learning_rate=0.05, max_depth=4, min_samples_leaf=500,
        l2_regularization=1.0, early_stopping=False, random_state=0)


MODELS = {
    "ridge_return": (_ridge, "target"),
    "ridge_rank": (_ridge, "target_rank"),
    "hgb_return": (_hgb, "target"),
    "hgb_rank": (_hgb, "target_rank"),
}


def design_matrix(frame, name):
    """Ranked features; the linear model needs missing ranks set to the middle (0)."""
    X = frame[FEATURES].to_numpy(dtype=np.float32)
    return np.nan_to_num(X, nan=0.0) if name.startswith("ridge") else X


def refit_dates(signal_dates, first_fit):
    """Fit at ``first_fit`` and then at the first signal date of each new year."""
    dates = pd.DatetimeIndex(sorted(set(signal_dates)))
    dates = dates[dates >= pd.Timestamp(first_fit)]
    first_of_year = pd.Series(dates, index=dates).groupby(dates.year).min()
    return pd.DatetimeIndex(sorted(set(first_of_year) | {dates[0]}))


def walk_forward_predict(panel, name, fit_dates, train_start=TRAIN_START):
    """Predict every row dated on/after the first fit date, without look-ahead.

    The model used on a signal date was fitted at the latest fit date on or
    before it, using only rows whose 90-day outcome had *already ended*
    before that fit date. Target winsorisation limits are learned on the
    training rows of each fit. Returns (predictions, fit log, fitted models).
    """
    make_estimator, target = MODELS[name]
    predictions = pd.Series(np.nan, index=panel.index, name="prediction")
    labelled = panel[target].notna() & (panel.date >= pd.Timestamp(train_start))
    log, models = [], {}
    fit_dates = list(fit_dates)
    for n, fit_date in enumerate(fit_dates):
        train = panel.loc[labelled & (panel.label_end < fit_date)]
        assert len(train) and train.label_end.max() < fit_date and train.date.max() < fit_date
        low, high = train[target].quantile([0.01, 0.99])
        model = make_estimator().fit(design_matrix(train, name), train[target].clip(low, high))
        models[fit_date] = model
        until = fit_dates[n + 1] if n + 1 < len(fit_dates) else pd.Timestamp.max
        apply_to = (panel.date >= fit_date) & (panel.date < until)
        predictions[apply_to] = model.predict(design_matrix(panel.loc[apply_to], name))
        log.append({"fit_date": fit_date, "train_rows": len(train),
                    "first_train_signal": train.date.min(), "last_train_signal": train.date.max(),
                    "last_label_end": train.label_end.max(), "predicted_rows": int(apply_to.sum())})
    return predictions, pd.DataFrame(log), models
