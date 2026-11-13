"""Chronological validation and walk-forward scoring with maturity checks.

Rule (assignment + Lecture 7.3/8.2): a model fitted after the close of F may
train only on rows with ``date < F`` *and* ``label_end < F``.  This module
checks it at every fit and records a fit audit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone

from . import protocol
from .models import build_model
from .protocol import Mode


class LeakageError(AssertionError):
    pass


def training_mask(panel, fit_date, train_start, target, window_years=None):
    fit_date = pd.Timestamp(fit_date)
    start = pd.Timestamp(train_start)
    if window_years:
        start = max(start, fit_date - pd.DateOffset(years=window_years))
    return (panel.eligible & panel.date.ge(start) & panel.date.lt(fit_date)
            & panel.label_end.lt(fit_date) & panel[target].notna())


def assert_matured(train: pd.DataFrame, fit_date) -> None:
    fit_date = pd.Timestamp(fit_date)
    if len(train) and not (train.label_end < fit_date).all():
        raise LeakageError(f"Training target completed at/after fit date {fit_date.date()}")
    if len(train) and not (train.date < fit_date).all():
        raise LeakageError("Training row observed at/after the fit date")


def validation_folds(panel, years, train_start, target, window_years=None):
    """One fold per validation year; model fitted at that year's first signal."""
    folds = []
    for y in years:
        valid = panel.eligible & panel.date.dt.year.eq(y) & panel.target.notna()
        if not valid.any():
            raise ValueError(f"No validation rows in {y}")
        fit_date = panel.loc[valid, "date"].min()
        train = training_mask(panel, fit_date, train_start, target, window_years)
        assert_matured(panel.loc[train], fit_date)
        folds.append({"year": y, "fit_date": fit_date, "train": np.flatnonzero(train),
                      "valid": np.flatnonzero(valid)})
    return folds


def fit_dates(grid_dates: pd.Series, start, end, refit: str) -> list[pd.Timestamp]:
    """First signal date on/after each period start within [start, end]."""
    d = pd.Series(pd.to_datetime(grid_dates.unique())).sort_values()
    d = d[(d >= pd.Timestamp(start)) & (d <= pd.Timestamp(end))]
    if refit == "once":
        return [d.iloc[0]]
    freq = {"quarterly": "Q", "annual": "Y", "monthly": "M"}[refit]
    return list(d.groupby(d.dt.to_period(freq)).min())


def walk_forward_scores(panel, features, model_kind, model_params, target, score_start, score_end,
                        train_start, refit="quarterly", window_years=None, mode=Mode.RESEARCH,
                        extra_columns=("sector", "vol_63")):
    """Fit on matured rows at each fit date; score eligible rows until the next fit."""
    if Mode(mode) is Mode.RESEARCH and pd.Timestamp(score_end) > protocol.DEVELOPMENT_END:
        raise protocol.HoldoutViolation("Research scoring may not extend past the development period")
    fits = fit_dates(panel.loc[panel.eligible, "date"], score_start, score_end, refit)
    template = build_model(model_kind, model_params)
    parts, audit = [], []
    for k, F in enumerate(fits):
        nxt = fits[k + 1] if k + 1 < len(fits) else pd.Timestamp(score_end) + pd.Timedelta(days=1)
        train = panel.loc[training_mask(panel, F, train_start, target, window_years)]
        assert_matured(train, F)
        to_score = panel.loc[panel.eligible & panel.date.ge(F) & panel.date.lt(nxt)
                             & panel.date.le(pd.Timestamp(score_end))]
        if not len(train) or not len(to_score):
            continue
        model = clone(template).fit(train[features].to_numpy(float), train[target].to_numpy(float))
        out = to_score[["date", "symbol", *extra_columns]].copy()
        out["score"] = model.predict(to_score[features].to_numpy(float))
        out["fit_date"] = F
        out["train_max_label_end"] = train.label_end.max()
        parts.append(out)
        audit.append({"fit_date": F, "train_rows": len(train), "train_first_signal": train.date.min(),
                      "train_last_signal": train.date.max(), "train_max_label_end": train.label_end.max(),
                      "scored_rows": len(out), "first_scored": out.date.min(), "last_scored": out.date.max(),
                      "_model": model})
    scores = pd.concat(parts, ignore_index=True)
    return scores, pd.DataFrame(audit)
