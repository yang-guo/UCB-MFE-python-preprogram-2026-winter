"""Models, chronological validation and point-in-time refits.

Training rows are weekly signal dates. A model fitted after the close of
``fit_date`` only sees rows whose 90-day outcome ended strictly before
``fit_date`` (``label_end < fit_date``), which is the purge rule required by
the assignment. All features are same-date percentile ranks, so the only
learned preprocessing is the median imputer inside the Ridge pipeline, which
is fitted on the training rows only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline

from .features import RANK_FEATURES, FeatureStore


def make_model(kind: str, seed: int = 0):
    if kind == "ridge":
        return make_pipeline(SimpleImputer(strategy="median", add_indicator=True), Ridge(alpha=1000.0))
    if kind == "hgb":
        # Shallow, heavily regularized trees: the signal-to-noise ratio of
        # 90-day stock returns is low. Missing ranks are handled natively.
        return HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
                                             min_samples_leaf=1000, l2_regularization=1.0,
                                             early_stopping=False, random_state=seed)
    raise ValueError(kind)


def training_dates(calendar: pd.DatetimeIndex, start="2010-01-01", end="2026-09-30", step=5):
    """Every ``step``-th session from ``start`` (weekly sampling of signal dates)."""
    dates = calendar[(calendar >= pd.Timestamp(start)) & (calendar <= pd.Timestamp(end))]
    return dates[::step][:-1] if dates[::step][-1] == calendar[-1] else dates[::step]


def training_rows(panel: pd.DataFrame, fit_date, start="2010-01-01") -> pd.DataFrame:
    """Rows usable by a model fitted after the close of ``fit_date``."""
    fit_date = pd.Timestamp(fit_date)
    rows = panel.loc[panel.date.ge(start) & panel.date.lt(fit_date)
                     & panel.label_end.lt(fit_date) & panel.target.notna()]
    assert rows.label_end.max() < fit_date
    return rows


def fit(kind, rows, target="target_rank"):
    model = make_model(kind)
    model.fit(rows[RANK_FEATURES].to_numpy(dtype="float32"), rows[target].to_numpy())
    return model


def predict(model, frame):
    return model.predict(frame[RANK_FEATURES].to_numpy(dtype="float32"))


def daily_ic(frame, score="score", target="target"):
    return frame.groupby("date").apply(lambda x: x[score].corr(x[target], method="spearman"),
                                       include_groups=False)


def top_bucket_excess(frame, score="score", q=0.1, clip=(-1.0, 2.0)):
    """Mean 90-day return of the top ``q`` fraction minus the same-date mean.

    Returns are clipped to [-100%, +200%] so that a few bad prices cannot
    dominate the comparison (the raw data contains moves of +10,000%).
    """
    def one(x):
        k = max(1, int(len(x) * q))
        t = x.target.clip(*clip)
        return t.loc[x.nlargest(k, score).index].mean() - t.mean()
    return frame.groupby("date").apply(one, include_groups=False)


class WalkForwardScorer:
    """Scores for any signal date from the latest scheduled point-in-time fit.

    The refit schedule is fixed in advance (first session of each quarter).
    A score requested for signal date t uses the most recent fit made at or
    before t, trained only on outcomes that ended before that fit.
    """

    def __init__(self, store: FeatureStore, panel: pd.DataFrame, kind: str, fit_dates,
                 max_rank: int | None, glitch: pd.DataFrame | None = None, train_start="2010-01-01",
                 shared_models: dict | None = None, shared_audit: list | None = None):
        self.store, self.panel, self.kind = store, panel, kind
        self.fit_dates = pd.DatetimeIndex(sorted(fit_dates))
        self.max_rank, self.glitch, self.train_start = max_rank, glitch, train_start
        # Fits do not depend on the trading screen, so configurations can share them.
        self.models = shared_models if shared_models is not None else {}
        self.audit = shared_audit if shared_audit is not None else []

    def model_for(self, signal_date):
        i = self.fit_dates.searchsorted(signal_date, side="right") - 1
        if i < 0:
            raise ValueError(f"no scheduled fit at or before {signal_date.date()}")
        fd = self.fit_dates[i]
        if fd not in self.models:
            rows = training_rows(self.panel, fd, self.train_start)
            self.models[fd] = fit(self.kind, rows)
            self.audit.append({"fit_date": fd, "train_rows": len(rows), "first_signal": rows.date.min(),
                               "last_signal": rows.date.max(), "last_label_end": rows.label_end.max()})
        return fd, self.models[fd]

    def scores(self, signal_date) -> pd.DataFrame:
        fd, model = self.model_for(signal_date)
        xs = self.store.cross_section([signal_date], max_rank=self.max_rank)
        if self.glitch is not None:
            r = self.store.calendar.get_loc(signal_date)
            flagged = self.glitch.iloc[r]
            xs = xs.loc[~xs.symbol.map(flagged).fillna(False).to_numpy()]
        xs["score"] = predict(model, xs)
        xs["fit_date"] = fd
        assert (xs.fit_date <= xs.date).all()
        return xs[["date", "symbol", "score", "fit_date", "universe_rank", "mcap"]]


def glitch_flags(store: FeatureStore, lookback=252, up=1.0, down=-0.5) -> pd.DataFrame:
    """True if a stock had a one-session move above +100% or below -50% in the last year.

    In this dataset such moves are mostly data problems: unadjusted reverse
    splits, ticker re-mappings (e.g. LCI in Dec 2013 jumps from $31 to $531)
    and one-day bad prints. All observations are at or before the signal
    date, so the screen uses only information available at the time. It also
    removes some genuine events (biotech news, buyouts), a deliberate cost.
    """
    flags = np.zeros(store.md.adj.shape, dtype=bool)
    for start in range(0, store.md.adj.shape[1], 500):
        px = store.md.adj.iloc[:, start:start + 500].astype("float64")
        ret = px.pct_change(fill_method=None)
        event = (ret > up) | (ret < down)
        flags[:, start:start + 500] = event.rolling(lookback, min_periods=1).max().fillna(0).to_numpy() > 0
    return pd.DataFrame(flags, index=store.md.adj.index, columns=store.md.adj.columns)


PANEL_COLUMNS = ["date", "symbol", "universe_rank", "target", "target_rank", "target_excess", "label_end"]


def build_panel(store: FeatureStore, dates, chunk: int = 40) -> pd.DataFrame:
    """Eligible cross-sections on the sampled signal dates, compact float32 storage."""
    parts = []
    for i in range(0, len(dates), chunk):
        xs = store.cross_section(dates[i:i + chunk])
        part = xs[PANEL_COLUMNS + RANK_FEATURES].copy()
        part[RANK_FEATURES + ["target", "target_rank", "target_excess"]] = \
            part[RANK_FEATURES + ["target", "target_rank", "target_excess"]].astype("float32")
        parts.append(part)
    return pd.concat(parts, ignore_index=True)
