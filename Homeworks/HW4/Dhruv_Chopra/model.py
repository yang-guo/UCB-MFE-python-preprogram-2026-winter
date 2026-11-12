"""Cross-sectional ranking model, walk-forward validation and a refit schedule.

Why rank-transform? Raw features and 90-day returns are heavy-tailed and their
scale drifts over 2010-2026 (e.g. dollar volume grows, volatility regimes
change). Ranking every feature and the target *within each date* (percentile
in [0, 1]) makes the problem stationary, needs no learned scaler (so there is
no preprocessing to leak), and focuses the model on what the strategy needs:
ordering stocks on the same day.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

from features import FEATURES, PANEL_FEATURES

TRAIN_START = pd.Timestamp("2010-01-01")
SAMPLE_EVERY = 5          # sessions between training cross-sections (weekly)
MIN_ADV = 10e6            # tradeable screen: >= $10M average daily dollar volume (63 sessions)


def eligible_mask(data, feats, t):
    """Stocks we may score at signal close t: dated universe member, quoted at
    t, 63 complete daily returns (lecture convention) and a volume history."""
    ok = (data.member[t] & np.isfinite(data.adj_close[t]) & np.isfinite(feats["vol_63"][t])
          & np.isfinite(feats["adv_63"][t]))
    dup = getattr(data, "dup_copy", None)
    return ok & ~dup[t] if dup is not None else ok    # one ticker per company (data.duplicate_copies)


def tradeable_mask(data, feats, t):
    """Eligible and liquid enough to buy. Uses only data through t."""
    return eligible_mask(data, feats, t) & (np.nan_to_num(feats["adv_63"][t]) >= MIN_ADV)


def _pct_rank(a):
    """Percentile rank in (0, 1]; NaN stays NaN. Ties share the average rank."""
    s = pd.Series(a)
    return s.rank(pct=True).to_numpy(np.float32)


def cross_section(data, feats, t, columns=None, features=FEATURES):
    """Ranked feature matrix for the eligible stocks at session t. Each feature
    is ranked on its own, so adding columns never changes another column."""
    cols = np.flatnonzero(eligible_mask(data, feats, t) if columns is None else columns)
    X = np.column_stack([_pct_rank(feats[f][t, cols]) for f in features]) if len(cols) else np.empty((0, len(features)))
    return cols, X


def training_panel(data, feats, target, label_end, start=TRAIN_START, end=None, step=SAMPLE_EVERY):
    """Stack weekly cross-sections. Future information is limited to the
    target/label_end columns, which are only used as y and for purging."""
    cal = data.calendar
    end = cal[-1] if end is None else pd.Timestamp(end)
    rows = np.flatnonzero((cal >= start) & (cal <= end))[::step]
    pieces = []
    for t in rows:
        cols, X = cross_section(data, feats, t, features=PANEL_FEATURES)
        if not len(cols):
            continue
        y_raw = target[t, cols]
        piece = pd.DataFrame(X, columns=PANEL_FEATURES)
        piece["t"] = t
        piece["n"] = cols
        piece["target"] = y_raw
        piece["target_rank"] = _pct_rank(y_raw)
        piece["label_end"] = label_end[t, cols]
        piece["tradeable"] = tradeable_mask(data, feats, t)[cols]
        pieces.append(piece)
    panel = pd.concat(pieces, ignore_index=True)
    panel["date"] = cal[panel.t.to_numpy()]
    return panel


def make_model(kind):
    if kind == "hgb":
        # Shallow, heavily regularised trees: the 90-day signal-to-noise ratio is tiny.
        return HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
                                             min_samples_leaf=2000, l2_regularization=1.0,
                                             random_state=0)
    if kind == "ridge":
        return RidgeOnRanks(alpha=10.0)
    raise ValueError(kind)


class RidgeOnRanks:
    """Ridge on centred ranks; missing ranks become 0 (the cross-sectional median)
    plus a missing-indicator per feature. No statistics are learned before Ridge."""

    def __init__(self, alpha=10.0):
        self.model = Ridge(alpha=alpha)

    @staticmethod
    def _design(X):
        missing = np.isnan(X)
        return np.hstack([np.where(missing, 0.0, X - 0.5), missing.astype(float)])

    def fit(self, X, y):
        self.model.fit(self._design(X), y)
        return self

    def predict(self, X):
        return self.model.predict(self._design(X))

    def contributions(self, X):
        """Per-feature contribution to the prediction [n, n_features]: the
        rank term plus that feature's missing-indicator term."""
        d = self._design(X) * self.model.coef_
        k = X.shape[1]
        return d[:, :k] + d[:, k:]


def usable_training_rows(panel, fit_t, start=TRAIN_START):
    """Rows whose 90-day outcome was observed strictly before the fit session."""
    return (panel.date.ge(start) & panel.label_end.lt(fit_t) & panel.target.notna()).to_numpy()


def fit(kind, panel, fit_t, features=FEATURES):
    rows = usable_training_rows(panel, fit_t)
    if rows.sum() < 10_000:
        raise ValueError(f"Too few matured labels ({rows.sum()}) to fit at session {fit_t}")
    train = panel.loc[rows]
    assert train.label_end.max() < fit_t and train.t.max() < fit_t   # purge check
    model = make_model(kind).fit(train[list(features)].to_numpy(), train.target_rank.to_numpy())
    model.n_train_ = int(rows.sum())
    model.features_ = list(features)
    return model


def daily_ic(frame, score="score", target="target"):
    """Mean Spearman correlation between score and realised return per date."""
    ic = frame.groupby("date").apply(
        lambda g: g[score].corr(g[target], method="spearman") if len(g) >= 20 else np.nan,
        include_groups=False)
    return ic


def walk_forward(kind, panel, data, years, refit="annual", features=FEATURES):
    """Out-of-sample scores for each validation year.

    The model for a year is fitted at that year's first sampled signal using
    only outcomes that matured before it (purged expanding window).
    """
    out = []
    for year in years:
        valid = panel.date.dt.year.eq(year) & panel.target.notna()
        fit_t = int(panel.loc[valid, "t"].min())
        model = fit(kind, panel, fit_t, features)
        part = panel.loc[valid, ["date", "t", "n", "target", "tradeable"]].copy()
        part["score"] = model.predict(panel.loc[valid, list(features)].to_numpy())
        part["n_train"] = model.n_train_
        out.append(part)
    return pd.concat(out, ignore_index=True)


class ModelSchedule:
    """Refit at the first signal session of each calendar quarter (and at the
    first signal), using only completed outcomes. predict(t) uses the most
    recent model fitted at or before t."""

    def __init__(self, kind, panel, data, feats, features=FEATURES):
        self.kind, self.panel, self.data, self.feats = kind, panel, data, feats
        self.features = list(features)
        self.models = {}       # fit session -> model
        self.fit_log = []
        self.first_t = None

    def _fit_session(self, t):
        cal = self.data.calendar
        quarter_start = cal[t].to_period("Q").start_time
        first_in_quarter = int(cal.searchsorted(quarter_start))
        return max(first_in_quarter, self.first_t)

    def model_for(self, t):
        if self.first_t is None:
            self.first_t = t   # the first signal of the run triggers the first fit
        fit_t = self._fit_session(t)
        if fit_t not in self.models:
            model = fit(self.kind, self.panel, fit_t, self.features)
            self.models = {fit_t: model}          # keep memory bounded
            self.fit_log.append({"fit_date": self.data.calendar[fit_t], "n_train": model.n_train_,
                                 "last_label_end": self.data.calendar[int(self.panel.loc[
                                     usable_training_rows(self.panel, fit_t), "label_end"].max())]})
        return self.models[fit_t]

    def scores(self, t):
        """Predicted 90-day rank score at signal close t for every eligible stock.

        Features are ranked among all eligible stocks, exactly as in training;
        the strategy applies its own liquidity screen afterwards.
        """
        cols, X = cross_section(self.data, self.feats, t, features=self.features)
        if not len(cols):
            return cols, np.array([])
        return cols, self.model_for(t).predict(X)

    def agreement(self, t, columns):
        """Feature agreement for the given stocks at t (Ridge only).

        A stock's score is intercept + sum of per-feature contributions
        (coefficient x centred rank, plus that feature's missing-indicator
        term). Agreement = positive contributions / absolute contributions:
        1 = every feature pushes the score up, 0.5 = a tug-of-war. A ratio, so
        it does not drift when a refit shrinks or grows all coefficients.
        """
        model = self.model_for(t)
        if not hasattr(model, "contributions"):
            return np.full(len(columns), np.nan)
        cols, X = cross_section(self.data, self.feats, t, features=self.features)
        pos = pd.Series(np.arange(len(cols)), index=cols).reindex(columns).to_numpy()
        c = model.contributions(X[pos.astype(int)])
        return np.clip(c, 0, None).sum(axis=1) / np.abs(c).sum(axis=1)
