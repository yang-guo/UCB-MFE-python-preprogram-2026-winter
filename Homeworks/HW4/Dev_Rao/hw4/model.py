"""Models, training targets, purged walk-forward folds and ranking metrics.

`QuantileClipper` and `make_ridge_pipeline` are adapted from
`Lectures/modeling/course_data.py` (`QuantileClipper`, `make_pipeline`): clip
tails -> median impute + missing indicators -> scale, all fitted on training
rows only inside the sklearn Pipeline. `purged_train_mask` follows
`course_data.date_folds`: a model fitted at date F trains only on rows with
signal date >= 2010-01-01 and label_end < F.
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted, validate_data

from . import config
from .features import MODEL_CATEGORICAL, MODEL_NUMERIC


# --------------------------------------------------------------------------------
# Pipelines
# --------------------------------------------------------------------------------
class QuantileClipper(TransformerMixin, OneToOneFeatureMixin, BaseEstimator):
    """Per-column tail cutoffs learned on training rows; NaN preserved.
    (Copied from course_data.QuantileClipper.)"""

    def __init__(self, lower=0.01, upper=0.99):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        a = validate_data(self, X, reset=True, dtype=float, ensure_all_finite="allow-nan")
        self.lower_bounds_ = np.full(a.shape[1], -np.inf)
        self.upper_bounds_ = np.full(a.shape[1], np.inf)
        observed = np.isfinite(a).any(axis=0)
        self.lower_bounds_[observed] = np.nanquantile(a[:, observed], self.lower, axis=0)
        self.upper_bounds_[observed] = np.nanquantile(a[:, observed], self.upper, axis=0)
        return self

    def transform(self, X):
        check_is_fitted(self, "lower_bounds_")
        a = validate_data(self, X, reset=False, dtype=float, ensure_all_finite="allow-nan")
        return np.clip(a, self.lower_bounds_, self.upper_bounds_)


def _sector_encoder():
    return OneHotEncoder(handle_unknown="ignore", sparse_output=False)


def make_ridge_pipeline(alpha=100.0):
    """course_data.make_pipeline with my feature lists."""
    numeric = Pipeline([
        ("clip", QuantileClipper()),
        ("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
        ("scale", StandardScaler()),
    ])
    pre = ColumnTransformer([("numeric", numeric, MODEL_NUMERIC),
                             ("sector", _sector_encoder(), MODEL_CATEGORICAL)], remainder="drop")
    return Pipeline([("preprocess", pre), ("model", Ridge(alpha=alpha))])


def make_hgb_pipeline(max_iter=200, max_leaf_nodes=15, min_samples_leaf=2000,
                      learning_rate=0.05, l2_regularization=1.0, random_state=0):
    """Trees need no scaling/imputation: NaN is routed natively at each split."""
    pre = ColumnTransformer([("numeric", "passthrough", MODEL_NUMERIC),
                             ("sector", _sector_encoder(), MODEL_CATEGORICAL)], remainder="drop")
    hgb = HistGradientBoostingRegressor(
        learning_rate=learning_rate, max_iter=max_iter, max_leaf_nodes=max_leaf_nodes,
        min_samples_leaf=min_samples_leaf, l2_regularization=l2_regularization,
        early_stopping=False, random_state=random_state)
    return Pipeline([("preprocess", pre), ("model", hgb)])


def make_model(spec):
    kind, params = spec["kind"], {k: v for k, v in spec.items() if k not in ("kind", "target", "name")}
    return make_ridge_pipeline(**params) if kind == "ridge" else make_hgb_pipeline(**params)


# Fixed-sign multi-factor composite (no fitting). Signs come from the published
# literature, not from this data: value (Fama-French 1992; Lakonishok-Shleifer-
# Vishny 1994), profitability/quality (Novy-Marx 2013; Sloan 1996 accruals),
# investment/issuance (Cooper-Gulen-Schill 2008; Pontiff-Woodgate 2008),
# momentum (Jegadeesh-Titman 1993; Moskowitz-Grinblatt 1999 industry momentum),
# earnings momentum (Bernard-Thomas 1989), low risk / lottery (Ang et al. 2006;
# Bali-Cakici-Whitelaw 2011). Each theme is the mean of its signed ranks, and
# the score is the mean of the available themes (equal weight per theme).
COMPOSITE_THEMES = {
    "value": {"r_ey": 1, "r_bm": 1, "r_sp": 1, "r_fcfy": 1},
    "quality": {"r_gp_a": 1, "r_roe": 1, "r_roa": 1, "r_accruals_a": -1},
    "investment": {"r_asset_growth": -1, "r_share_iss": -1},
    "momentum": {"r_mom_12_1": 1, "r_sector_mom_126": 1, "r_ni_chg_a": 1},
    "low_risk": {"r_ivol_63": -1, "r_max_ret_21": -1},
}


def composite_score(frame):
    themes = []
    for spec in COMPOSITE_THEMES.values():
        parts = [(frame[c] - 0.5) * s for c, s in spec.items()]
        themes.append(pd.concat(parts, axis=1).mean(axis=1))
    return pd.concat(themes, axis=1).mean(axis=1).fillna(0.0).to_numpy()


# --------------------------------------------------------------------------------
# Training targets (evaluation always uses the raw forward return)
# --------------------------------------------------------------------------------
def training_target(train, variant):
    """Build y from TRAINING rows only.

    rank:    cross-sectional percentile rank of the forward return per date - 0.5
    demean:  forward return minus the date mean, winsorised at the 1st/99th
             percentiles of THESE training rows (so cutoffs never see the future)
    resid:   rank of the beta-residual return: target - beta_252 * benchmark
             return over the SAME entry->exit window (beta known at the signal;
             the benchmark leg is realised exactly when the label is)
    """
    y = train.target.astype("float64")
    if variant == "rank":
        return (y.groupby(train.date).rank(pct=True) - 0.5).to_numpy()
    if variant == "resid":
        beta = train.beta_252.astype("float64").fillna(1.0).clip(-1, 4)
        e = y - beta * train.bench_fwd
        return (e.groupby(train.date).rank(pct=True) - 0.5).to_numpy()
    if variant == "demean":
        d = y - y.groupby(train.date).transform("mean")
        lo, hi = np.nanquantile(d, [0.01, 0.99])
        return d.clip(lo, hi).to_numpy()
    raise ValueError(variant)


def purged_train_mask(panel, fit_date, train_start=config.TRAIN_START):
    """Rows a model fitted at `fit_date` may learn from."""
    fit_date = pd.Timestamp(fit_date)
    m = (panel.date.ge(train_start) & panel.date.lt(fit_date)
         & panel.label_end.lt(fit_date) & panel.target.notna())
    return m.to_numpy()


def fit_predict(panel, spec, fit_date, score_mask, sample_every=1):
    """Fit on purged rows before fit_date, predict rows in score_mask.

    sample_every=2 keeps every other weekly signal date for training (bi-weekly
    sampling) to bound RAM/time on the full sample; documented in the README.
    """
    tr = purged_train_mask(panel, fit_date)
    if sample_every > 1:
        dates = np.sort(panel.loc[tr, "date"].unique())
        keep = set(dates[::sample_every])
        tr &= panel.date.isin(keep).to_numpy()
    train = panel.loc[tr]
    # Timing evidence: every training outcome ended before the fit date, and no
    # scored row is before the fit date.
    assert train.label_end.max() < pd.Timestamp(fit_date)
    assert panel.loc[score_mask, "date"].min() >= pd.Timestamp(fit_date)
    if spec["kind"] == "composite":           # nothing is fitted
        return composite_score(panel.loc[score_mask]), None, 0
    model = make_model(spec)
    model.fit(train[MODEL_NUMERIC + MODEL_CATEGORICAL], training_target(train, spec["target"]))
    pred = model.predict(panel.loc[score_mask, MODEL_NUMERIC + MODEL_CATEGORICAL])
    return pred, model, len(train)


def rank_average(frame, columns):
    """Ensemble: average of each model's same-date percentile rank."""
    return frame.groupby("date")[columns].rank(pct=True).mean(axis=1)


# --------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------
def newey_west_t(x, lags=13):
    """t-stat of the mean with Newey-West SE. 90-day labels on weekly signals
    overlap by ~12 weeks, so naive t-stats overstate significance."""
    x = np.asarray(pd.Series(x).dropna(), dtype=float)
    n = len(x)
    if n < 3:
        return np.nan
    e = x - x.mean()
    s = e @ e / n
    for k in range(1, min(lags, n - 1) + 1):
        s += 2 * (1 - k / (lags + 1)) * (e[k:] @ e[:-k]) / n
    return x.mean() / np.sqrt(s / n)


def ranking_report(frame, score="score", top_n=25):
    """Weekly rank IC, decile spread/monotonicity and top-N statistics.

    frame needs date, target (raw forward return) and the score column; rows
    without a realised target are excluded from these predictive metrics.
    """
    f = frame.loc[frame.target.notna(), ["date", "target", score]].copy()
    ic = f.groupby("date").apply(lambda g: stats.spearmanr(g[score], g.target)[0]
                                 if len(g) > 20 else np.nan, include_groups=False)
    f["excess"] = f.target - f.groupby("date").target.transform("mean")
    f["decile"] = f.groupby("date")[score].transform(
        lambda s: pd.qcut(s.rank(method="first"), 10, labels=False) + 1)
    dec = f.groupby(["date", "decile"]).excess.mean().unstack().mean()
    top = f.loc[f.groupby("date")[score].rank(ascending=False, method="first") <= top_n]
    out = {
        "mean_IC": ic.mean(), "IC_std": ic.std(), "IC_t_naive": ic.mean() / ic.std() * np.sqrt(ic.count()),
        "IC_t_NW13": newey_west_t(ic), "IC_hit": (ic > 0).mean(), "n_dates": int(ic.count()),
        "top_decile_excess": dec.get(10, np.nan), "bottom_decile_excess": dec.get(1, np.nan),
        "decile_monotonicity": stats.spearmanr(dec.index, dec.values)[0],
        f"top{top_n}_mean_ret": top.target.mean(), f"top{top_n}_excess": top.excess.mean(),
        f"top{top_n}_hit": (top.excess > 0).mean(), "universe_mean_ret": f.target.mean(),
    }
    return out, ic, dec
