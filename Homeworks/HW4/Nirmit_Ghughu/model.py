"""Cross-sectional return model with walk-forward (quarterly) refits.

Preprocessing
-------------
Each numeric feature is converted to its **percentile rank within the same
signal date** (among tradable stocks), centred on 0.  This needs no fitted
state, so it cannot leak future information, and it neutralises the extreme
vendor values we saw in the raw data (e.g. 5-day "returns" of +130,000x).
Missing values stay missing: the gradient-boosting model routes them natively;
Ridge gets 0 (the cross-sectional median) plus missing indicators.

Target
------
The prediction target is the 90-calendar-day holding return from next-session
entry.  We compare two training targets in development:
  * ``raw``  - the return itself, winsorised at the training sample's 1st/99th
    percentiles (quantiles fitted on training rows only);
  * ``rank`` - the return's percentile rank within its signal date.  The strategy
    only needs to *order* stocks on a date, so this removes market-wide moves
    that no cross-sectional model can predict.

No-look-ahead fitting
---------------------
A model fitted for use from signal date ``F`` trains only on rows with
``date < F`` **and** ``label_end < F`` (the 90-day outcome had already been
observed by F's close).
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

from data import NUMERIC_FEATURES

TRAIN_START = pd.Timestamp("2010-01-01")

# Slow-moving, economically motivated signals.  Short-horizon momentum (5-126
# sessions) is dropped: in development its sign flipped between 2010-11 and
# 2012-15, so a model trained on the early years learned the wrong direction.
CORE_FEATURES = [
    "mom_252_21", "vol_63", "max_ret_21", "drawdown_252", "beta_252", "log_dollar_vol_63",
    "gross_profitability", "operating_roa", "net_margin", "debt_to_assets", "cash_to_assets",
    "fcf_to_assets", "accruals_to_assets", "asset_growth", "earnings_yield", "book_to_market",
    "filing_age_days", "log_mcap",
]


def add_rank_features(panel):
    """Percentile-rank features within each date (tradable rows only)."""
    out = panel.loc[panel.tradable].copy()
    g = out.groupby("date")
    for c in NUMERIC_FEATURES:
        out[c] = (g[c].rank(pct=True) - 0.5).astype(np.float32)
    out["target_rank"] = g["target"].rank(pct=True) - 0.5
    out["sector_code"] = out.sector.astype("category")
    return out


def _features(kind):
    return CORE_FEATURES if kind.endswith("_core") else NUMERIC_FEATURES


def _design(frame, kind):
    X = frame[_features(kind)]
    if kind.startswith("hgb"):
        X = X.assign(sector_code=frame.sector_code.cat.codes.astype(np.int16))
        return X
    # Ridge: missing -> median rank (0) + indicators, plus sector dummies.
    ind = X.isna().astype(np.float32).add_suffix("_missing")
    dummies = pd.get_dummies(frame.sector, dtype=np.float32)
    return pd.concat([X.fillna(0.0), ind, dummies], axis=1)


def make_model(kind, seed=0):
    if kind.startswith("ridge"):
        return Ridge(alpha=1000.0)
    if kind.startswith("hgb"):
        return HistGradientBoostingRegressor(
            loss="squared_error", learning_rate=0.05, max_iter=250, max_leaf_nodes=15,
            min_samples_leaf=500, l2_regularization=1.0, categorical_features=[len(_features(kind))],
            early_stopping=False, random_state=seed)
    raise ValueError(kind)


def training_rows(ranked, fit_date, thin=2):
    """Rows whose outcome finished before ``fit_date``; keep every ``thin``-th week."""
    rows = ranked.loc[(ranked.date >= TRAIN_START) & (ranked.date < fit_date)
                      & ranked.label_end.notna() & (ranked.label_end < fit_date)
                      & ranked.target.notna()]
    dates = np.sort(rows.date.unique())[::-1][::thin]       # newest week always included
    return rows.loc[rows.date.isin(dates)]


def fit(train, kind, target):
    if target == "rank":
        y = train.target_rank.to_numpy()
    else:
        lo, hi = np.nanquantile(train.target, [0.01, 0.99])  # fitted on training rows only
        y = train.target.clip(lo, hi).to_numpy()
    model = make_model(kind)
    model.fit(_design(train, kind), y)
    return model


def walk_forward_predictions(ranked, start, end, kind="hgb", target="rank", refit="QS", thin=2,
                             verbose=True):
    """Score every tradable row with signal date in [start, end].

    A new model is fitted at the first signal date of each period (quarterly by
    default) using only completed labels, then scores that period's signals.
    """
    dates = np.sort(ranked.date.unique())
    dates = dates[(dates >= np.datetime64(start)) & (dates <= np.datetime64(end))]
    periods = pd.DatetimeIndex(dates).to_period(refit[0])
    out, fits = [], []
    for period in periods.unique():
        block = dates[periods == period]
        fit_date = pd.Timestamp(block[0])
        train = training_rows(ranked, fit_date, thin)
        model = fit(train, kind, target)
        score_rows = ranked.loc[ranked.date.isin(block)]
        pred = model.predict(_design(score_rows, kind))
        out.append(score_rows[["date", "symbol", "sector", "target", "label_end"]].assign(score=pred))
        fits.append({"fit_date": fit_date, "train_rows": len(train),
                     "train_first": train.date.min(), "train_last_signal": train.date.max(),
                     "max_label_end": train.label_end.max()})
        assert train.label_end.max() < fit_date          # timing audit, every fit
        if verbose:
            print(f"  fit {fit_date.date()}  rows={len(train):>7,}  last label {train.label_end.max().date()}")
    return pd.concat(out, ignore_index=True), pd.DataFrame(fits)


def prediction_metrics(pred):
    """Pooled metrics plus per-date rank IC and top-decile excess return."""
    p = pred.dropna(subset=["target"]).copy()
    # Bad vendor ticks create 90-day "returns" of +100,000%; winsorise within each
    # date (1%/99%) so one error cannot dominate a cross-sectional average.
    g = p.groupby("date").target
    p["target"] = p.target.clip(g.transform(lambda x: x.quantile(0.01)), g.transform(lambda x: x.quantile(0.99)))
    ic = p.groupby("date").apply(lambda x: x.score.corr(x.target, method="spearman"), include_groups=False)
    def top_minus_all(x):
        cut = x.score.quantile(0.9)
        return x.loc[x.score >= cut, "target"].mean() - x.target.mean()
    spread = p.groupby("date").apply(top_minus_all, include_groups=False)
    return pd.Series({"dates": ic.notna().sum(), "mean_rank_IC": ic.mean(),
                      "IC_t_stat": ic.mean() / ic.std() * np.sqrt(ic.notna().sum() / 13),  # ~13 wk overlap
                      "top_decile_excess_90d": spread.mean(), "hit_rate_IC>0": (ic > 0).mean()})
