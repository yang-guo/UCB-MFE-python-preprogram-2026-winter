"""Scores: one number per stock per signal date, higher means buy first.

Three candidates get compared in development:
  momentum   just the 12-1 momentum percentile
  blend      simple average of the six factor percentiles
  ridge      Ridge fit on those percentiles, refit once a year

Percentiles are taken among the stocks tradable on that one date. Other
dates never enter the calculation, so a future row can't leak into a score.
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge

# sign tells which end is good: +1 high values preferred, -1 low values preferred
TILTS = {
    "momentum_252_21": +1,   # price change from 12 months back to 1 month back
    "trend_6m": +1,          # 126 session return
    "volatility_63": -1,     # quieter is better
    "drawdown_252": +1,      # near the yearly high
    "operating_roa": +1,     # operating income over assets
    "net_margin": +1,        # net income over revenue
}
PCT_COLS = ["pct_" + f for f in TILTS]
HISTORY_FLOOR = pd.Timestamp("2010-01-01")


def percentile_view(panel):
    """Tradable rows only, with each factor turned into a 0 to 1 percentile
    for its own date. A blank factor sits in the middle at 0.5."""
    view = panel[panel.tradable].copy()
    by_day = view.groupby("date")
    for factor, sign in TILTS.items():
        p = by_day[factor].rank(pct=True)
        if sign < 0:
            p = 1 - p
        view["pct_" + factor] = p.fillna(0.5)
    # what the model learns: did the stock beat that date's average tradable stock
    view["beat_peers"] = view.target - by_day.target.transform("mean")
    return view


def fixed_rule_score(view, rule):
    if rule == "momentum":
        s = view["pct_momentum_252_21"]
    else:
        s = view[PCT_COLS].mean(axis=1)
    out = view[["date", "symbol"]].copy()
    out["score"] = s.to_numpy()
    return out


def rolling_ridge(view, refits, alpha, end=None):
    """Refit Ridge on each date in `refits`, using only rows whose 90 day
    outcome was already known (label_end before the refit), and use that fit
    until the next refit. Also returns the weights so I can see them move."""
    refits = sorted(pd.Timestamp(d) for d in refits)
    stop = (pd.Timestamp(end) if end else view.date.max()) + pd.Timedelta(days=1)
    scored_parts, fit_notes = [], []
    for refit_day, next_refit in zip(refits, refits[1:] + [stop]):
        known = view[(view.date >= HISTORY_FLOOR) & (view.label_end < refit_day) & view.target.notna()]
        to_score = view[(view.date >= refit_day) & (view.date < next_refit)]
        if len(known) == 0 or len(to_score) == 0:
            continue
        reg = Ridge(alpha=alpha).fit(known[PCT_COLS], known.beat_peers)
        part = to_score[["date", "symbol"]].copy()
        part["score"] = reg.predict(to_score[PCT_COLS])
        scored_parts.append(part)
        note = {"refit_day": refit_day, "n_train": len(known), "newest_outcome": known.label_end.max()}
        note.update({"w_" + f: w for f, w in zip(TILTS, reg.coef_)})
        fit_notes.append(note)
    return pd.concat(scored_parts).reset_index(drop=True), pd.DataFrame(fit_notes)


def january_firsts(view, years):
    """First signal date on or after Jan 1 of each year."""
    all_days = np.sort(view.date.unique())
    return [pd.Timestamp(all_days[all_days >= np.datetime64(f"{y}-01-01")][0]) for y in years]


def score_ic(view, scores):
    """Average over dates of the rank correlation between score and the
    actual 90 day return. Dates with 10 or fewer stocks are left out."""
    both = scores.merge(view[["date", "symbol", "target"]], on=["date", "symbol"]).dropna()
    daily_ic = {}
    for day, g in both.groupby("date"):
        daily_ic[day] = spearmanr(g.score, g.target).statistic if len(g) > 10 else np.nan
    daily_ic = pd.Series(daily_ic)
    return daily_ic.mean(), daily_ic


def by_signal_date(scores):
    return {day: g.set_index("symbol").score for day, g in scores.groupby("date")}
