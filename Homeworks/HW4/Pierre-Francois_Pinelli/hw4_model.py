"""HW4 model: elastic net of the 90-day return on 18 ranked characteristics.

1. Each month-end, we rank every characteristic across the stocks, from -0.5 to +0.5.
2. Target = 90-day return minus the average 90-day return of the same month.
3. Elastic net (linear regression with L1 + L2 penalties), refit every quarter, only on outcomes already over.
4. Score = predicted 90-day return vs the average stock. We buy the highest scores.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet

from hw4_data import FEATURES

Z_COLS = ["z_" + f for f in FEATURES]


def add_ranks(panel):
    """Rank each characteristic among the stocks of the same date, between -0.5 and 0.5.

    Missing = 0 (the middle). We only use the same date, so no look-ahead here.
    """
    panel = panel.copy()
    ok = panel.eligible
    by_date = panel[ok].groupby("date")[FEATURES]
    n = by_date.transform("count")
    z = (by_date.rank() - 1) / (n - 1).where(n > 1) - 0.5  # lowest = -0.5, highest = +0.5
    for f in FEATURES:
        panel.loc[ok, "z_" + f] = z[f].fillna(0.0)
    return panel


def clip_by_date(df, column="target"):
    """Cut the 1% and 99% tails of each date, so a few crazy returns don't drive the fit."""
    g = df.groupby("date")[column]
    return df[column].clip(g.transform(lambda s: s.quantile(0.01)), g.transform(lambda s: s.quantile(0.99)))


def walk_forward(panel, fit_dates, score_dates, alpha, l1_ratio):
    """Each fit date we refit on the past, then we score the months until the next refit.

    Also returns an audit table per refit, like fit_audit in Lecture 8.2.
    """
    usable = panel[panel.eligible]
    scores, models, audit = [], {}, []
    for i, fit_date in enumerate(fit_dates):
        next_fit = fit_dates[i + 1] if i + 1 < len(fit_dates) else pd.Timestamp.max
        # train on 2010+ rows whose 90 days are already over at the fit date
        train = usable[(usable.date >= "2010-01-01") & (usable.label_end < fit_date) & usable.target.notna()]
        assert train.label_end.max() < fit_date  # no future outcome in the training set
        y = clip_by_date(train)
        y = y - y.groupby(train.date).transform("mean")  # return minus the average stock of that month
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10000).fit(train[Z_COLS], y)
        # score the month-ends between this fit and the next one
        test = usable[usable.date.isin([d for d in score_dates if fit_date <= d < next_fit])]
        scores.append(test[["date", "symbol"]].assign(score=model.predict(test[Z_COLS])))
        models[fit_date] = model
        # what each model used and what it scored (Lecture 8.2)
        audit.append({"fit_date": fit_date, "rows": len(train), "last_training_signal": train.date.max(),
                      "last_label_end": train.label_end.max(), "first_scored_signal": test.date.min(),
                      "last_scored_signal": test.date.max()})
    return pd.concat(scores, ignore_index=True), models, pd.DataFrame(audit)


def ranking_metrics(scores, panel, top_n=30):
    """Does the score rank the stocks well? Rank IC per date (like in the course) and top N minus all."""
    df = scores.merge(panel[["date", "symbol", "target"]], on=["date", "symbol"]).dropna(subset=["target"])
    df["target"] = clip_by_date(df)
    rows = []
    for d, g in df.groupby("date"):
        top = g.sort_values(["score", "symbol"], ascending=[False, True]).head(top_n)
        rows.append({"date": d, "ic": g.score.corr(g.target, method="spearman"),
                     "top_minus_all": top.target.mean() - g.target.mean()})
    by_date = pd.DataFrame(rows).set_index("date")
    ic = by_date.ic.dropna()
    summary = pd.Series({"mean_rank_IC": ic.mean(), "IC_t_stat": ic.mean() / ic.std() * np.sqrt(len(ic)),
                         "share_IC_positive": (ic > 0).mean(), f"top{top_n}_minus_all_90d": by_date.top_minus_all.mean(),
                         "dates": len(by_date)})
    return summary, by_date
