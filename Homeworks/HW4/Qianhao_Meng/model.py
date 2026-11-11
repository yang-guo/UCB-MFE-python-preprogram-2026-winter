"""Stock scores for HW4: an equal-weight rank composite and a Ridge-weighted comparison."""

import pandas as pd
from sklearn.linear_model import Ridge

from data import course


# Each component, its label, and the direction that published research found to be rewarded.
COMPONENTS = {
    "momentum_252_21": ("12-1 momentum", 1),
    "drawdown_252": ("52-week high", 1),
    "volatility_63": ("Low volatility", -1),
    "operating_roa": ("Profitability", 1),
}
RANKS = [f"rank_{name}" for name in COMPONENTS]
SCORE_KEYS = ["date", "symbol", "entry_date", "target"]


def add_ranks(panel):
    """Rank every component within its signal date so that a higher rank is better.

    A rank uses only the stocks eligible on that date, so no other date can
    change it. A missing value gets the neutral rank of 0.5.
    """
    ranked = panel.copy()
    by_date = panel.groupby("date")
    for name, (_, direction) in COMPONENTS.items():
        rank = by_date[name].rank(pct=True)
        ranked[f"rank_{name}"] = (rank if direction > 0 else 1 - rank).fillna(0.5)
    return ranked


def component_scores(ranked, name):
    """Score by one ranked component."""
    return ranked[SCORE_KEYS].assign(score=ranked[f"rank_{name}"]).reset_index(drop=True)


def composite_scores(ranked):
    """Average the four component ranks with equal weights."""
    return ranked[SCORE_KEYS].assign(score=ranked[RANKS].mean(axis=1)).reset_index(drop=True)


def fit_weights(training, alpha):
    """Let Ridge learn non-negative weights on the same ranks, clipping only the training targets."""
    limits = training.target.quantile([0.01, 0.99])
    targets = training.target.clip(lower=limits.iloc[0], upper=limits.iloc[1])
    return Ridge(alpha=alpha, positive=True).fit(training[RANKS], targets)


def walk_forward_scores(ranked, alpha, fit_dates):
    """Refit the Ridge weights at each fit date, then score every signal until the next fit.

    A fit uses only outcomes that ended before its date. Signals before the
    first fit date get no score. The audit lists what each fit saw and learned.
    """
    fit_dates = pd.DatetimeIndex(fit_dates)
    parts, audit = [], []
    for fit_date, next_fit in zip(fit_dates, [*fit_dates[1:], pd.Timestamp.max]):
        training = ranked.loc[ranked.label_end.lt(fit_date) & ranked.target.notna()]
        to_score = ranked.loc[ranked.date.ge(fit_date) & ranked.date.lt(next_fit)]
        if to_score.empty:
            continue
        assert training.label_end.max() < to_score.date.min()
        fitted = fit_weights(training, alpha)
        scored = to_score[SCORE_KEYS].copy()
        scored["score"] = fitted.predict(to_score[RANKS])
        parts.append(scored)
        weights = fitted.coef_
        shares = weights / weights.sum() if weights.sum() > 0 else weights
        audit.append({"fit_date": fit_date, "training_rows": len(training),
                      "last_label_end": training.label_end.max(),
                      "first_scored_signal": to_score.date.min(),
                      **{label: share for (label, _), share in zip(COMPONENTS.values(), shares)}})
    return pd.concat(parts, ignore_index=True), pd.DataFrame(audit)


def rank_ic_by_year(scores):
    """Average the same-date Spearman correlation of scores and completed outcomes by entry year."""
    completed = scores.loc[scores.target.notna()]
    return completed.groupby(completed.entry_date.dt.year).apply(
        lambda year: course.prediction_metrics(year, year.score)["mean_daily_rank_IC"],
        include_groups=False)


def to_rankings(scores):
    """Order each signal date's symbols from the highest score, ticker breaking ties."""
    ranked = scores.sort_values(["date", "score", "symbol"], ascending=[True, False, True])
    return ranked.groupby("date", sort=False).symbol.agg(list).to_dict()
