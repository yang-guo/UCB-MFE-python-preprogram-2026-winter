"""HW4: walk-forward scores with the pre-registered quarterly expanding refit.

At each fit date F (first market session of every calendar quarter):
    train = eligible rows with signal date >= 2010-01-01 and label_end < F
            (only outcomes whose 90-day window has ENDED before F)
    model = the frozen pipeline (cd.make_pipeline: clip -> impute -> scale -> model),
            fitted on train with the per-date rank target
    score = eligible rows with F <= date < next F
The same code produces development scores (to tune sizing) and test scores (final run).
"""

import numpy as np
import pandas as pd

import hw4_data as hd
import hw4_model as hm

cd = hd.cd
TRAIN_START = "2010-01-01"


def quarter_fit_dates(sessions, start, end):
    """First market session of each calendar quarter inside [start, end]."""
    s = pd.Series(pd.DatetimeIndex(sessions))
    s = s[(s >= pd.Timestamp(start)) & (s <= pd.Timestamp(end))]
    return pd.DatetimeIndex(s.groupby(s.dt.to_period("Q")).min().to_numpy())


def walk_forward_scores(panel, model_name, feature_set, start, end):
    """Return DataFrame(date, symbol, score, fit_date) for eligible rows in [start, end]."""
    feats = hm.FEATURE_SETS[feature_set]
    elig = panel.loc[panel.eligible].reset_index(drop=True)
    fits = quarter_fit_dates(elig.date.unique(), start, end)
    bounds = list(fits[1:]) + [pd.Timestamp(end) + pd.Timedelta(days=1)]
    pieces = []
    for fit, nxt in zip(fits, bounds):
        train = elig.loc[elig.date.ge(TRAIN_START) & elig.target.notna() & elig.label_end.lt(fit)]
        score = elig.loc[elig.date.ge(fit) & elig.date.lt(nxt)]
        if train.empty or score.empty:
            continue
        assert train.label_end.max() < fit <= score.date.min()      # no unfinished outcome used
        train = hm.add_target_rank(train)
        pipe = cd.make_pipeline(hm.make_model(model_name), feats).fit(train, train.target_rank)
        pieces.append(pd.DataFrame({"date": score.date.to_numpy(), "symbol": score.symbol.to_numpy(),
                                    "score": pipe.predict(score), "fit_date": fit}))
    return pd.concat(pieces, ignore_index=True)


if __name__ == "__main__":
    # Usage: hw4_walkforward.py <cache> <model> <feature_set> <start> <end>
    import sys
    import hw4_build_cache as hc
    name, model_name, fs, start, end = sys.argv[1:6]
    cols = ["date", "symbol", "eligible", "target", "label_end", "sector"] + hm.FEATURE_SETS[fs]
    panel = pd.read_parquet(hc.cache_path(name), columns=cols)
    scores = walk_forward_scores(panel, model_name, fs, start, end)
    out = hc.OUT / f"scores_{name}_{model_name}_{fs}.parquet"
    scores.to_parquet(out, index=False)
    print(f"{len(scores):,} scores, {scores.fit_date.nunique()} fits, "
          f"{scores.date.min().date()}..{scores.date.max().date()} -> {out}")
