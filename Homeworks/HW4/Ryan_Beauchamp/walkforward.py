"""Shared walk-forward fitting and scoring (used by development and the final run).

One code path for both, so the final test uses exactly the procedure chosen in
development.

* ``load_model_rows``: eligible rows that pass the causal clean flag, only the
  columns the model needs, read with parquet filters (bounded memory).
* ``fit_dates``: refit schedule. "once" = a single fit at the start; "annual" /
  "quarterly" = refit at the first session of each later year / quarter.
* ``walk_forward_scores``: at each fit date f, train on rows with
  date >= train_start, a known target, and label_end < f (the outcome had
  finished before the fit). Score every clean eligible row with
  f <= date < next fit date, whether or not its target is known.

Target: each date's cross-sectional percentile rank of the 90-day return,
centred at 0 (stage 2c). Model: Ridge in the lecture pipeline.

Credits: make_pipeline and FEATURES come from Lectures/modeling/course_data.py
(imported, unmodified). Written with AI assistance (Claude Code) and reviewed
by the author.
"""

import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "Lectures" / "modeling"))
import course_data as cd  # noqa: E402

MAX_LOG_MOVE = np.log(5)   # clean flag: largest |daily log move| over trailing 252 sessions
MIN_CLOSE = 1.0            # clean flag: supplied close floor
MODEL_COLUMNS = ["date", "symbol", "target", "label_end"] + cd.FEATURES


def load_model_rows(panel_dir, measures_path, start, end, extra_path=None, extra_columns=None):
    """Clean eligible rows with signal dates in [start, end].

    Optional extra features (e.g. GARP) are left-merged from their own parquet
    file on (date, symbol); rows without them keep missing values.
    """
    frame = pd.read_parquet(panel_dir, columns=MODEL_COLUMNS,
                            filters=[("date", ">=", pd.Timestamp(start)), ("date", "<=", pd.Timestamp(end)),
                                     ("eligible", "==", True)])
    measures = pd.read_parquet(measures_path, columns=["symbol", "date", "close", "max_abs_log_move_252"],
                               filters=[("date", ">=", pd.Timestamp(start)), ("date", "<=", pd.Timestamp(end))])
    frame = frame.merge(measures, on=["symbol", "date"], how="left", validate="one_to_one")
    broken = frame.max_abs_log_move_252.gt(MAX_LOG_MOVE) | frame.close.lt(MIN_CLOSE)
    frame = frame.loc[~broken].drop(columns=["close", "max_abs_log_move_252"])
    if extra_path is not None and extra_columns:
        extra = pd.read_parquet(extra_path, columns=["date", "symbol"] + list(extra_columns),
                                filters=[("date", ">=", pd.Timestamp(start)), ("date", "<=", pd.Timestamp(end))])
        frame = frame.merge(extra, on=["date", "symbol"], how="left", validate="one_to_one")
    return frame.sort_values(["date", "symbol"]).reset_index(drop=True), int(broken.sum())


def fit_dates(sessions, start, end, rule):
    """First session on/after `start`, then the first session of each later period up to `end`."""
    sessions = pd.DatetimeIndex(sessions)
    sessions = sessions[(sessions >= pd.Timestamp(start)) & (sessions <= pd.Timestamp(end))]
    if rule == "once":
        return pd.DatetimeIndex([sessions[0]])
    freq = {"annual": "Y", "quarterly": "Q"}[rule]
    periods = sessions.to_period(freq)
    firsts = pd.Series(sessions, index=periods).groupby(level=0).min()
    dates = pd.DatetimeIndex(firsts.to_numpy())
    return dates.union(pd.DatetimeIndex([sessions[0]]))


def date_rank(dates, values):
    """Cross-sectional percentile rank per date, centred at 0."""
    return pd.Series(values).groupby(np.asarray(dates)).rank(pct=True).to_numpy() - 0.5


def walk_forward_scores(frame, fits, score_end, alpha=1000.0, train_start="2010-01-01", log=print,
                        numeric_features=None, include_sector=True):
    """Return (scores, fit_log). scores: date, symbol, score, fit_date.

    Defaults = the lecture feature set (course_data.FEATURES: numeric features + sector).
    """
    numeric = list(cd.NUMERIC_FEATURES if numeric_features is None else numeric_features)
    columns = numeric + (["sector"] if include_sector else [])
    fits = pd.DatetimeIndex(fits).sort_values()
    score_end = pd.Timestamp(score_end)
    pieces, fit_log = [], []
    for k, fit in enumerate(fits):
        t = time.perf_counter()
        next_fit = fits[k + 1] if k + 1 < len(fits) else score_end + pd.Timedelta(days=1)
        train = frame.loc[frame.date.ge(pd.Timestamp(train_start)) & frame.target.notna()
                          & frame.label_end.lt(fit)]
        assert len(train) and train.label_end.max() < fit and train.date.max() < fit
        model = cd.make_pipeline(Ridge(alpha=alpha), numeric_features=numeric, include_sector=include_sector)
        model.fit(train[columns], date_rank(train.date, train.target.to_numpy()))
        rows = frame.loc[frame.date.ge(fit) & frame.date.lt(next_fit) & frame.date.le(score_end)]
        assert rows.date.min() >= fit
        pieces.append(rows[["date", "symbol"]].assign(score=model.predict(rows[columns]), fit_date=fit))
        fit_log.append({"fit_date": fit.date(), "train_rows": len(train),
                        "last_train_signal": train.date.max().date(),
                        "last_train_label_end": train.label_end.max().date(),
                        "scored_from": rows.date.min().date() if len(rows) else None,
                        "scored_to": rows.date.max().date() if len(rows) else None,
                        "scored_rows": len(rows), "seconds": round(time.perf_counter() - t, 1)})
        log(f"    fit {fit.date()}: {len(train):,} training rows (last label_end "
            f"{train.label_end.max().date()}), scored {len(rows):,} rows, {time.perf_counter() - t:.0f}s")
    return pd.concat(pieces, ignore_index=True), pd.DataFrame(fit_log)


def scores_by_date(scores):
    return {d: g.set_index("symbol").score for d, g in scores.groupby("date")}
