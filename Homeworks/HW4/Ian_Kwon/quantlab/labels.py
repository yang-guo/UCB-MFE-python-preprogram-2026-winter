"""Prediction targets aligned with the backtester's execution rule.

Signal after close t  ->  entry at the close of the next market session e
->  exit at the first *quoted* close on/after e + 90 calendar days.
This mirrors ``course_data.make_labels``.  Same conventions: a target is left
missing (right-censored) if the entry session has no quote, if no exit quote
arrives within 5 sessions of maturity, or if maturity lies beyond the loaded
data.  Missing targets are not investment filters.

``label_end`` (the exit date) is when the target becomes *known*.  A model
fitted after the close of F may use a row only if label_end < F.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MAX_EXIT_DELAY = 5


def next_quote_index(px: np.ndarray) -> np.ndarray:
    """nq[t, s] = first session index >= t with a quote for s, or T if none."""
    T = px.shape[0]
    idx = np.where(np.isfinite(px), np.arange(T)[:, None], T).astype(np.int32)
    return np.minimum.accumulate(idx[::-1], axis=0)[::-1]


def label_arrays(px: np.ndarray, calendar: pd.DatetimeIndex, signal_rows: np.ndarray,
                 holding_days: int, unreliable_day: np.ndarray | None = None):
    """Targets for every (signal row, symbol).  Returns dict of [len(rows), N] arrays."""
    T, N = px.shape
    nq = next_quote_index(px)
    rows = np.asarray(signal_rows)
    entry_idx = rows + 1
    has_entry_session = entry_idx < T
    entry_idx_c = np.minimum(entry_idx, T - 1)
    entry_dates = calendar[entry_idx_c]
    due = entry_dates + pd.Timedelta(days=holding_days)
    due_idx = calendar.searchsorted(due)                      # first session >= due (T if beyond data)
    in_data = has_entry_session & (due_idx < T)
    exit_idx = nq[np.minimum(due_idx, T - 1)]                 # first quoted session >= due
    ok_exit = in_data[:, None] & (exit_idx < T) & (exit_idx - due_idx[:, None] <= MAX_EXIT_DELAY)
    entry_px = px[entry_idx_c]
    entry_px = np.where(has_entry_session[:, None], entry_px, np.nan)
    cols = np.arange(N)[None, :]
    exit_px = px[np.minimum(exit_idx, T - 1), cols]
    valid = ok_exit & np.isfinite(entry_px) & np.isfinite(exit_px)
    target = np.where(valid, exit_px / entry_px - 1, np.nan)
    label_end = np.where(valid, exit_idx, -1)
    out = {"target": target, "label_end_idx": label_end, "entry_idx": np.broadcast_to(entry_idx[:, None], (len(rows), N)).copy()}
    if unreliable_day is not None:
        cs = np.cumsum(unreliable_day, axis=0, dtype=np.int32)
        hit = cs[np.minimum(exit_idx, T - 1), cols] - cs[entry_idx_c]
        out["label_has_unreliable_break"] = valid & (hit > 0)
    return out
