"""Historical entry screening for the documented top-500 research revision.

This module never fits the predictor or edits quotes. It joins a signal to the
universe for its next-session ENTRY quarter. No future outcomes, current active
flags, final coverage dates or security-name exceptions enter the decision.
The backtester receives fewer new-entry forecasts but retains all held prices.
"""

import numpy as np
import pandas as pd


def annotate_entry_ranks(scores, membership, calendar):
    """Attach the supplied dated rank, checking that it was known at the signal.

    Parameters
    ----------
    scores : DataFrame
        Unique (date, symbol) forecasts. Optional target/label columns are
        carried through unchanged and never determine eligibility.
    membership : DataFrame
        Supplied symbol, quarter_start, selection_date, effective_start and
        universe_rank. The ranking is a vendor capitalization proxy, not a
        certified historical market cap or S&P index membership list.
    calendar : DatetimeIndex
        Supplied market sessions, including the signal and next-session entry.
    """
    left = scores.copy()
    required = [
        "symbol",
        "quarter_start",
        "selection_date",
        "effective_start",
        "universe_rank",
    ]
    if not {"date", "symbol", "score"}.issubset(left.columns):
        raise ValueError("Missing forecast columns")
    if set(required[1:] + ["entry_date"]) & set(left.columns):
        raise ValueError("Forecasts already contain entry-rank metadata")
    right = membership[required].copy()
    left["date"] = pd.to_datetime(left.date).astype("datetime64[ns]")
    for name in ["quarter_start", "selection_date", "effective_start"]:
        right[name] = pd.to_datetime(right[name]).astype("datetime64[ns]")
    if (
        left[["date", "symbol"]].isna().any().any()
        or left.duplicated(["date", "symbol"]).any()
    ):
        raise ValueError("Forecast keys must be unique and present")
    if right.duplicated(["symbol", "quarter_start"]).any():
        raise ValueError("Duplicate membership keys")
    ranks = right.universe_rank.to_numpy(dtype=float)
    if not (np.isfinite(ranks) & (ranks >= 1) & (ranks == np.floor(ranks))).all():
        raise ValueError("Membership ranks must be positive integers")
    calendar = pd.DatetimeIndex(calendar)
    if calendar.has_duplicates or not calendar.is_monotonic_increasing:
        raise ValueError("Calendar must be unique and sorted")
    positions = calendar.get_indexer(left.date)
    if (positions < 0).any() or (positions + 1 >= len(calendar)).any():
        raise ValueError("Each signal needs its next market session")
    left["entry_date"] = calendar.take(positions + 1).to_numpy()
    left["quarter_start"] = left.entry_date.dt.to_period("Q").dt.start_time
    joined = left.merge(
        right,
        on=["symbol", "quarter_start"],
        how="left",
        sort=False,
        validate="many_to_one",
    )
    if joined[required[1:]].isna().any().any():
        raise ValueError("A forecast lacks its historical entry membership")
    if not joined.selection_date.le(joined.date).all():
        raise ValueError("Membership was not yet known at the signal")
    if not joined.effective_start.le(joined.entry_date).all():
        raise ValueError("Membership was not yet effective at entry")
    return joined


def select_entry_universe(annotated, max_rank=500):
    """Keep a uniform historical rank cutoff; do not touch forecasts or targets."""
    if isinstance(max_rank, bool) or not isinstance(max_rank, int) or max_rank < 1:
        raise ValueError("Rank cutoff must be a positive integer")
    return annotated.loc[annotated.universe_rank.le(max_rank)].copy()
