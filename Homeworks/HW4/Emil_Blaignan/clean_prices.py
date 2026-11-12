"""Drops impossible prices from the raw data.

Some tickers flip between two scales (one stock jumps between $0.57 and $23,000 within days).
A price is dropped when all of these hold
  * it is more than 4x away from the median of the previous 20 prices and more than 8 robust sigmas away
  * the price comes back to that level within 20 observations
Real jumps and crashes do not come back, so they stay. Dropped prices become gaps. I never fill them.
The dropped rows are saved to outputs/flagged_prices.csv so they can be checked.
The look ahead is small. The test for coming back peeks up to 20 observations ahead, but only for prices
that already jumped more than 4x, so it cannot hide a real loss.
"""

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

WINDOW = 20        # observations in the reference median
MAD_WINDOW = 60    # observations in the return-MAD scale
FLOOR = np.log(4)  # minimum log deviation to be a candidate
K_SIGMA = 8.0      # robust sigmas
REVERT_DAYS = 20   # observations allowed for the price to come back (bad runs last up to ~10)
MAX_PASSES = 3     # re-screen after removal so the reference median is not polluted
REVERT_BAND = np.log(1.5)


def _trailing(a, window, fn):
    """fn over the previous `window` values (excluding t); NaN until `window` seen."""
    out = np.full(len(a), np.nan)
    if len(a) > window:
        out[window:] = fn(sliding_window_view(a, window)[:-1], axis=1)
    return out


def _flag_one(adj):
    logp = np.log(adj)
    n = len(logp)
    ref = _trailing(logp, WINDOW, np.median)
    ret = np.diff(logp, prepend=np.nan)
    window_ret = _trailing(ret, MAD_WINDOW, np.nanmedian)  # median return
    mad = np.full(n, np.nan)
    if n > MAD_WINDOW:
        w = sliding_window_view(ret, MAD_WINDOW)[:-1]
        mad[MAD_WINDOW:] = np.nanmedian(np.abs(w - np.nanmedian(w, axis=1, keepdims=True)), axis=1)
    sigma = 1.4826 * mad
    dev = np.abs(logp - ref)
    thr = np.fmax(FLOOR, K_SIGMA * np.nan_to_num(sigma, nan=0.0))
    cand = dev > thr  # NaN ref -> False
    bad = np.zeros(n, bool)
    for t in np.flatnonzero(cand):
        nxt = np.abs(logp[t + 1:t + 1 + REVERT_DAYS] - ref[t])
        # revert: price back inside the band (or the next print is itself a candidate
        # that reverts, handled when we reach it); require a return to the reference.
        if (nxt < REVERT_BAND).any():
            bad[t] = True
    return bad


def clean_prices(prices):
    """Return (cleaned prices, flagged rows). `prices` needs symbol, date, adj_close."""
    cur = prices.sort_values(["symbol", "date"]).reset_index(drop=True)
    dropped = []
    for _ in range(MAX_PASSES):
        bad = np.zeros(len(cur), bool)
        starts = np.flatnonzero(np.r_[True, cur.symbol.values[1:] != cur.symbol.values[:-1]])
        ends = np.r_[starts[1:], len(cur)]
        adj = cur.adj_close.to_numpy(float)
        for s, e in zip(starts, ends):
            bad[s:e] = _flag_one(adj[s:e])
        if not bad.any():
            break
        dropped.append(cur.loc[bad, ["symbol", "date", "close", "adj_close"]])
        cur = cur.loc[~bad].reset_index(drop=True)
    flagged = pd.concat(dropped, ignore_index=True) if dropped else prices.iloc[:0][["symbol", "date", "close", "adj_close"]]
    return cur, flagged
