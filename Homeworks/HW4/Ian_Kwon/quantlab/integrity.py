"""Price-integrity audit: classify one-day price breaks instead of deleting them.

Evidence found in ``equities_top2000.duckdb`` (see README "Data quality"):

* **Bad prints / interleaved series.** Some tickers alternate between two
  price levels (AAM: $0.19 <-> $26,000; COEP: $10 <-> $200).  A move that is
  undone within a few sessions is not an economic return.
* **Unadjusted splits.** ``adj_close`` is not split-adjusted for many splits
  (NFLX 2025-10-03 x0.099, ORLY x0.067, BKNG x0.040).  Reported shares cannot
  corroborate them because the vendor back-adjusts fundamentals.
* **Genuine large moves** (trial failures, buyouts, 2008) look similar.

Classification of a break on day d (|log gross| > log 1.8 vs the previous quote):

``bad_print``   the pre-break level returns (within 20%) within 5 quotes:
                the segment is quarantined (set to NaN, never filled).
``split``       not a bad print; gross within 2.5% (log) of a standard ratio
                1/k or k, AND event-day dollar volume <= 3x its median over
                the prior 20 sessions.  Calibrated on research data only:
                known splits (ALXN, BLL, CERN, COG, SJI, UFS, SWM, LVLT) have
                ratios 0.7-2.0; genuine crashes at split-like ratios (MGI
                2008-01-15, AVEO 2013-05-02, CPE 2008-11-28, VSTM, CYRX) have
                4.4-383, because panic trading is heavy while a split day
                trades normal dollar volume.  A price-level test was rejected:
                ``close`` is on the vendor's *current* split basis, so old
                prices are not contemporaneous.  Repaired by back-adjusting
                earlier prices by the ratio, as a broker scales share counts.
``level_shift`` not a bad print or split; |move| beyond x5 or x0.2 on event-day
                dollar volume <= 3x normal (e.g. LCI 2013-12-13 x16.9 at 1.9x
                volume, never reverting).  A genuine 5x move cannot trade on
                normal volume; genuine extreme moves here show 17x-3,600x
                (CMRX, NLTX, HGEN).  Neutralized: the day's return is set to 0
                (back-adjusted by the observed gross) and the stock becomes
                ineligible for new entries for a year.
``unclassified`` everything else: kept as observed and flagged.

Timing: classifying day d uses quotes up to 5 sessions after d.  Therefore
any stock with a break in the last ``COOLOFF_SESSIONS`` sessions is ineligible
for new entries (see ``features``), so no trading decision ever depends on a
classification that was not yet knowable.  Valuation of an already-held lot
uses the cleaned series; that is a correction of data errors, not a decision.
"""

from __future__ import annotations

from dataclasses import dataclass

import warnings

import numpy as np
import pandas as pd

BREAK_GROSS = 1.8
SPLIT_FACTORS = (2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 25, 30, 40, 50, 100)
SPLIT_LOG_TOL = np.log(1.025)
SPLIT_MAX_DOLLAR_VOLUME_RATIO = 3.0
LEVEL_SHIFT_GROSS = 5.0
DOLLAR_VOLUME_LOOKBACK = 20
REVERT_TOL = 0.20
REVERT_MAX_QUOTES = 5
COOLOFF_SESSIONS = REVERT_MAX_QUOTES + 1  # break day + classification horizon

_RATIOS = np.array([1 / k for k in SPLIT_FACTORS] + list(SPLIT_FACTORS), dtype=float)


@dataclass
class IntegrityResult:
    clean: np.ndarray            # adjusted prices after quarantine + split repair
    events: pd.DataFrame         # one row per raw break with its classification
    break_day: np.ndarray        # bool [T,N]: a raw break of any class occurred that day
    unreliable_day: np.ndarray   # bool [T,N]: bad_print, level_shift or unclassified break that day
    artifact_day: np.ndarray     # bool [T,N]: bad_print or level_shift (data artifacts) that day
    quarantined: np.ndarray      # bool [T,N]: quote removed as a bad print
    split_factor: np.ndarray     # float [T,N]: ratio applied on split days, else 1

    def summary(self) -> pd.Series:
        counts = self.events["kind"].value_counts() if len(self.events) else pd.Series(dtype=int)
        return pd.Series({"breaks": len(self.events), **counts.to_dict(),
                          "quarantined_quotes": int(self.quarantined.sum())})


def nearest_split_ratio(gross: float):
    j = int(np.argmin(np.abs(np.log(gross) - np.log(_RATIOS))))
    return _RATIOS[j], abs(np.log(gross) - np.log(_RATIOS[j]))


def classify_column(px: np.ndarray, close: np.ndarray, volume: np.ndarray):
    """Classify the breaks of one symbol.  Returns events and per-day arrays."""
    T = len(px)
    obs = np.flatnonzero(np.isfinite(px))
    quarantine = np.zeros(T, bool)
    factor = np.ones(T)
    events = []
    if len(obs) < 2:
        return events, quarantine, factor
    p = px[obs]
    gross = p[1:] / p[:-1]
    candidates = np.flatnonzero(np.abs(np.log(gross)) > np.log(BREAK_GROSS)) + 1  # positions in obs
    consumed_until = -1
    for k in candidates:
        if k <= consumed_until:
            continue
        pre = p[k - 1]
        g = p[k] / pre
        d, prev_d = obs[k], obs[k - 1]
        # 1) bad print: the pre-break level is restored within REVERT_MAX_QUOTES quotes
        restore = None
        for j in range(k + 1, min(k + 1 + REVERT_MAX_QUOTES, len(obs))):
            if abs(p[j] / pre - 1) < REVERT_TOL:
                restore = j
                break
        horizon = obs[min(k + REVERT_MAX_QUOTES, len(obs) - 1)]
        dollar = close[max(0, d - DOLLAR_VOLUME_LOOKBACK):d] * volume[max(0, d - DOLLAR_VOLUME_LOOKBACK):d]
        before = np.nanmedian(dollar) if np.isfinite(dollar).any() else np.nan  # noqa: E501
        dv_ratio = close[d] * volume[d] / before if before > 0 else np.nan
        base = {"day": d, "prev_day": prev_d, "gross": g, "pre_price": pre, "post_price": p[k],
                "pre_close": close[prev_d], "dv_day_ratio": dv_ratio, "confirm_day": horizon}
        if restore is not None:
            quarantine[obs[k:restore]] = True
            consumed_until = restore
            events.append({**base, "kind": "bad_print", "ratio": np.nan, "segment_quotes": restore - k})
            continue
        ratio, dist = nearest_split_ratio(g)
        plausible = np.isfinite(dv_ratio) and dv_ratio <= SPLIT_MAX_DOLLAR_VOLUME_RATIO
        if dist < SPLIT_LOG_TOL and plausible:
            factor[d] = ratio
            events.append({**base, "kind": "split", "ratio": ratio, "segment_quotes": 0})
        elif plausible and abs(np.log(g)) > np.log(LEVEL_SHIFT_GROSS):
            factor[d] = g
            events.append({**base, "kind": "level_shift", "ratio": g, "segment_quotes": 0})
        else:
            events.append({**base, "kind": "unclassified", "ratio": ratio if dist < SPLIT_LOG_TOL else np.nan,
                           "segment_quotes": 0})
    return events, quarantine, factor


def audit_prices(adj: np.ndarray, close: np.ndarray, volume: np.ndarray, calendar: pd.DatetimeIndex,
                 symbols: pd.Index, repair_splits: bool = True) -> IntegrityResult:
    """Classify breaks for every symbol and build the cleaned price matrix."""
    T, N = adj.shape
    clean = adj.copy()
    quarantined = np.zeros((T, N), bool)
    split_factor = np.ones((T, N))
    break_day = np.zeros((T, N), bool)
    unreliable = np.zeros((T, N), bool)
    artifact = np.zeros((T, N), bool)
    rows = []
    for s in range(N):
        events, q, f = classify_column(adj[:, s], close[:, s], volume[:, s])
        if not events:
            continue
        quarantined[:, s] = q
        split_factor[:, s] = f
        for e in events:
            break_day[e["day"], s] = True
            if e["kind"] != "split":
                unreliable[e["day"], s] = True
            if e["kind"] in ("bad_print", "level_shift"):
                artifact[e["day"], s] = True
            rows.append({"symbol": symbols[s], "date": calendar[e["day"]], "prev_date": calendar[e["prev_day"]],
                         "confirm_date": calendar[e["confirm_day"]], **{k: v for k, v in e.items()
                                                                       if k not in ("day", "prev_day", "confirm_day")}})
        col = clean[:, s]
        col[q] = np.nan
        if repair_splits and (f != 1).any():  # repair_splits=False: "as delivered" sensitivity bound
            # Back-adjust: every price before a repaired day is multiplied by that day's factor.
            after = np.cumprod(f[::-1])[::-1]          # product of factors at t..T-1
            cumulative = np.append(after[1:], 1.0)      # product of factors strictly after t
            col *= cumulative
        clean[:, s] = col
    events = pd.DataFrame(rows, columns=["symbol", "date", "prev_date", "confirm_date", "gross", "pre_price",
                                         "post_price", "pre_close", "dv_day_ratio", "kind", "ratio",
                                         "segment_quotes"])
    return IntegrityResult(clean, events, break_day, unreliable, artifact, quarantined, split_factor)


def neutralized_prices(adj: np.ndarray, calendar: pd.DatetimeIndex, symbols: pd.Index,
                       result: IntegrityResult) -> np.ndarray:
    """Sensitivity bound: remove the one-day return of every non-bad-print break.

    Each split/unclassified break's gross return is divided out (that day's
    return becomes zero), on top of the bad-print quarantine.  This erases
    fake split losses *and* genuine crashes/jumps, so it brackets the repair.
    """
    clean = adj.copy()
    clean[result.quarantined] = np.nan
    ev = result.events.loc[result.events.kind.isin(["split", "level_shift", "unclassified"])]
    if not len(ev):
        return clean
    di = calendar.get_indexer(ev.date)
    si = symbols.get_indexer(ev.symbol)
    for s in np.unique(si):
        f = np.ones(len(calendar))
        mask = si == s
        f[di[mask]] = ev.gross.to_numpy()[mask]
        after = np.cumprod(f[::-1])[::-1]
        clean[:, s] *= np.append(after[1:], 1.0)
    return clean


def trailing_count(mask: np.ndarray, window: int) -> np.ndarray:
    """Number of True values in the trailing ``window`` sessions, including today."""
    c = np.cumsum(mask, axis=0, dtype=np.int32)
    out = c.copy()
    out[window:] -= c[:-window]
    return out
