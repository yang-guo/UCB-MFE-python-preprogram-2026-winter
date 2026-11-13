"""Prefix-invariance check: future data must not change past features.

Build the panel twice, once from data *masked* after ``cutoff`` (prices,
filings and membership after the cutoff are removed; the calendar is kept so
the cutoff row still has an entry session) and once from the full data, then
compare every model-input column on eligible rows dated on/before the cutoff.
Masking rather than truncating matters: a truncated build drops the cutoff
row itself, and with a weekly grid a 5-session look-ahead then goes unseen
(the leaked-feature test caught exactly that).  Any difference means some feature looked
past its signal date (centered window, backfill, future-dated join, peer
statistic computed with later rows, ...).  Run it whenever a feature is added.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd

from . import integrity as I
from .data import MarketData, Reference
from .features import PLACEHOLDER_AVAILABILITY_DAYS
from .panel import PanelConfig, build_panel


def mask_market(m: MarketData, cutoff) -> MarketData:
    """Same calendar; every quote after ``cutoff`` removed."""
    k = int(m.calendar.searchsorted(pd.Timestamp(cutoff), side="right"))
    adj, close, volume = m.adj.copy(), m.close.copy(), m.volume.copy()
    for a in (adj, close, volume):
        a[k:] = np.nan
    return dataclasses.replace(m, adj=adj, close=close, volume=volume)


def truncate_reference(r: Reference, cutoff) -> Reference:
    cutoff = pd.Timestamp(cutoff)
    f = r.fundamentals
    valid = f.filing_date > f.fiscal_date
    avail = np.where(valid, f.filing_date + pd.Timedelta(days=1),
                     f.fiscal_date + pd.Timedelta(days=PLACEHOLDER_AVAILABILITY_DAYS))
    return Reference(r.universe, r.membership.loc[r.membership.selection_date <= cutoff],
                     f.loc[pd.to_datetime(avail) <= cutoff])


def prefix_invariance(market: MarketData, ref: Reference, cutoff, columns, panel_cfg=PanelConfig(),
                      panel_end=None, builder=build_panel, audit_fn=None) -> pd.DataFrame:
    """Max abs difference per column between truncated and full builds (should all be 0)."""
    audit_fn = audit_fn or (lambda m: I.audit_prices(m.adj, m.close, m.volume, m.calendar, m.symbols))
    cutoff = pd.Timestamp(cutoff)
    short_m, short_r = mask_market(market, cutoff), truncate_reference(ref, cutoff)
    short = builder(short_m, short_r, audit_fn(short_m), end=cutoff, config=panel_cfg)
    full = builder(market, ref, audit_fn(market), end=panel_end or cutoff, config=panel_cfg)
    keys = ["date", "symbol"]
    a = short.loc[short.date <= cutoff]
    b = full.loc[full.date <= cutoff]
    merged = a[keys + ["eligible", *columns]].merge(b[keys + ["eligible", *columns]], on=keys,
                                                    how="outer", suffixes=("_cut", "_full"), indicator=True)
    rows = [{"column": "row_set", "max_abs_diff": float((merged._merge != "both").sum())},
            {"column": "eligible", "max_abs_diff": float((merged.eligible_cut != merged.eligible_full).sum())}]
    both = merged.loc[merged._merge.eq("both") & merged.eligible_full.astype(bool)]
    for c in columns:
        x, y = both[f"{c}_cut"].astype(float), both[f"{c}_full"].astype(float)
        nan_mismatch = (x.isna() != y.isna()).sum()
        diff = (x - y).abs().max()
        rows.append({"column": c, "max_abs_diff": float(np.nan_to_num(diff) + nan_mismatch)})
    return pd.DataFrame(rows)
