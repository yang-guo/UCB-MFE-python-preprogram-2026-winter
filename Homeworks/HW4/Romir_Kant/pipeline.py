"""Shared steps for validate.py and final_test.py: build/cache the panel and
turn predictions into backtest scores."""

from pathlib import Path
import time

import numpy as np
import pandas as pd

import research

HERE = Path(__file__).resolve().parent
OUTPUTS = HERE / "outputs"   # big generated files, ignored by Git
RESULTS = HERE / "results"   # small tables and charts that I submit
OUTPUTS.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)


def log(msg, t0=[time.time()]):
    print(f"[{time.time() - t0[0]:6.0f}s] {msg}", flush=True)


def load(database=research.FULL_DB, rebuild=False):
    """Load the market data and the weekly panel (cached as parquet).

    The cache is rebuilt whenever research.py is newer than it, so a change to
    the features or the price repair can't silently reuse old results.
    """
    md = research.MarketData(database)
    log(f"loaded {len(md.symbols)} symbols x {len(md.calendar)} sessions; quality {md.quality}")
    log(f"price repairs: {md.repairs.groupby('repair').rows.sum().to_dict()}")
    cache = OUTPUTS / f"panel_{Path(database).stem}.parquet"
    fresh = cache.exists() and cache.stat().st_mtime > Path(research.__file__).stat().st_mtime
    if fresh and not rebuild:
        panel = pd.read_parquet(cache)
    else:
        panel = research.build_panel(md)
        panel.to_parquet(cache)
    log(f"panel: {len(panel):,} eligible stock-weeks, {panel.date.nunique()} signal dates")
    return md, panel


MAX_ZERO_VOLUME_DAYS = 3        # in the last 63 sessions
MAX_ONE_DAY_MOVE = np.log(3.0)  # no day above +200% or below -67% in the last year


def tradeable(panel, min_dollar_volume):
    """Screens a trader could apply at the signal close (trailing data only).

    The quality thresholds are fixed rules I set before validation; only the
    dollar-volume level is chosen in validate.py.
    """
    return (panel.dollar_volume_3m.ge(min_dollar_volume)
            & panel.zero_volume_days_3m.le(MAX_ZERO_VOLUME_DAYS)
            & panel.max_abs_log_move_12m.lt(MAX_ONE_DAY_MOVE))


def scores_for(panel, preds, mask):
    s = panel.loc[mask & preds.notna(), ["date", "symbol"]].copy()
    s["score"] = preds[s.index]
    return s


def sectors(md):
    return md.universe.set_index("symbol").sector.fillna("Unknown")
