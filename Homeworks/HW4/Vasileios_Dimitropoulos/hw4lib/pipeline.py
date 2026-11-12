"""Experiment configuration and the cached feature panel."""

from pathlib import Path
import time

import pandas as pd

from .data import FULL_DB, load_market_data
from .features import attach_labels, build_panel, weekly_signal_dates
from .model import SECTORS, add_model_columns

HERE = Path(__file__).resolve().parents[1]
OUTPUTS = HERE / "outputs"       # large generated files (git-ignored)
RESULTS = HERE / "results"       # small tables and figures committed with the submission

CONFIG = {
    "database": "equities_top2000.duckdb",
    "history_start": "2008-01-01",           # warm-up for 252-session features
    "train_start": "2010-01-01",             # development period starts here
    "dev_end": "2015-12-31",
    "test_start": "2016-04-01",
    "test_end": "2026-09-30",
    "first_test_signal": "2016-03-31",       # so the first purchase is at the first test close
    "holding_days": 90,
    "initial_cash": 1_000_000.0,
    "fee_bps_per_side": 10.0,
    "cash_interest": 0.0,
    "risk_free_for_sharpe": 0.02,
    "min_price": 5.0,
    "min_dollar_volume": 2e6,
    "cohort_fraction": 1 / 13,               # 13 weekly cohorts ~ one 90-day holding period
    "name_cap": 0.05,
    "sector_cap": 0.30,
    "max_stale_sessions": 5,
    "suspect_jump": 3.0,
}


def timed(label):
    class _T:
        def __enter__(self):
            self.t = time.time()

        def __exit__(self, *a):
            print(f"{label}: {time.time() - self.t:.1f}s")
    return _T()


def load_everything(rebuild=False):
    """Return (market data, model-ready panel).  The panel is cached as parquet."""
    OUTPUTS.mkdir(exist_ok=True)
    with timed("load database"):
        md = load_market_data(FULL_DB, start=CONFIG["history_start"], end=CONFIG["test_end"])
    cache = OUTPUTS / "panel.parquet"
    if cache.exists() and not rebuild:
        panel = pd.read_parquet(cache)
        panel["sector_cat"] = pd.Categorical(panel.sector_cat.astype(str),
                                             categories=SECTORS)
        return md, panel
    with timed("build features + labels"):
        dates = weekly_signal_dates(md.calendar, CONFIG["train_start"], CONFIG["test_end"],
                                    extra=[CONFIG["first_test_signal"]])
        panel = build_panel(md, dates, CONFIG["min_price"], CONFIG["min_dollar_volume"])
        panel = add_model_columns(attach_labels(panel, md, CONFIG["holding_days"]))
    panel.to_parquet(cache)
    return md, panel
