"""Shared setup for the development and final scripts."""

from pathlib import Path

import numpy as np
import pandas as pd

from backtester import Backtester, benchmark_nav, performance
from data import DEFAULT_DB, load_market_data
from features import FEATURES, IDEA_FEATURES, build_all
from model import ModelSchedule, training_panel
from strategy import CohortTopNStrategy

HERE = Path(__file__).resolve().parent
OUTPUTS = HERE / "outputs"          # git-ignored (Homeworks/HW4/*/outputs/)
RESULTS = HERE / "results"          # small committed summaries and charts

DEV_BACKTEST = ("2012-01-03", "2015-12-31")
TEST_BACKTEST = ("2016-04-01", "2026-09-30")

# Submitted configuration: frozen after development (run_development.py), before
# the final test was run. Do not tune on the test.
CHOSEN = {"model": "ridge", "top_n": 30, "n_cohorts": 3, "fee_bps": 10.0,
          "name_cap": 0.05, "sector_cap": 0.25}
V1_CONFIG = CHOSEN

# v2: a variant chosen afterwards from further 2010-2015 research (run_ideas.py,
# run_allocation.py, run_agreement.py), *after* the submitted configuration's
# final result was known. Run once on the test period (run_final.py --variant v2)
# and reported in one line in the README; it is not the submission.
#   features      + skew of the previous 180 daily returns
#   weighting     linear in rank (best pick ~2x average), name-cap excess redistributed
#   min_agreement a pick is bought only if >= 73.2% of its absolute feature
#                 contributions are positive; otherwise its slot stays in cash
V2_CONFIG = {**CHOSEN, "features": FEATURES + ["skew_180"], "weighting": "rank",
             "redistribute": True, "min_agreement": 0.7324}


def setup(database=DEFAULT_DB):
    data = load_market_data(database)
    feats, target, label_end = build_all(data)
    panel = training_panel(data, feats, target, label_end)
    return data, feats, target, label_end, panel


class MomentumBaseline:
    """No model: score = 12-1 month momentum. Same interface as ModelSchedule."""

    def __init__(self, data, feats):
        self.data, self.feats = data, feats
        self.fit_log = []

    def scores(self, t):
        from model import eligible_mask
        cols = np.flatnonzero(eligible_mask(self.data, self.feats, t))
        s = self.feats["mom_252_21"][t, cols].astype(float)
        keep = np.isfinite(s)
        return cols[keep], s[keep]


def run_strategy(data, feats, panel, start, end, model="ridge", top_n=30, n_cohorts=3,
                 fee_bps=10.0, name_cap=0.05, sector_cap=0.25, features=FEATURES, weighting="equal",
                 redistribute=False, min_score=None, min_agreement=None, per_sector=None,
                 corr_cap=None, rotation=None):
    schedule = (MomentumBaseline(data, feats) if model == "momentum"
                else ModelSchedule(model, panel, data, feats, features))
    strategy = CohortTopNStrategy(data, feats, schedule, top_n=top_n, n_cohorts=n_cohorts,
                                  name_cap=name_cap, sector_cap=sector_cap, weighting=weighting,
                                  redistribute=redistribute, min_score=min_score,
                                  min_agreement=min_agreement, per_sector=per_sector,
                                  corr_cap=corr_cap, rotation=rotation)
    result = Backtester(data, strategy, fee_bps=fee_bps).run(start, end)
    # Prepend the initial capital at the signal close before the first session, so
    # the first entry fees show up in returns and drawdown. The benchmark is
    # bought at the first test close, so it is also $1M on both of these rows.
    first = data.calendar.get_loc(result.daily.index[0])
    t0 = data.calendar[first - 1]
    nav = pd.concat([pd.Series({t0: result.ledger.initial_cash}), result.daily.nav])
    bench = benchmark_nav(data, result.daily.index[0], end)
    bench = pd.concat([pd.Series({t0: result.ledger.initial_cash}), bench])
    return result, strategy, schedule, nav, bench


def compare(nav, bench, label="strategy"):
    return pd.DataFrame({label: performance(nav), "S&P 500 TR (^SP500TR)": performance(bench)})
