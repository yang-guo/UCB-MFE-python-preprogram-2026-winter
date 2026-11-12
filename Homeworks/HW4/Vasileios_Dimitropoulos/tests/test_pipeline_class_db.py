"""End-to-end smoke test on the small class database (hw4.md: "use equities_class.duckdb
to get your code working quickly").  Runs the whole pipeline in ~10 seconds:
features -> labels -> walk-forward Ridge -> weekly top-3 backtest -> audits.
Skipped if the class database has not been downloaded.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hw4lib.data import CLASS_DB, load_market_data  # noqa: E402
from hw4lib.features import attach_labels, build_panel, weekly_signal_dates  # noqa: E402
from hw4lib.model import RidgeRanker, add_model_columns, walk_forward  # noqa: E402
from hw4lib.strategy import WeeklyTopN  # noqa: E402
from hw4lib.backtester import Backtester  # noqa: E402
from hw4lib.analytics import audit_cash, audit_nav, audit_timing, comparison_table  # noqa: E402

pytestmark = pytest.mark.skipif(not CLASS_DB.is_file(), reason="equities_class.duckdb not downloaded")


@pytest.fixture(scope="module")
def pipeline():
    md = load_market_data(CLASS_DB, start="2008-01-01", end="2015-12-31")
    dates = weekly_signal_dates(md.calendar, "2010-01-01", "2015-12-31")
    panel = add_model_columns(attach_labels(build_panel(md, dates), md))
    d = pd.DatetimeIndex(sorted(panel.date.unique()))
    fits = [d[(d.year == y) & (d.quarter == q)].min() for y in (2014, 2015) for q in (1, 2, 3, 4)]
    preds, audit = walk_forward(panel, lambda: RidgeRanker("y_rank", 1000.0), fits, "2015-12-31")
    sectors = md.universe.set_index("symbol").sector
    start = md.calendar[md.calendar.get_loc(fits[0]) + 1]
    res = Backtester(md.adj, WeeklyTopN(preds, sectors, top_n=3), fee_bps=10).run(start, "2015-12-31")
    return md, panel, preds, audit, res


def test_features_are_dated_before_entry(pipeline):
    md, panel, *_ = pipeline
    assert len(panel) > 1000 and panel.date.lt(panel.entry_date).all()
    assert (panel.available_date.isna() | panel.available_date.le(panel.date)).all()
    assert panel.label_end.dropna().gt(panel.entry_date[panel.label_end.notna()]).all()


def test_walk_forward_uses_only_finished_outcomes(pipeline):
    _, _, _, audit, _ = pipeline
    assert (pd.to_datetime(audit.last_training_label_end) < pd.to_datetime(audit.first_scored_signal)).all()


def test_backtest_accounting_on_class_db(pipeline):
    md, _, _, _, res = pipeline
    assert len(res.fills) > 100
    assert audit_cash(res) < 1e-6
    assert np.abs(audit_nav(res, md.adj)).max() < 1e-6
    t = audit_timing(res, md.calendar)
    assert t["buys filled at the session after their signal"] and t["no sale before maturity"]
    assert res.history.cash.min() >= 0
    perf = comparison_table(res, md.benchmark)
    assert np.isfinite(perf.loc["CAGR"].astype(float)).all()
