"""Look-ahead checks for the feature/label pipeline, on the class database.

    uv run --with pytest pytest Homeworks/HW4/Oscar_Sallusti/tests -q
"""

import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from features import (FEATURES, HOLDING_DAYS, build_panel, load_fundamental_events,  # noqa: E402
                      load_market)
from model import training_mask  # noqa: E402

DB = HERE.parents[2] / "Lectures/modeling/data/equities_class.duckdb"
pytestmark = [pytest.mark.skipif(not DB.exists(), reason="class database not downloaded"),
              pytest.mark.filterwarnings("ignore:Quarantined")]
CUT = "2013-06-28"
ANCHOR = "2010-03-31"  # class data ends before the real 2016-03-31 anchor


@pytest.fixture(scope="module")
def full():
    m = load_market(DB, end="2015-12-31")
    return m, build_panel(m, load_fundamental_events(DB, end="2015-12-31"), "2010-01-01", "2015-12-31",
                          anchor=ANCHOR)


def test_features_do_not_change_when_future_data_is_removed(full):
    """Rebuild with data ending at CUT: every feature on dates <= CUT must match."""
    _, p_full = full
    m_cut = load_market(DB, end=CUT)
    p_cut = build_panel(m_cut, load_fundamental_events(DB, end=CUT), "2010-01-01", CUT,
                        anchor=ANCHOR)
    a = p_full[p_full.date <= p_cut.date.max()].set_index(["date", "symbol"]).sort_index()
    b = p_cut.set_index(["date", "symbol"]).sort_index()
    assert a.index.equals(b.index)                       # same eligible rows
    pd.testing.assert_frame_equal(a[FEATURES + ["sector", "universe_rank"]],
                                  b[FEATURES + ["sector", "universe_rank"]])


def test_labels_match_raw_prices(full):
    m, p = full
    cal = m["calendar"]
    con = duckdb.connect(str(DB), read_only=True)
    rows = p[p.ret_90.notna()].sample(40, random_state=0)
    for r in rows.itertuples():
        q = con.execute("""SELECT date, adj_close FROM prices WHERE symbol = ? AND volume > 0
                           ORDER BY date""", [r.symbol]).fetchdf()
        q["date"] = pd.to_datetime(q.date)
        q = q.set_index("date").adj_close
        entry = cal[cal.get_loc(r.date) + 1]              # next market session
        assert r.entry_date == entry
        exit_date = q.index[q.index >= entry + pd.Timedelta(days=HOLDING_DAYS)][0]
        assert r.label_end == exit_date
        assert r.ret_90 == pytest.approx(q[exit_date] / q[entry] - 1)


def test_membership_uses_entry_quarter(full):
    m, p = full
    mem = m["membership"]
    # 2010-03-31 is a signal date; entry 2010-04-01 is in Q2 -> Q2 list.
    row = p[p.date == "2010-03-31"]
    assert not row.empty
    q2 = set(mem[mem.quarter_start == "2010-04-01"].symbol)
    assert set(row.symbol) <= q2


def test_financials_were_public_at_signal(full):
    _, p = full
    con = duckdb.connect(str(DB), read_only=True)
    f = con.execute("""SELECT symbol, fiscal_date, filing_date FROM fundamentals
                       WHERE period = 'quarter' AND filing_date > fiscal_date""").fetchdf()
    f["fiscal_date"] = pd.to_datetime(f.fiscal_date)
    f["filing_date"] = pd.to_datetime(f.filing_date)
    for r in p[p.fiscal_date.notna()].sample(60, random_state=1).itertuples():
        known = f[(f.symbol == r.symbol) & (f.filing_date + pd.Timedelta(days=1) <= r.date)]
        assert r.fiscal_date == known.fiscal_date.max()   # newest period public by then


def test_training_rows_have_finished_outcomes(full):
    _, p = full
    fit_date = pd.Timestamp("2014-01-08")
    train = p[training_mask(p, fit_date)]
    assert train.label_end.max() < fit_date
    assert train.date.min() >= pd.Timestamp("2010-01-01")
    # Rows signalled in the last ~90 days before the fit are excluded.
    assert train.date.max() < fit_date - pd.Timedelta(days=HOLDING_DAYS)
