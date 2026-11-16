"""Final-policy wiring: audited history units and sector/event accounting."""

from dataclasses import replace

import numpy as np
import pandas as pd

from event_accounting import run_event_account as original_events
from rank_event_accounting import run_event_account
from hw4.settings import TradingRules
from rank_final_inputs import rebase_reviewed_prefix
from rank_accounting_prices import normalize_prefix
import pytest


def fixture():
    dates = pd.bdate_range("2012-12-31", "2013-04-05")
    q = pd.DataFrame(10.0, index=dates, columns=["A", "B", "C"])
    s = pd.DataFrame(
        dict(date=[dates[0]] * 3, symbol=["A", "B", "C"], score=[3.0, 2.0, 1.0])
    )
    b = pd.Series(100.0, index=dates)
    r = replace(
        TradingRules(), initial_cash=1000.0, top_n=3, cohort_fraction=1.0, fee_bps=0.0
    )
    return q, s, b, r


def test_disabled_sector_limit_preserves_original_event_account():
    q, s, b, r = fixture()
    events = [
        dict(
            id="cash_A",
            old="A",
            kind="cash_merger",
            date="2013-02-01",
            basis_date="2013-01-31",
            cash_per_share=12.0,
        )
    ]
    a = original_events(q, b, s, q, events, q.index[1], q.index[-1], r)
    c = run_event_account(q, b, s, q, events, q.index[1], q.index[-1], r)
    for key in a:
        if isinstance(a[key], pd.DataFrame):
            pd.testing.assert_frame_equal(a[key], c[key])
        else:
            assert a[key] == c[key]


def test_sector_budgets_and_claims_obey_independent_limits():
    q, s, b, r = fixture()
    events = [
        dict(
            id="cash_A",
            old="A",
            kind="cash_merger",
            date="2013-02-01",
            basis_date="2013-01-31",
            cash_per_share=12.0,
        )
    ]
    result = run_event_account(
        q,
        b,
        s,
        q,
        events,
        q.index[1],
        q.index[-1],
        r,
        sectors={"A": "Energy", "B": "Energy", "C": "Technology"},
        sector_cap=0.08,
    )
    buys = result["fills"].loc[lambda f: f.side.eq("BUY")].set_index("symbol")
    np.testing.assert_allclose(
        -buys.cash_change.reindex(["A", "B", "C"]), [40.0, 40.0, 50.0]
    )
    h = result["history"]
    assert h.loc["2013-02-01", "receivables"] == 48.0
    assert h.loc["2013-02-01", "cash"] == 870.0
    assert h.loc["2013-02-05", "cash"] == 918.0
    assert h.nav.iloc[-1] == 1008.0


def test_reviewed_prefix_rebase_preserves_returns_and_future_source_units():
    dates = pd.to_datetime(["2015-12-29", "2015-12-30", "2015-12-31", "2016-01-04"])
    original = pd.Series([480.0, 496.0, 512.0, 504.0], index=dates, name="LCI")
    reviewed = pd.Series([30.0, 31.0, 32.0], index=dates[:3], name="LCI")
    rebased, proof = rebase_reviewed_prefix(original, reviewed)
    pd.testing.assert_series_equal(rebased, original)
    assert proof["factor"] == 16.0
    np.testing.assert_allclose(rebased.pct_change().iloc[-1], 504.0 / 512.0 - 1)


def test_verified_unit_seam_does_not_create_an_investment_loss():
    dates = pd.bdate_range("2025-05-20", periods=4)
    raw = pd.Series([1000.0, 1100.0, 110.0, 121.0], index=dates)
    normalized = normalize_prefix(raw, dates[2], 10.0)
    np.testing.assert_allclose(normalized, [100.0, 110.0, 110.0, 121.0])
    # Same $1,000 investment: 10 normalized units, final value $1,210.
    assert 1000 / normalized.iloc[0] * normalized.iloc[-1] == 1210
    assert normalized.pct_change().iloc[2] == 0
    with pytest.raises(ValueError):
        normalize_prefix(raw, dates[2], 0)
