"""Refresh dates must not create extra orders or alter the 90-day horizon."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from cap_sensitivity_engine import run_sector_backtest as original_engine
from refreshed_signal_engine import run_sector_backtest
from refreshed_features import daily_rank_features
from hw4.settings import TradingRules


def example():
    dates = pd.bdate_range("2013-01-01", "2013-05-01")
    q = pd.DataFrame(100.0, index=dates, columns=["X", "Y"])
    s = pd.DataFrame(dict(date=[dates[0]] * 2, symbol=["X", "Y"], score=[2.0, 1.0]))
    r = replace(
        TradingRules(),
        initial_cash=1000.0,
        top_n=1,
        cohort_fraction=1.0,
        name_cap=1.0,
        fee_bps=0.0,
    )
    kw = dict(
        sectors={"X": "A", "Y": "B"},
        sector_cap=1.0,
        reinvest_maturity_cash=True,
        entry_eligibility=pd.DataFrame(True, index=dates, columns=q.columns),
    )
    return q, s, r, kw


def test_default_is_exact_parent_parity():
    q, s, r, kw = example()
    b = pd.Series(100.0, index=q.index)
    a = original_engine(q, b, s, q.index[1], q.index[-1], r, **kw)
    c = run_sector_backtest(q, b, s, q.index[1], q.index[-1], r, **kw)
    pd.testing.assert_frame_equal(a.history, c.history)
    pd.testing.assert_frame_equal(a.fills, c.fills)
    pd.testing.assert_frame_equal(a.decisions, c.decisions)


@pytest.mark.parametrize("collection", [list, set])
def test_refresh_changes_next_entry_rank_without_creating_daily_buy_triggers(
    collection,
):
    q, initial, r, kw = example()
    signals = q.index[:-1]
    s = pd.concat(
        [
            initial.assign(date=day, score=[2.0, 1.0] if i == 0 else [1.0, 2.0])
            for i, day in enumerate(signals)
        ],
        ignore_index=True,
    )
    result = run_sector_backtest(
        q,
        pd.Series(100.0, index=q.index),
        s,
        q.index[1],
        q.index[-1],
        r,
        monthly_signal_dates=collection([q.index[0]]),
        require_fresh_predictions=True,
        **kw,
    )
    buys = result.fills.loc[result.fills.side.eq("BUY")]
    assert buys.symbol.tolist() == ["X", "Y"]
    due = q.index[q.index.searchsorted(q.index[1] + pd.Timedelta(days=90))]
    assert buys.date.tolist() == [q.index[1], q.index[q.index.get_loc(due) + 1]]
    assert (result.decisions.model_score_date == result.decisions.signal_date).all()
    assert (buys.maturity - buys.date).eq(pd.Timedelta(days=90)).all()
    assert len(result.decisions) == 2
    # Refuse a stale forecast when the stricter alignment policy is requested.
    with pytest.raises(ValueError, match="preceding close"):
        run_sector_backtest(
            q,
            pd.Series(100.0, index=q.index),
            initial,
            q.index[1],
            q.index[-1],
            r,
            monthly_signal_dates=[q.index[0]],
            require_fresh_predictions=True,
            **kw,
        )


def test_daily_features_match_direct_arithmetic_and_ignore_future_prices():
    d = pd.bdate_range("2010-01-01", periods=380)
    q = pd.DataFrame(
        {"X": np.exp(np.arange(380) * 0.001), "Y": np.exp(np.arange(380) * 0.002)},
        index=d,
    )
    e = pd.DataFrame(True, index=d, columns=q.columns)
    f = pd.DataFrame(
        dict(
            symbol=["X", "Y"],
            period="quarter",
            fiscal_date=pd.Timestamp("2009-12-31"),
            filing_date=pd.Timestamp("2010-02-01"),
            total_assets=100.0,
            revenue=100.0,
            net_income=10.0,
            operating_income=[5.0, 10.0],
            total_debt=20.0,
            cash_and_equivalents=10.0,
            free_cash_flow=5.0,
            operating_cash_flow=10.0,
        )
    )
    sectors = pd.Series({"X": "A", "Y": "A"})
    a = daily_rank_features(q, e, f, sectors, d[300], d[350])
    changed = q.copy()
    changed.loc[changed.index > d[330], "X"] *= 7
    b = daily_rank_features(changed, e, f, sectors, d[300], d[350])
    pd.testing.assert_frame_equal(a.loc[a.date.le(d[330])], b.loc[b.date.le(d[330])])
    first = a.loc[a.date.eq(d[300]) & a.symbol.eq("X")].iloc[0]
    np.testing.assert_allclose(first.momentum_63, np.exp(0.063) - 1)
    np.testing.assert_allclose(first.rank_momentum_63, 0.5)
    assert first.entry_date == d[301]
