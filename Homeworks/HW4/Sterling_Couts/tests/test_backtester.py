"""Hand-specified answers and invariants, not snapshots of implementation output."""

import numpy as np
import pandas as pd
import pytest

from hw4.engine import comparison, performance, plan_budgets, run_backtest
from hw4.settings import TradingRules


def fixture_market():
    dates = pd.bdate_range("2022-12-30", "2023-05-05")
    quotes = pd.DataFrame({"X": 10.0, "Y": 20.0}, index=dates)
    benchmark = pd.Series(100.0, index=dates)
    scores = pd.DataFrame({"date": [dates[0]], "symbol": ["X"], "score": [0.1]})
    return dates, quotes, benchmark, scores


def rules(**kwargs):
    return TradingRules(
        initial_cash=1000, top_n=1, cohort_fraction=1, name_cap=1, **kwargs
    )


def test_round_trip_fees_and_weekend_maturity():
    d, q, b, s = fixture_market()
    r = run_backtest(q, b, s, "2023-01-02", d[-1], rules(fee_bps=10))
    # Jan 2 + 90 calendar days = Sunday Apr 2; next supplied session is Apr 3.
    assert r.fills.iloc[1].date == pd.Timestamp("2023-04-03")
    assert r.fills.iloc[0].maturity == pd.Timestamp("2023-04-02")
    assert r.history.nav.iloc[-1] == pytest.approx(1000 * 0.999 / 1.001)
    assert r.history.nav.iloc[1] == pytest.approx(1000 / 1.001)
    assert r.open_lots.empty


def test_terminal_lot_is_marked_not_sold_early():
    d, q, b, s = fixture_market()
    q.loc["2023-02-01":, "X"] = 12
    r = run_backtest(q, b, s, "2023-01-02", "2023-02-01", rules(fee_bps=0))
    assert r.history.nav.iloc[-1] == pytest.approx(1200)
    assert len(r.open_lots) == 1
    assert len(r.fills) == 1
    assert r.history.unrealized_pnl.iloc[-1] == pytest.approx(200)


def test_missing_entry_is_rejected_not_filled_or_reallocated():
    d, q, b, s = fixture_market()
    q.loc["2023-01-02", "X"] = np.nan
    r = run_backtest(q, b, s, "2023-01-02", d[-1], rules(fee_bps=0))
    assert r.fills.empty
    assert (r.history.nav == 1000).all()
    assert "rejected_entry_missing_quote" in r.incidents.event.tolist()


def test_short_mark_gap_and_long_gap_block():
    d, q, b, s = fixture_market()
    q.loc["2023-01-04":"2023-01-05", "X"] = np.nan
    r = run_backtest(q, b, s, "2023-01-02", d[-1], rules(fee_bps=0))
    assert r.status == "complete"
    assert r.history.stale_holdings.sum() == 2
    q.loc["2023-01-04":, "X"] = np.nan
    blocked = run_backtest(q, b, s, "2023-01-02", d[-1], rules(fee_bps=0))
    assert blocked.status == "blocked_data_gap"
    with pytest.raises(ValueError, match="blocked"):
        comparison(blocked)


def test_missing_maturity_quote_delays_until_observed():
    d, q, b, s = fixture_market()
    q.loc["2023-04-03", "X"] = np.nan
    r = run_backtest(q, b, s, "2023-01-02", d[-1], rules(fee_bps=0))
    assert r.fills.iloc[-1].date == pd.Timestamp("2023-04-04")
    assert "deferred_maturity_missing_quote" in r.incidents.event.tolist()


def test_repeated_ticker_lots_keep_distinct_maturities():
    d, q, b, s = fixture_market()
    s = pd.concat(
        [
            s,
            pd.DataFrame(
                {"date": [pd.Timestamp("2023-01-31")], "symbol": ["X"], "score": [0.1]}
            ),
        ]
    )
    rule = TradingRules(
        initial_cash=1000, top_n=1, cohort_fraction=0.5, name_cap=1, fee_bps=0
    )
    r = run_backtest(q, b, s, "2023-01-02", d[-1], rule)
    sales = r.fills.loc[r.fills.side.eq("SELL")]
    assert sales.date.tolist() == [
        pd.Timestamp("2023-04-03"),
        pd.Timestamp("2023-05-02"),
    ]
    assert sales.lot_id.nunique() == 2
    assert r.history.nav.iloc[-1] == pytest.approx(1000)


def test_cap_aggregates_existing_positions_and_includes_fees():
    s = pd.Series({"X": 2.0, "Y": 1.0})
    r = TradingRules(top_n=2, cohort_fraction=1, name_cap=0.05, fee_bps=10)
    planned = plan_budgets(s, 960, 1000, {"X": 40}, r)
    assert planned["X"] / 1.001 == pytest.approx(10)
    assert planned["Y"] / 1.001 == pytest.approx(50)
    assert sum(planned.values()) <= 960


def test_future_price_changes_do_not_change_earlier_trades():
    d, q, b, s = fixture_market()
    first = run_backtest(q, b, s, "2023-01-02", d[-1], rules(fee_bps=0))
    q.loc["2023-03-01":, "X"] *= 2
    second = run_backtest(q, b, s, "2023-01-02", d[-1], rules(fee_bps=0))
    pd.testing.assert_frame_equal(first.fills.head(1), second.fills.head(1))


def test_metrics_have_independent_known_answers():
    idx = pd.to_datetime(["2022-12-30", "2023-01-02", "2023-06-01", "2024-01-02"])
    nav = pd.Series([100, 100, 80, 110], index=idx, dtype=float)
    p = performance(nav)
    assert p["max_drawdown"] == pytest.approx(-0.2)
    assert p["total_return"] == pytest.approx(0.1)
    assert p["CAGR"] == pytest.approx(1.1 ** (365.25 / 365) - 1)
    returns = np.array([0, -0.2, 0.375])
    expected = (
        np.sqrt(252) * (returns.mean() - (1.02 ** (1 / 252) - 1)) / returns.std(ddof=1)
    )
    assert p["Sharpe"] == pytest.approx(expected)


def test_fresh_runs_do_not_share_account_state():
    d, q, b, s = fixture_market()
    a = run_backtest(q, b, s, "2023-01-02", d[-1], rules(fee_bps=0))
    c = run_backtest(q, b, s, "2023-01-02", d[-1], rules(fee_bps=0))
    pd.testing.assert_frame_equal(a.history, c.history)
