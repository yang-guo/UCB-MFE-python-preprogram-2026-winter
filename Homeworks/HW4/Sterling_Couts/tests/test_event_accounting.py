"""Hand-calculated event tests, separate from the frozen portfolio tests."""

import numpy as np
import pandas as pd

from event_accounting import run_event_account
from hw4.engine import run_backtest
from hw4.settings import TradingRules


def fixture():
    dates = pd.bdate_range("2022-12-30", "2023-04-05")
    quotes = pd.DataFrame({"OLD": 10.0, "NEW": 10.0}, index=dates)
    raw = pd.DataFrame({"OLD": 10.0, "NEW": 20.0}, index=dates)
    benchmark = pd.Series(100.0, index=dates)
    scores = pd.DataFrame({"date": [dates[0]], "symbol": ["OLD"], "score": [0.2]})
    rules = TradingRules(
        initial_cash=1000.0, top_n=1, cohort_fraction=1.0, name_cap=1.0, fee_bps=0.0
    )
    return quotes, raw, benchmark, scores, rules


def test_event_free_account_matches_original():
    q, r, b, s, rules = fixture()
    original = run_backtest(q, b, s, "2023-01-02", "2023-04-05", rules)
    revised = run_event_account(q, b, s, r, [], "2023-01-02", "2023-04-05", rules)
    np.testing.assert_allclose(
        original.history.nav, revised["history"].nav, rtol=0, atol=1e-10
    )
    np.testing.assert_allclose(
        original.history.cash, revised["history"].cash, rtol=0, atol=1e-10
    )


def test_cash_stock_event_adjusted_units_and_original_maturity():
    q, r, b, s, rules = fixture()
    event = dict(
        id="mixed",
        kind="cash_stock_merger",
        old="OLD",
        new="NEW",
        date="2023-01-04",
        basis_date="2023-01-03",
        cash_per_share=2.0,
        share_ratio=0.5,
    )
    result = run_event_account(q, b, s, r, [event], "2023-01-02", "2023-04-05", rules)
    # $1,000 purchases 100 OLD actual shares. Terms give $200 plus 50 NEW
    # actual shares, equivalent to 100 adjusted units at the 20/10 price basis.
    ev = result["events"].iloc[0]
    assert ev.actual_share_equivalent == 100
    assert ev.cash_entitlement == 200
    assert ev.new_units == 100
    h = result["history"]
    assert h.loc["2023-01-04", "cash"] == 0
    assert h.loc["2023-01-04", "receivables"] == 200
    assert h.loc["2023-01-04", "nav"] == 1200
    assert h.loc["2023-01-06", "cash"] == 200
    sales = result["fills"].loc[lambda x: x.side.eq("SELL")]
    assert sales.iloc[0].date == pd.Timestamp("2023-04-03")
    assert sales.iloc[0].maturity == pd.Timestamp("2023-04-02")
    assert sales.iloc[0].symbol == "NEW"
    assert h.nav.iloc[-1] == 1200
    np.testing.assert_allclose(h.realized_pnl.iloc[-1], 200)


def test_receivable_cannot_fund_orders_and_retired_quotes_do_not_trade():
    q, r, b, s, rules = fixture()
    s = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2022-12-30", "2023-01-04", "2023-01-06", "2023-01-09"]
            ),
            "symbol": ["OLD", "NEW", "OLD", "NEW"],
            "score": [0.2] * 4,
        }
    )
    event = dict(
        id="cash",
        kind="cash_merger",
        old="OLD",
        date="2023-01-04",
        basis_date="2023-01-03",
        cash_per_share=12.0,
    )
    result = run_event_account(q, b, s, r, [event], "2023-01-02", "2023-04-05", rules)
    buys = result["fills"].loc[lambda x: x.side.eq("BUY")]
    assert list(buys.date) == [pd.Timestamp("2023-01-02"), pd.Timestamp("2023-01-10")]
    assert "rejected_retired_security" in result["incidents"].event.values
    assert len(result["settlements"]) == 1
    assert result["settlements"].cash_change.iloc[0] == 1200
    assert result["history"].cash.min() >= 0


def test_rename_preserves_value_when_adjustment_scales_differ():
    q, r, b, s, rules = fixture()
    r["NEW"] = 10.0
    q["NEW"] = 5.0
    event = dict(
        id="rename",
        kind="rename",
        old="OLD",
        new="NEW",
        date="2023-01-04",
        basis_date="2023-01-03",
        share_ratio=1.0,
    )
    result = run_event_account(q, b, s, r, [event], "2023-01-02", "2023-04-05", rules)
    assert result["events"].new_units.iloc[0] == 200
    np.testing.assert_allclose(result["history"].nav, 1000)
    assert result["settlements"].empty


def test_unresolved_gap_still_stops_and_has_no_invented_sale():
    q, r, b, s, rules = fixture()
    q.loc["2023-01-04":, "OLD"] = np.nan
    result = run_event_account(q, b, s, r, [], "2023-01-02", "2023-04-05", rules)
    assert result["status"] == "blocked_data_gap"
    assert result["history"].index[-1] == pd.Timestamp("2023-01-10")
    assert result["fills"].side.eq("BUY").all()


def test_end_date_includes_unsettled_cash_claim():
    q, r, b, s, rules = fixture()
    event = dict(
        id="cash",
        kind="cash_merger",
        old="OLD",
        date="2023-01-04",
        basis_date="2023-01-03",
        cash_per_share=12.0,
    )
    result = run_event_account(q, b, s, r, [event], "2023-01-02", "2023-01-04", rules)
    assert result["history"].cash.iloc[-1] == 0
    assert result["history"].receivables.iloc[-1] == 1200
    assert result["history"].nav.iloc[-1] == 1200
    assert result["open_lots"].empty
