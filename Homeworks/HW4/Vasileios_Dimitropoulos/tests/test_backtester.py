"""Hand-checkable accounting and timing tests for the HW4 backtester.

Run from the submission folder:  HW4/Scripts/python.exe -m pytest tests -q
Every expected number below can be verified with a calculator.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hw4lib.backtester import Account, Backtester  # noqa: E402
from hw4lib.strategy import WeeklyTopN  # noqa: E402
from hw4lib.analytics import audit_cash, audit_nav, performance  # noqa: E402

FEE = 10 / 10_000  # 10 bps per side


class FixedOrders:
    """Stub strategy: fixed budgets on given signal dates; records what it saw."""

    def __init__(self, plan):
        self.plan = {pd.Timestamp(k): v for k, v in plan.items()}
        self.seen = []

    def orders(self, signal_date, execution_date, snapshot):
        self.seen.append((signal_date, execution_date, snapshot.date, snapshot.cash))
        return pd.Series(self.plan.get(signal_date, {}), dtype=float)


def make_prices(n=140, **series):
    cal = pd.bdate_range("2020-01-01", periods=n)   # Mon-Fri sessions
    return pd.DataFrame({k: v(cal) if callable(v) else v for k, v in series.items()}, index=cal)


def run(adj, plan, start=None, end=None):
    strat = FixedOrders(plan)
    bt = Backtester(adj, strat, fee_bps=10)
    res = bt.run(start or adj.index[1], end or adj.index[-1])
    return res, strat


def test_buy_hold_sell_by_hand():
    adj = make_prices(AAA=lambda c: np.where(c < pd.Timestamp("2020-03-01"), 100.0, 110.0))
    signal = adj.index[1]                       # 2020-01-02 (Thu) -> fill 2020-01-03 (Fri)
    res, _ = run(adj, {signal: {"AAA": 10_000.0}})
    buy, sell = res.fills.iloc[0], res.fills.iloc[1]
    # Purchase: budget includes the fee.  units = 10,000 / (100 * 1.001)
    assert buy.date == pd.Timestamp("2020-01-03")
    assert buy.units == pytest.approx(99.9000999001)
    assert buy.fee == pytest.approx(9.99000999)
    assert buy.cash_change == pytest.approx(-10_000.0)
    # Due 2020-01-03 + 90 days = 2020-04-02 (Thu) -> sold that day at 110.
    assert sell.date == pd.Timestamp("2020-04-02") and sell.due_date == pd.Timestamp("2020-04-02")
    proceeds = 99.9000999001 * 110
    assert sell.cash_change == pytest.approx(proceeds * (1 - FEE))            # 10,978.02
    assert sell.realized_pnl == pytest.approx(proceeds * (1 - FEE) - 10_000)  # +978.02
    assert res.history.cash.iloc[-1] == pytest.approx(1_000_000 - 10_000 + proceeds * (1 - FEE))
    # NAV on the purchase day = cash + units * price = 1,000,000 - fee
    assert res.history.loc["2020-01-03", "nav"] == pytest.approx(1_000_000 - 9.99000999)
    assert audit_cash(res) < 1e-6
    assert audit_nav(res, adj).abs().max() < 1e-6


def test_weekend_maturity_sells_next_session():
    adj = make_prices(AAA=100.0)
    res, _ = run(adj, {"2020-01-06": {"AAA": 5_000.0}})   # Mon signal -> Tue 2020-01-07 fill
    sell = res.fills[res.fills.side.eq("SELL")].iloc[0]
    assert sell.due_date == pd.Timestamp("2020-04-06")    # Monday: a session
    res, _ = run(adj, {"2020-01-08": {"AAA": 5_000.0}})   # Wed signal -> Thu 2020-01-09 fill
    sell = res.fills[res.fills.side.eq("SELL")].iloc[0]
    assert sell.due_date == pd.Timestamp("2020-04-08")
    res, _ = run(adj, {"2020-01-03": {"AAA": 5_000.0}})   # Fri signal -> Mon 01-06 fill, due Sun 04-05
    sell = res.fills[res.fills.side.eq("SELL")].iloc[0]
    assert sell.due_date == pd.Timestamp("2020-04-05") and sell.date == pd.Timestamp("2020-04-06")


def test_missing_quote_at_maturity_defers_exit():
    adj = make_prices(AAA=100.0)
    adj.loc["2020-04-02":"2020-04-06", "AAA"] = np.nan        # due 04-02, quotes back 04-07
    adj.loc["2020-04-07":, "AAA"] = 90.0
    res, _ = run(adj, {"2020-01-02": {"AAA": 10_000.0}})
    sell = res.fills[res.fills.side.eq("SELL")].iloc[0]
    assert sell.date == pd.Timestamp("2020-04-07") and sell.price == 90.0
    assert "exit_deferred" in set(res.diagnostics.event)
    # While stale, the holding is carried at its last mark (100), not dropped.
    assert res.history.loc["2020-04-03", "holdings_value"] == pytest.approx(99.9000999001 * 100)


def test_delisting_is_frozen_not_sold():
    adj = make_prices(AAA=100.0, BBB=50.0)
    adj.loc["2020-02-03":, "AAA"] = np.nan                  # AAA disappears for good
    res, _ = run(adj, {"2020-01-02": {"AAA": 10_000.0, "BBB": 10_000.0}})
    assert set(res.fills.loc[res.fills.side.eq("SELL"), "symbol"]) == {"BBB"}  # no invented AAA sale
    last = res.history.iloc[-1]
    aaa_value = 99.9000999001 * 100
    assert last.unresolved_value == pytest.approx(aaa_value)
    assert last.nav - last.nav_low == pytest.approx(aaa_value)
    assert bool(res.open_lots.unresolved.iloc[0])
    # Stale for <= 5 sessions is not yet unresolved.
    assert res.history.loc["2020-02-07", "unresolved_value"] == 0.0     # 5th missing session
    assert res.history.loc["2020-02-10", "unresolved_value"] == pytest.approx(aaa_value)


def test_suspect_spike_is_ignored():
    adj = make_prices(AAA=100.0)
    adj.loc["2020-02-03", "AAA"] = 0.2                      # bad print (like AAM in the data)
    adj.loc["2020-04-02", "AAA"] = 25_000.0                 # bad print exactly at maturity
    res, _ = run(adj, {"2020-01-02": {"AAA": 10_000.0}})
    assert res.history.loc["2020-02-03", "holdings_value"] == pytest.approx(99.9000999001 * 100)
    sell = res.fills[res.fills.side.eq("SELL")].iloc[0]
    assert sell.date == pd.Timestamp("2020-04-03") and sell.price == 100.0


def test_persistent_new_level_is_accepted_after_five_sessions():
    adj = make_prices(AAA=100.0)
    adj.loc["2020-02-03":, "AAA"] = 20.0                    # a genuine -80% collapse
    res, _ = run(adj, {"2020-01-02": {"AAA": 10_000.0}})
    assert res.history.loc["2020-02-06", "holdings_value"] == pytest.approx(99.9000999001 * 100)
    assert res.history.loc["2020-02-07", "holdings_value"] == pytest.approx(99.9000999001 * 20)


def test_persistent_jump_up_is_never_accepted():
    adj = make_prices(AAA=100.0)
    adj.loc["2020-02-03":, "AAA"] = 1_700.0                 # broken adjustment (like LCI)
    res, _ = run(adj, {"2020-01-02": {"AAA": 10_000.0}})
    assert res.fills.side.eq("SELL").sum() == 0             # never sold at the bad price
    last = res.history.iloc[-1]
    assert last.holdings_value == pytest.approx(99.9000999001 * 100)   # last good mark
    assert last.unresolved_value == pytest.approx(99.9000999001 * 100)  # zero in lower bound


def test_cannot_spend_cash_twice():
    acct = Account(initial_cash=10_000, fee_bps=10)
    d, s = pd.Timestamp("2020-01-03"), pd.Timestamp("2020-01-02")
    acct.buy("AAA", 6_000, 100.0, date=d, signal_date=s, holding_days=90)
    with pytest.raises(ValueError):
        acct.buy("BBB", 6_000, 100.0, date=d, signal_date=s, holding_days=90)
    with pytest.raises(ValueError):          # decision date must precede the fill
        acct.buy("BBB", 1_000, 100.0, date=d, signal_date=d, holding_days=90)


def test_strategy_only_sees_previous_close():
    adj = make_prices(AAA=100.0)
    res, strat = run(adj, {})
    for signal, execution, snap_date, _ in strat.seen:
        assert signal < execution and snap_date == signal
        assert adj.index.get_loc(execution) == adj.index.get_loc(signal) + 1


def test_orders_larger_than_cash_are_refused():
    adj = make_prices(AAA=100.0)
    with pytest.raises(ValueError):
        run(adj, {"2020-01-02": {"AAA": 2_000_000.0}})


def test_weekly_top_n_caps():
    from hw4lib.backtester import Snapshot
    scores = pd.DataFrame({"date": pd.Timestamp("2020-01-03"), "symbol": list("ABCD"),
                           "score": [4.0, 3.0, 2.0, 1.0]})
    sectors = pd.Series({"A": "Tech", "B": "Tech", "C": "Energy", "D": "Energy"})
    strat = WeeklyTopN(scores, sectors, top_n=2, cohort_fraction=0.10, name_cap=0.05, sector_cap=0.30)
    # Already 29% in Tech -> A gets the 1% room left, B is skipped, C and ... budget 100k/2 = 50k each.
    snap = Snapshot(pd.Timestamp("2020-01-03"), 500_000.0, 1_000_000.0,
                    pd.Series({"X": 290_000.0, "Y": 210_000.0}), FEE)
    strat.sectors = pd.concat([sectors, pd.Series({"X": "Tech", "Y": "Energy"})])
    orders = strat.orders(pd.Timestamp("2020-01-03"), pd.Timestamp("2020-01-06"), snap)
    assert orders["A"] == pytest.approx(10_000 * (1 + FEE))       # sector room 10,000 of notional
    assert "B" not in orders                                       # Tech now full
    assert orders["C"] == pytest.approx(50_000.0)                  # equal budget (< 5% name cap)
    assert len(orders) == 2


def test_performance_metrics_by_hand():
    idx = pd.to_datetime(["2020-01-01", "2021-01-01"])
    nav = pd.Series([100.0, 121.0], index=idx)
    p = performance(nav)
    assert p.CAGR == pytest.approx(1.21 ** (365.25 / 366) - 1)
    nav = pd.Series([100, 120, 90, 130.0], index=pd.date_range("2020-01-01", periods=4))
    assert performance(nav).max_drawdown == pytest.approx(-0.25)


class SpendAvailable:
    """Stub: on given signal dates, spend everything the snapshot says is available."""

    def __init__(self, plan):
        self.plan = {pd.Timestamp(k): v for k, v in plan.items()}

    def orders(self, signal_date, execution_date, snapshot):
        sym = self.plan.get(signal_date)
        if sym is None:
            return pd.Series(dtype=float)
        if sym == "AAA":
            return pd.Series({sym: 500_000.0})
        return pd.Series({sym: snapshot.available})


def test_maturing_proceeds_fund_same_day_purchase_and_scale_down():
    adj = make_prices(AAA=100.0, BBB=50.0)
    adj.loc["2020-04-02":, "AAA"] = 90.0       # AAA falls on its maturity day (due 2020-04-02)
    bt = Backtester(adj, SpendAvailable({"2020-01-02": "AAA", "2020-04-01": "BBB"}), fee_bps=10)
    res = bt.run(adj.index[1], adj.index[-1])
    f = res.fills.set_index(["date", "side"])
    # The sale happens first, then the purchase, at the same 2020-04-02 close.
    sale = f.loc[(pd.Timestamp("2020-04-02"), "SELL")]
    buy = f.loc[(pd.Timestamp("2020-04-02"), "BUY")]
    # Planned at the 04-01 close: 500,000 cash + AAA lot (4995.005 units x 100) after a 10 bp fee.
    units = 500_000 / (100 * 1.001)
    planned = 500_000 + units * 100 * (1 - FEE)
    actual_cash = 500_000 + units * 90 * (1 - FEE)
    assert sale.cash_change == pytest.approx(units * 90 * (1 - FEE))
    assert -buy.cash_change == pytest.approx(actual_cash, rel=1e-9)        # scaled down to cash
    assert any(e.startswith("orders_scaled_to_cash") for e in res.diagnostics.event)
    assert res.history.loc["2020-04-02", "cash"] == pytest.approx(0.0, abs=1e-3)
    assert planned > actual_cash
    assert audit_cash(res) < 1e-6
