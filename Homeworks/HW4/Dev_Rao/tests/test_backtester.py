"""Hand-checkable accounting tests for hw4.backtester.

Run:  uv run --with pytest pytest tests -q      (from Homeworks/HW4/dev_rao)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hw4.backtester import Backtester, Ledger, Order, TopNCohortStrategy, reconcile  # noqa: E402

FEE_BPS = 10  # 0.1% per side


class FixedOrders:
    """Test strategy: emits pre-set fee-inclusive budgets on given signal dates."""

    def __init__(self, orders):
        self.orders = {pd.Timestamp(d): o for d, o in orders.items()}
        self.schedule = pd.DatetimeIndex(sorted(self.orders))

    def plan(self, snap):
        return [Order(s, "Tech", b, snap.date, k) for k, (s, b) in enumerate(self.orders.get(snap.date, []), 1)]


def make_prices(calendar, **series):
    return pd.DataFrame(series, index=pd.DatetimeIndex(calendar), dtype="float64")


def bdays(start, end, drop=()):
    cal = pd.bdate_range(start, end)
    return cal[~cal.isin(pd.DatetimeIndex(list(drop)))]


# 1 --------------------------------------------------------------------------
def test_one_buy_one_sale_exact_numbers():
    cal = bdays("2020-01-01", "2020-04-10")
    px = pd.Series(100.0, index=cal)
    px[px.index >= "2020-04-02"] = 110.0          # due date: 2020-01-03 + 90d = 2020-04-02 (Thu)
    prices = make_prices(cal, AAA=px)
    res = Backtester(prices, FixedOrders({"2020-01-02": [("AAA", 100_100.0)]}), {}, fee_bps=FEE_BPS,
                     initial_cash=1_000_000).run("2020-01-02", "2020-04-10")
    buy, sell = res.fills.iloc[0], res.fills.iloc[1]
    # units = 100,100 / (100 * 1.001) = 1,000 exactly; notional 100,000; fee 100
    assert buy.date == pd.Timestamp("2020-01-03") and buy.units == pytest.approx(1000.0)
    assert buy.fee == pytest.approx(100.0) and buy.cash_change == pytest.approx(-100_100.0)
    assert res.history.loc["2020-01-03", "cash"] == pytest.approx(899_900.0)
    assert res.history.loc["2020-01-03", "nav"] == pytest.approx(999_900.0)   # 899,900 + 1,000*100
    # sale: 1,000 * 110 = 110,000 notional, fee 110, proceeds 109,890
    assert sell.date == pd.Timestamp("2020-04-02") and sell.fee == pytest.approx(110.0)
    assert sell.realized_pnl == pytest.approx(9_790.0)                     # 109,890 - 100,100
    assert res.history.loc["2020-04-02", "cash"] == pytest.approx(1_009_790.0)
    assert res.history.nav.iloc[-1] == pytest.approx(1_009_790.0)
    assert res.ledger.total_fees == pytest.approx(210.0)
    # initial-capital row exists before the first entry
    assert res.history.index[0] == pd.Timestamp("2020-01-02") and res.history.nav.iloc[0] == 1_000_000


# 2 --------------------------------------------------------------------------
def test_maturity_on_weekend_and_holiday_sells_next_session():
    # entry Mon 2020-01-06 -> due Sun 2020-04-05; Mon 2020-04-06 is a "holiday"
    cal = bdays("2020-01-01", "2020-04-15", drop=["2020-04-06"])
    prices = make_prices(cal, AAA=np.linspace(100, 120, len(cal)))
    res = Backtester(prices, FixedOrders({"2020-01-03": [("AAA", 50_000.0)]}), {}, fee_bps=FEE_BPS,
                     initial_cash=100_000).run("2020-01-03", "2020-04-15")
    sell = res.fills.loc[res.fills.side == "SELL"].iloc[0]
    assert res.fills.iloc[0].date == pd.Timestamp("2020-01-06")
    assert sell.date == pd.Timestamp("2020-04-07")                         # next session with a quote
    assert sell.price == pytest.approx(prices.loc["2020-04-07", "AAA"])


# 3 --------------------------------------------------------------------------
def test_two_orders_same_day_cannot_exceed_cash():
    cal = bdays("2020-01-01", "2020-01-31")
    prices = make_prices(cal, AAA=100.0, BBB=50.0)
    res = Backtester(prices, FixedOrders({"2020-01-02": [("AAA", 60_000.0), ("BBB", 60_000.0)]}), {},
                     fee_bps=FEE_BPS, initial_cash=100_000).run("2020-01-02", "2020-01-31")
    b = res.fills.set_index("symbol")
    assert b.loc["AAA", "cash_change"] == pytest.approx(-60_000.0)
    assert b.loc["BBB", "cash_change"] == pytest.approx(-40_000.0)        # second order reduced
    assert res.history.cash.min() >= 0 and res.history.loc["2020-01-03", "cash"] == pytest.approx(0.0, abs=1e-6)
    assert (res.diagnostics.event == "reduced_to_cash").sum() == 1
    with pytest.raises(ValueError):                                       # the ledger itself refuses
        Ledger(100.0, FEE_BPS).buy("AAA", "Tech", "2020-01-03", "2020-01-02", 10.0, 100.01)


# 4 --------------------------------------------------------------------------
def test_maturity_cash_is_usable_same_day_but_never_exceeded():
    cal = bdays("2020-01-01", "2020-05-01")
    prices = make_prices(cal, AAA=100.0, BBB=20.0)
    # buy AAA 2020-01-03 -> due 2020-04-02. Plan BBB on 2020-04-01 for the 04-02 fill.
    res = Backtester(prices, FixedOrders({"2020-01-02": [("AAA", 100_000.0)],
                                          "2020-04-01": [("BBB", 150_000.0)]}), {},
                     fee_bps=FEE_BPS, initial_cash=100_000).run("2020-01-02", "2020-05-01")
    units = 100_000 / (100 * 1.001)
    proceeds = units * 100 * (1 - 0.001)                                   # 99,800.1998...
    day = res.fills.loc[res.fills.date == "2020-04-02"].set_index("side")
    assert day.loc["SELL", "cash_change"] == pytest.approx(proceeds)
    assert day.loc["BUY", "cash_change"] == pytest.approx(-proceeds)      # capped at post-sale cash
    assert res.history.loc["2020-04-02", "cash"] == pytest.approx(0.0, abs=1e-6)
    assert res.history.loc["2020-04-01", "cash"] == pytest.approx(0.0, abs=1e-6)  # nothing before the sale


# 5 --------------------------------------------------------------------------
def test_stale_marks_then_forced_exit_and_zero_recovery():
    cal = bdays("2020-01-01", "2020-02-28")
    aaa = pd.Series(100.0, index=cal)
    aaa.iloc[10:13] = np.nan                                               # 3-session gap: stale, carried
    bbb = pd.Series(50.0, index=cal)
    bbb.iloc[20:] = np.nan                                                 # quotes stop: "delisting"
    prices = make_prices(cal, AAA=aaa, BBB=bbb)
    strat = FixedOrders({cal[1]: [("AAA", 10_010.0), ("BBB", 10_010.0)]})
    for recovery, expected_cash in [(1.0, 79_980.0 + 200 * 50 * 0.999), (0.0, 79_980.0)]:
        res = Backtester(prices, strat, {}, fee_bps=FEE_BPS, initial_cash=100_000,
                         forced_exit_recovery=recovery).run(cal[1], cal[-1])
        h = res.history
        assert (h.loc[cal[10]:cal[12], "stale_marks"] == 1).all()        # AAA carried 3 sessions
        assert h.loc[cal[11], "holdings_value"] == pytest.approx(100 * 100 + 200 * 50)
        # BBB last quote cal[19]; stale on cal[20..24] (5 sessions), forced exit on cal[25]
        assert (h.loc[cal[20]:cal[24], "stale_marks"] >= 1).all()
        exit_ = res.fills.loc[(res.fills.symbol == "BBB") & (res.fills.side == "SELL")].iloc[0]
        assert exit_.date == cal[25] and exit_.reason == "no_quote_forced_exit"
        assert h.loc[cal[25], "cash"] == pytest.approx(expected_cash)
        assert (res.diagnostics.event == "forced_exit").sum() == 1


# 6 --------------------------------------------------------------------------
def test_reconciliation_on_a_multi_stock_run():
    rng = np.random.default_rng(0)
    cal = bdays("2019-01-01", "2020-12-31")
    syms = [f"S{i}" for i in range(30)]
    rets = rng.normal(0.0004, 0.02, (len(cal), len(syms)))
    prices = pd.DataFrame(100 * np.cumprod(1 + rets, axis=0), index=cal, columns=syms)
    prices.iloc[200:230, 3] = np.nan                                       # a long gap -> forced exit
    prices.iloc[100:103, 5] = np.nan                                       # a short gap -> stale marks
    fridays = cal[cal.weekday == 4]
    scores = pd.DataFrame([(d, s, 10.0 if s == "S3" else rng.normal(), "Tech" if i % 2 else "Energy", i + 1)
                           for d in fridays for i, s in enumerate(syms)],
                          columns=["date", "symbol", "score", "sector", "liq_rank"])
    strat = TopNCohortStrategy(scores, fridays, top_n=5, cohort_fraction=1 / 13, name_cap=0.05, sector_cap=0.6)
    res = Backtester(prices, strat, {}, fee_bps=FEE_BPS, initial_cash=1_000_000).run(fridays[0], cal[-1])
    h, f = res.history, res.fills
    assert reconcile(res)
    cash = 1_000_000 + f.groupby("date").cash_change.sum().reindex(h.index, fill_value=0).cumsum()
    assert np.allclose(cash, h.cash)
    assert np.allclose(h.nav - 1_000_000, h.realized_pnl_cum + h.unrealized_pnl)
    assert f.fee.sum() == pytest.approx(h.fees.sum())
    assert f.loc[f.side == "BUY", "fee"].sum() == pytest.approx(0.001 * f.loc[f.side == "BUY", "notional"].sum())
    assert (h.cash >= -1e-6).all() and (f.date > f.signal_date).all()
    lots = res.lots
    held = lots.loc[lots.exit_reason == "maturity"]
    assert ((held.exit_date - held.entry_date).dt.days >= 90).all()
    assert (lots.exit_reason == "no_quote_forced_exit").any()


# 7 --------------------------------------------------------------------------
def test_no_fill_can_use_the_signal_date_price():
    led = Ledger(1_000.0, FEE_BPS)
    with pytest.raises(AssertionError):
        led.buy("AAA", "Tech", "2020-01-02", "2020-01-02", 10.0, 100.0)  # same-day fill refused
    cal = bdays("2020-01-01", "2020-01-20")
    prices = make_prices(cal, AAA=np.arange(len(cal), dtype=float) + 10)
    res = Backtester(prices, FixedOrders({"2020-01-06": [("AAA", 500.0)]}), {}, fee_bps=FEE_BPS,
                     initial_cash=1_000).run("2020-01-02", "2020-01-20")
    buy = res.fills.iloc[0]
    assert buy.date == pd.Timestamp("2020-01-07") and buy.signal_date == pd.Timestamp("2020-01-06")
    assert buy.price == prices.loc["2020-01-07", "AAA"] != prices.loc["2020-01-06", "AAA"]


def test_buy_rejected_without_entry_quote():
    cal = bdays("2020-01-01", "2020-01-20")
    aaa = pd.Series(10.0, index=cal)
    aaa["2020-01-07"] = np.nan
    res = Backtester(make_prices(cal, AAA=aaa), FixedOrders({"2020-01-06": [("AAA", 500.0)]}), {},
                     fee_bps=FEE_BPS, initial_cash=1_000).run("2020-01-02", "2020-01-20")
    assert res.fills.empty and (res.diagnostics.event == "rejected_no_quote").sum() == 1
    assert res.history.nav.iloc[-1] == pytest.approx(1_000.0)


def test_name_cap_passes_budget_to_next_name():
    cal = bdays("2020-01-01", "2020-03-31")
    prices = make_prices(cal, AAA=100.0, BBB=100.0, CCC=100.0)
    fridays = cal[cal.weekday == 4][:6]
    scores = pd.DataFrame([(d, s, sc, "Tech", 1) for d in fridays for s, sc in [("AAA", 3), ("BBB", 2), ("CCC", 1)]],
                          columns=["date", "symbol", "score", "sector", "liq_rank"])
    strat = TopNCohortStrategy(scores, fridays, top_n=1, cohort_fraction=0.04, name_cap=0.05, sector_cap=1.0)
    res = Backtester(prices, strat, {}, fee_bps=0, initial_cash=1_000_000).run(fridays[0], cal[-1])
    buys = res.fills.loc[res.fills.side == "BUY"].groupby("symbol").notional.sum()
    assert buys["AAA"] == pytest.approx(50_000.0)                          # 4% + 1% (capped at 5%)
    assert buys["BBB"] > 0                                                 # remainder passed down
