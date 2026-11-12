"""Hand-checkable accounting and timing tests.  Run: ``python test_backtester.py``
(also works under pytest).  Every expected number below is worked out by hand in
the comments, so the backtester is checked against arithmetic, not against itself.
"""

import numpy as np
import pandas as pd

from backtester import Account, Backtester, TopNStrategy, performance

FEE = 10 / 10_000


class FixedOrders:
    """Strategy stub: {signal_date: {symbol: budget}}; records what it was shown."""
    def __init__(self, plan):
        self.plan = {pd.Timestamp(k): v for k, v in plan.items()}
        self.seen = []

    def orders(self, signal_date, snapshot):
        self.seen.append((signal_date, snapshot.cash, snapshot.nav))
        return pd.Series(self.plan.get(signal_date, {}), dtype=float)


def make_prices(dates, **cols):
    return pd.DataFrame(cols, index=pd.DatetimeIndex(dates), dtype=float)


def test_single_round_trip_by_hand():
    # A simple Monday-Friday calendar.
    cal = pd.bdate_range("2021-01-04", "2021-04-30")
    px = make_prices(cal, AAA=np.where(cal < "2021-04-05", 100.0, 110.0))
    # Signal Mon 2021-01-04 -> buy Tue 2021-01-05 at 100 with a $10,000 budget.
    bt = Backtester(px, FixedOrders({"2021-01-04": {"AAA": 10_000}}), fee_bps=10)
    res = bt.run("2021-01-05", "2021-04-30")
    buy, sell = res.fills.iloc[0], res.fills.iloc[1]
    # units = 10,000 / (100 * 1.001) = 99.9000999; notional 9,990.01; fee 9.99
    assert np.isclose(buy.units, 10_000 / 100.1)
    assert np.isclose(buy.notional + buy.fee, 10_000)
    # Due Tue 2021-01-05 + 90 days = Mon 2021-04-05 (a session) -> sell at 110 that close.
    assert sell.date == pd.Timestamp("2021-04-05") and sell.price == 110
    # Proceeds 99.9001 * 110 = 10,989.01, minus 0.1% fee = 10,978.02.
    proceeds = 10_000 / 100.1 * 110 * (1 - FEE)
    assert np.isclose(sell.cash_change, proceeds)
    assert np.isclose(res.account.cash, 1_000_000 - 10_000 + proceeds)
    # PnL = 10,978.02 - 10,000 = 978.02 and it reconciles to the change in NAV.
    assert np.isclose(sell.pnl, proceeds - 10_000)
    assert np.isclose(res.history.nav.iloc[-1] - 1_000_000, sell.pnl)
    # Day of purchase: NAV = 1,000,000 - fee (9.99): fees show up immediately.
    assert np.isclose(res.history.nav.loc["2021-01-05"], 1_000_000 - 10_000 / 100.1 * 100 * FEE)


def test_maturity_on_weekend_and_missing_quote():
    cal = pd.bdate_range("2021-01-04", "2021-05-31")
    a = pd.Series(50.0, index=cal)
    # Buy Fri 2021-01-08 -> due Jan 8 + 90 days = Thu 2021-04-08.
    # Make Apr 8 and Apr 9 have NO quote for AAA: exit must wait for Mon Apr 12.
    a.loc[["2021-04-08", "2021-04-09"]] = np.nan
    a.loc["2021-04-12":] = 60.0
    px = make_prices(cal, AAA=a.values)
    bt = Backtester(px, FixedOrders({"2021-01-07": {"AAA": 5_000}}))
    res = bt.run("2021-01-05", "2021-05-31")
    sell = res.fills.query("side == 'SELL'").iloc[0]
    assert sell.date == pd.Timestamp("2021-04-12") and sell.price == 60
    assert (res.events.event == "exit_delayed_no_quote").sum() == 2

    # Weekend maturity: signal Fri Jan 8 -> buy Mon Jan 11 -> due Sun Apr 11 -> sell Mon Apr 12.
    px2 = make_prices(cal, BBB=50.0)
    res2 = Backtester(px2, FixedOrders({"2021-01-08": {"BBB": 1_000}})).run("2021-01-05", "2021-05-31")
    s2 = res2.fills.query("side == 'SELL'").iloc[0]
    assert pd.Timestamp("2021-01-11") + pd.Timedelta(days=90) == pd.Timestamp("2021-04-11")   # a Sunday
    assert s2.date == pd.Timestamp("2021-04-12")


def test_no_quote_on_entry_cancels_order():
    cal = pd.bdate_range("2021-01-04", "2021-01-29")
    a = pd.Series(10.0, index=cal)
    a.loc["2021-01-06"] = np.nan
    res = Backtester(make_prices(cal, AAA=a.values), FixedOrders({"2021-01-05": {"AAA": 1_000}})).run("2021-01-05", "2021-01-29")
    assert res.fills.empty and res.account.cash == 1_000_000
    assert res.events.event.tolist() == ["buy_cancelled_no_quote"]


def test_cash_is_never_spent_twice():
    acct = Account(1_000, fee_bps=10)
    d0, d1 = pd.Timestamp("2021-01-04"), pd.Timestamp("2021-01-05")
    acct.buy("AAA", 600, 10, d1, d0)
    try:
        acct.buy("BBB", 600, 10, d1, d0)   # only $400 left
    except ValueError:
        pass
    else:
        raise AssertionError("overspending was allowed")
    # A strategy that plans more than its signal-date cash is stopped by the loop.
    cal = pd.bdate_range("2021-01-04", "2021-01-15")
    px = make_prices(cal, AAA=10.0, BBB=10.0)
    try:
        Backtester(px, FixedOrders({"2021-01-04": {"AAA": 700_000, "BBB": 700_000}})).run("2021-01-05", "2021-01-15")
    except ValueError:
        pass
    else:
        raise AssertionError("plan larger than cash was accepted")


def test_strategy_only_sees_previous_close_and_ledger_reconciles():
    cal = pd.bdate_range("2021-01-04", "2021-12-31")
    rng = np.random.default_rng(0)
    cols = {s: 20 * np.exp(np.cumsum(rng.normal(0, 0.02, len(cal)))) for s in "ABCDEFGH"}
    px = make_prices(cal, **cols)
    sig = cal[cal.weekday == 4][:-1]          # Fridays
    scores = pd.DataFrame([(d, s, rng.normal(), "Tech" if s < "E" else "Energy") for d in sig for s in cols],
                          columns=["date", "symbol", "score", "sector"])
    strat = TopNStrategy(scores, scores.drop_duplicates("symbol").set_index("symbol").sector, top_n=3,
                         cohort_fraction=1 / 12, name_cap=0.2, sector_cap=0.6)
    res = Backtester(px, strat).run(cal[1], cal[-1])
    f = res.fills
    # Every buy was decided at the immediately preceding session.
    buys = f.query("side == 'BUY'")
    assert (buys.signal_date.map(cal.get_loc) + 1 == buys.date.map(cal.get_loc)).all()
    # Every sell is the first session on/after maturity (no quotes are missing here).
    sells = f.query("side == 'SELL'")
    first_ok = sells.due_date.map(lambda d: cal[cal.searchsorted(d)])
    assert (sells.date == first_ok).all()
    # Ledger: initial cash + sum of cash changes == final cash.
    assert np.isclose(1_000_000 + f.cash_change.sum(), res.account.cash)
    # NAV identity on the final day: cash + units * last close.
    last = px.iloc[-1]
    val = sum(l.units * last[l.symbol] for l in res.account.lots.values())
    assert np.isclose(res.history.nav.iloc[-1], res.account.cash + val)
    # Total PnL = realized + unrealized (net of the open lots' entry fees).
    unreal = sum(l.units * (last[l.symbol] - l.entry_price) - l.entry_fee for l in res.account.lots.values())
    assert np.isclose(res.history.nav.iloc[-1] - 1_000_000, sells.pnl.sum() + unreal)
    assert (res.history.cash >= 0).all()


def test_performance_metrics_by_hand():
    idx = pd.to_datetime(["2020-01-01", "2021-01-01", "2022-01-01"])
    nav = pd.Series([1_000_000, 1_200_000, 900_000], index=idx)
    p = performance(nav)
    years = (idx[-1] - idx[0]).days / 365.25          # 731 / 365.25
    assert np.isclose(p.CAGR, 0.9 ** (1 / years) - 1)
    assert np.isclose(p.max_drawdown, 900 / 1200 - 1)  # -25%


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS {name}")
