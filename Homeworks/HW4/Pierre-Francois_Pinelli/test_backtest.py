"""Accounting and timing checks on small cases computed by hand.

Run from this folder: python3 test_backtest.py
"""

import numpy as np
import pandas as pd

from hw4_backtest import benchmark_nav, performance, run_backtest

DAYS = pd.bdate_range("2015-01-01", "2015-08-31")  # Mon-Fri sessions


def flat_prices(*symbols):
    """One column per stock, price 100 every day unless the test changes it."""
    return pd.DataFrame(100.0, index=DAYS, columns=list(symbols))


def scores_for(date, *symbols):
    return pd.DataFrame({"date": pd.Timestamp(date), "symbol": list(symbols),
                         "score": np.arange(len(symbols), 0, -1, dtype=float)})


def test_buy_next_close_and_sell_after_90_days():
    prices = flat_prices("A")
    prices.loc["2015-01-02", "A"] = 50.0     # signal day close: must NOT be the fill price
    prices.loc["2015-01-05", "A"] = 100.0    # next session: fill here
    prices.loc["2015-04-06":, "A"] = 120.0   # Jan 5 + 90 days = Apr 5 (Sunday) -> sell Monday Apr 6
    r = run_backtest(prices, scores_for("2015-01-02", "A"), "2015-01-05", "2015-05-29",
                     top_n=1, cohort_fraction=1.0, cost_bps=0)
    buy, sell = r["trades"].iloc[0], r["trades"].iloc[1]
    assert buy.date == pd.Timestamp("2015-01-05") and buy.price == 100.0 and buy.signal_date < buy.date
    assert np.isclose(buy.units, 10_000)                      # 1,000,000 / 100
    assert sell.date == pd.Timestamp("2015-04-06") and sell.price == 120.0
    assert np.isclose(sell.cash_after, 1_200_000)              # 10,000 x 120
    assert np.isclose(r["history"].nav.iloc[-1], 1_200_000)
    assert r["history"].index[0] == pd.Timestamp("2015-01-02")  # starting cash on the signal day


def test_proportional_costs():
    prices = flat_prices("A")
    prices.loc["2015-04-06":, "A"] = 110.0
    r = run_backtest(prices, scores_for("2015-01-02", "A"), "2015-01-05", "2015-05-29",
                     top_n=1, cohort_fraction=1.0, cost_bps=10, reinvest=False)
    buy, sell = r["trades"].iloc[0], r["trades"].iloc[1]
    units = 1_000_000 / (100 * 1.001)                           # budget pays price + 0.1% cost
    assert np.isclose(buy.units, units) and np.isclose(buy.cost, units * 100 * 0.001)
    assert abs(buy.cash_after) < 1e-6                           # all cash used, not more
    expected = 1_000_000 * 1.1 * 0.999 / 1.001                  # = 1,097,802.20
    assert np.isclose(sell.cash_after, expected)
    assert np.isclose(sell.pnl, expected - 1_000_000)
    assert np.isclose(r["history"].costs.sum(), buy.cost + sell.cost)


def test_cohorts_never_spend_cash_twice():
    prices = flat_prices("A", "B", "C")
    scores = pd.concat([scores_for("2015-01-30", "A", "B"), scores_for("2015-02-27", "C", "A"),
                        scores_for("2015-03-31", "B", "C")])
    r = run_backtest(prices, scores, "2015-02-02", "2015-04-30", top_n=2, cohort_fraction=0.5, cost_bps=0)
    t, h = r["trades"], r["history"]
    assert (h.cash >= 0).all()
    feb, mar = t[t.date == "2015-02-02"], t[t.date == "2015-03-02"]
    assert np.allclose(feb.value, 250_000)                    # half of $1M split over 2 stocks
    assert np.allclose(mar.value, 250_000)                    # min(cash 500k, NAV/2 = 500k)
    assert (t.date == "2015-04-01").sum() == 0                 # no cash left in April: no buy
    assert np.isclose(h.nav.iloc[-1], 1_000_000)               # flat prices, no costs


def test_due_date_without_quote_sells_at_next_quote():
    prices = flat_prices("A")
    prices.loc["2015-04-06":"2015-04-08", "A"] = np.nan        # no quote on the due date and 2 days after
    prices.loc["2015-04-09":, "A"] = 90.0
    r = run_backtest(prices, scores_for("2015-01-02", "A"), "2015-01-05", "2015-05-29",
                     top_n=1, cohort_fraction=1.0, cost_bps=0)
    sell = r["trades"].iloc[1]
    assert sell.date == pd.Timestamp("2015-04-09") and np.isclose(sell.cash_after, 900_000)
    assert r["events"].event.str.contains("exit delayed").any()
    # while frozen, the lot keeps its last mark (100), so NAV stays 1,000,000
    assert np.isclose(r["history"].loc["2015-04-08", "nav"], 1_000_000)


def test_missing_quotes_freeze_or_write_off():
    prices = flat_prices("A", "B")
    prices.loc["2015-02-02":, "B"] = np.nan                     # B stops trading after Jan 30
    scores = scores_for("2015-01-02", "A", "B")
    frozen = run_backtest(prices, scores, "2015-01-05", "2015-02-27", top_n=2, cohort_fraction=1.0, cost_bps=0)
    written = run_backtest(prices, scores, "2015-01-05", "2015-02-27", top_n=2, cohort_fraction=1.0,
                           cost_bps=0, missing_policy="write_off")
    assert np.isclose(frozen["history"].nav.iloc[-1], 1_000_000)      # B kept at its last close
    assert len(frozen["open_lots"]) == 2
    assert np.isclose(written["history"].nav.iloc[-1], 500_000)       # B worth 0 after 5 missing sessions
    off = written["events"][written["events"].event.str.contains("written off")]
    assert off.date.iloc[0] == pd.Timestamp("2015-02-09")              # 6th session without a quote


def test_open_lots_valued_at_the_end():
    prices = flat_prices("A")
    prices.loc["2015-02-27", "A"] = 130.0
    r = run_backtest(prices, scores_for("2015-01-02", "A"), "2015-01-05", "2015-02-27",
                     top_n=1, cohort_fraction=1.0, cost_bps=0)
    lot = r["open_lots"].iloc[0]
    assert lot.mark == 130.0 and np.isclose(r["history"].nav.iloc[-1], 10_000 * 130)
    assert r["history"].index[-1] == pd.Timestamp("2015-02-27")       # no later prices used


def test_sale_proceeds_reinvested_next_close():
    prices = flat_prices("A", "B")
    prices.loc["2015-04-06":, "A"] = 120.0
    scores = scores_for("2015-01-02", "A", "B").assign(score=[2.0, 1.0])  # latest scores: A first
    r = run_backtest(prices, scores, "2015-01-05", "2015-05-29", top_n=1, cohort_fraction=1.0, cost_bps=0)
    t = r["trades"]
    rebuy = t[(t.side == "BUY") & (t.date == "2015-04-07")].iloc[0]  # day after the Apr 6 sale
    assert rebuy.symbol == "A" and np.isclose(rebuy.value, 1_200_000) and rebuy.signal_date == pd.Timestamp("2015-04-06")
    assert abs(rebuy.cash_after) < 1e-6
    off = run_backtest(prices, scores, "2015-01-05", "2015-05-29", top_n=1, cohort_fraction=1.0, cost_bps=0, reinvest=False)
    assert np.isclose(off["history"].cash.iloc[-1], 1_200_000)  # without the rule the cash waits


def test_bad_print_is_ignored():
    prices = flat_prices("A")
    prices.loc["2015-02-10", "A"] = 1.0       # one wrong quote, 100x too low
    prices.loc["2015-03-02":, "A"] = 1500.0   # unadjusted 15-for-1 reverse split from March 2
    r = run_backtest(prices, scores_for("2015-01-02", "A"), "2015-01-05", "2015-05-29",
                     top_n=1, cohort_fraction=1.0, cost_bps=0)
    h, ev = r["history"], r["events"]
    assert np.isclose(h.loc["2015-02-10", "nav"], 1_000_000)          # mark stays at the last good close
    assert ev.event.str.contains("bad print").sum() == 2               # each episode logged once
    assert (r["trades"].side == "SELL").sum() == 0                      # never sold at a broken price
    assert np.isclose(h.nav.iloc[-1], 1_000_000)                       # frozen at 100 (write_off would give 0)


def test_benchmark_and_metrics():
    history = pd.DataFrame(index=DAYS[:3])
    bench = pd.Series([90.0, 100.0, 110.0], index=DAYS[:3])
    nav = benchmark_nav(bench, history)
    assert list(nav.round(6)) == [1_000_000, 1_000_000, 1_100_000]   # invested at the first trading close
    one_year = pd.Series([100.0, 110.0], index=pd.to_datetime(["2015-01-01", "2016-01-01"]))
    p = performance(one_year)
    assert np.isclose(p.CAGR, 1.1 ** (365.25 / 365) - 1) and p.max_drawdown == 0
    dd = performance(pd.Series([100.0, 120, 90, 130], index=DAYS[:4]))
    assert np.isclose(dd.max_drawdown, 90 / 120 - 1)


if __name__ == "__main__":
    tests = [f for name, f in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print("ok  ", test.__name__)
    print(f"all {len(tests)} accounting checks passed")
