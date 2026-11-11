"""Hand checks for cash, fees, the fixed budget, and a weekend maturity."""

import pandas as pd

from backtest import run_backtest

FEE = 0.001
CAPITAL = 1_000_000.0


def prices_from(rows):
    frame = pd.DataFrame(rows)
    frame["date"] = pd.to_datetime(frame["date"])
    return frame.pivot(index="date", columns="symbol", values="adj_close").sort_index()


def test_one_round_trip_reserves_the_fee_inside_the_budget():
    # December close is the signal. The January fill is at 100, and the lot
    # matures on Sunday 2016-04-03, so it sells at the Monday close.
    px = prices_from([
        {"date": "2015-12-31", "symbol": "AAA", "adj_close": 100.0},
        {"date": "2016-01-04", "symbol": "AAA", "adj_close": 100.0},
        {"date": "2016-04-04", "symbol": "AAA", "adj_close": 110.0},
    ])
    scores = pd.DataFrame([{"date": "2015-12-31", "symbol": "AAA", "score": 0.2}])
    result = run_backtest(px, scores, "2016-01-04", "2016-04-04", top_n=10, fee_rate=FEE, capital=CAPITAL)
    buy = result["fills"].iloc[0]
    budget = CAPITAL / 3
    assert buy.side == "buy"
    assert abs((buy.notional + buy.fee) - budget) < 1e-6
    assert abs(buy.units - budget / (100 * (1 + FEE))) < 1e-8
    sell = result["fills"].iloc[1]
    assert sell.date == pd.Timestamp("2016-04-04")
    assert sell.due_date == pd.Timestamp("2016-04-03")
    proceeds = sell.units * 110 * (1 - FEE)
    assert abs(sell.cash_change - proceeds) < 1e-6
    assert abs(result["cash"] - (CAPITAL - budget + proceeds)) < 1e-4
    assert result["open_lots"] == []


def test_sale_today_does_not_increase_yesterdays_budget():
    # Fully invested after the first buy, so the next month's budget is the leftover cash,
    # even though a sale on the fill date frees a lot of cash.
    # The January buy spends 90% of NAV. The February buy is planned at the January
    # close, while that cash is still invested. The lot matures on the February fill,
    # so the sale frees cash on the same day without enlarging the planned budget.
    dates = ["2015-12-31", "2016-01-04", "2016-01-29", "2016-02-01"]
    px = prices_from([{"date": d, "symbol": "AAA", "adj_close": 50.0} for d in dates])
    scores = pd.DataFrame([
        {"date": "2015-12-31", "symbol": "AAA", "score": 1.0},
        {"date": "2016-01-29", "symbol": "AAA", "score": 1.0},
    ])
    # Force the first cohort to spend almost all cash by using a fraction of 1, then the
    # second order is planned while cash is ~0. Use fraction 0.9 so some cash remains.
    result = run_backtest(px, scores, "2016-01-04", "2016-02-01", top_n=1, cohort_fraction=0.9, fee_rate=FEE, capital=CAPITAL, holding_days=28)
    buys = result["fills"].query("side == 'buy'")
    assert len(buys) == 2
    first_spend = -(buys.iloc[0].cash_change)
    cash_after_first = CAPITAL - first_spend
    assert abs(buys.iloc[1].notional + buys.iloc[1].fee - cash_after_first) < 1e-4


def test_maturity_on_a_weekend_sells_the_next_session():
    # A Monday fill plus 90 calendar days lands on Sunday, so the sale waits for Monday.
    px = prices_from([
        {"date": "2015-12-31", "symbol": "AAA", "adj_close": 10.0},
        {"date": "2016-01-04", "symbol": "AAA", "adj_close": 10.0},
        {"date": "2016-04-04", "symbol": "AAA", "adj_close": 12.0},
    ])
    scores = pd.DataFrame([{"date": "2015-12-31", "symbol": "AAA", "score": 1.0}])
    result = run_backtest(px, scores, "2016-01-04", "2016-04-04", top_n=1, cohort_fraction=1, fee_rate=0, capital=1000, holding_days=90)
    sell = result["fills"].loc[result["fills"]["side"].eq("sell")].iloc[0]
    assert sell.due_date == pd.Timestamp("2016-04-03")
    assert sell.date == pd.Timestamp("2016-04-04")
    assert abs(result["cash"] - 1200) < 1e-6


if __name__ == "__main__":
    test_one_round_trip_reserves_the_fee_inside_the_budget()
    test_sale_today_does_not_increase_yesterdays_budget()
    test_maturity_on_a_weekend_sells_the_next_session()
    print("hand checks ok")
