import numpy as np
import pandas as pd
import pytest

from backtester import Portfolio, TopNStrategy, run_backtest, performance

def make_market():
    """2 tickers, B has no quote on maturity day"""
    days = pd.DatetimeIndex(["2020-01-02", "2020-01-03", "2020-04-02", "2020-04-03", "2020-04-06"])
    quotes = pd.DataFrame({"A": [10.0, 10.0, 12.0, 12.0, 12.0],
                           "B": [20.0, 20.0, np.nan, 25.0, 25.0]}, index=days)
    scores = pd.DataFrame({"date": [days[0]] * 2, "symbol": ["A", "B"],
                           "score": [0.2, 0.1], "close": [10.0, 20.0]})
    return quotes, scores

def test_buy_and_sell_by_hand():
    p = Portfolio(initial_cash=1000, fee_bps=10)
    lot = p.buy("A", budget=100.1, price=10.0, date=pd.Timestamp("2020-01-03"))
    assert lot.units == pytest.approx(10.0)            # notional 100 / price 10
    assert p.cash == pytest.approx(899.9)               # 1000 - 100 - 0.10 fee
    assert lot.exit_date == pd.Timestamp("2020-04-02")   # 90 calendar days
    pnl = p.sell(lot.lot_id, price=12.0, date=pd.Timestamp("2020-04-02"))
    assert p.cash == pytest.approx(899.9 + 120 - 0.12)
    assert pnl == pytest.approx(120 - 0.12 - 100.1)

def test_cannot_overspend():
    p = Portfolio(initial_cash=100)
    with pytest.raises(ValueError):
        p.buy("A", 100.01, 10.0, pd.Timestamp("2020-01-03"))


def test_full_run_by_hand():
    quotes, scores = make_market()
    p = Portfolio(initial_cash=1000, fee_bps=10)
    strat = TopNStrategy(scores, top_n=2, cohort_fraction=0.5, min_price=5, min_order=0)
    history, fills, open_lots, log = run_backtest(quotes, strat, p)

    # budget min(1000, 0.5*1000) = 500, 250 each, filled on 2020-01-03
    buys = fills[fills.side == "buy"].set_index("symbol")
    assert list(buys.date.unique()) == [pd.Timestamp("2020-01-03")]   # next session, not signal day
    assert buys.loc["A", "units"] == pytest.approx(250 / 1.001 / 10)
    assert history.loc["2020-01-03", "cash"] == pytest.approx(500)

    # 3 month passed, sell A, B does not have closing px
    sells = fills[fills.side == "sell"].set_index("symbol")
    assert sells.loc["A", "date"] == pd.Timestamp("2020-04-02")
    assert sells.loc["B", "date"] == pd.Timestamp("2020-04-03")
    assert "exit_waiting" in set(log.event)

    # on 04-02, B marked at its last price (20)
    units_b = 250 / 1.001 / 20
    a_proceeds = 250 / 1.001 / 10 * 12 * 0.999
    assert history.loc["2020-04-02", "nav"] == pytest.approx(500 + a_proceeds + units_b * 20)

    # end: sell everything, pos flat, NAV == cash == initial + total PnL
    assert len(open_lots) == 0
    assert history.nav.iloc[-1] == pytest.approx(history.cash.iloc[-1])
    assert history.nav.iloc[-1] == pytest.approx(1000 + fills.pnl.sum())


def test_performance_simple():
    nav = pd.Series([100.0, 110.0, 99.0], index=pd.to_datetime(["2020-01-01", "2020-07-01", "2021-01-01"]))
    m = performance(nav)
    assert m.max_drawdown == pytest.approx(99 / 110 - 1)
    assert m.final_value == 99.0