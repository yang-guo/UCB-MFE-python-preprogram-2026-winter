"""Hand-checkable tests for backtester.py. Run with: python -m pytest -q"""

import numpy as np
import pandas as pd
import pytest

from backtester import (Config, Ledger, Strategy, TopNStrategy, audit, benchmark_curve,
                        performance, run_backtest)

CAL = pd.bdate_range("2020-01-01", periods=140)
FREE = Config(fee_bps=0)          # world rules without fees, for easy hand checks


def make_prices(**columns):
    return pd.DataFrame(columns, index=CAL, dtype=float)


def make_scores(rows):
    """rows: {signal_date: {symbol: score}}"""
    return pd.DataFrame(rows).T.sort_index()


# ---------- ledger arithmetic, checked by hand ----------

def test_one_lot_by_hand():
    # Fee 10 bps. Spend $100,000 (fee included) at $50, sell at $55.
    ledger = Ledger(cash=1_000_000, fee_rate=0.001)
    lot_id = ledger.buy("A", "2020-01-02", "2020-01-01", price=50, dollars=100_000, hold_days=90)
    units = 100_000 / (50 * 1.001)                      # = 1998.001998...
    assert ledger.lots[lot_id].units == pytest.approx(units)
    assert ledger.lots[lot_id].entry_fee == pytest.approx(units * 50 * 0.001)   # about 99.90
    assert ledger.cash == pytest.approx(900_000)         # the whole $100,000 left cash
    pnl = ledger.sell(lot_id, "2020-04-01", price=55)
    sale_fee = units * 55 * 0.001
    assert pnl == pytest.approx(units * 5 - units * 50 * 0.001 - sale_fee)       # about 9,780.22
    assert ledger.cash == pytest.approx(900_000 + units * 55 - sale_fee)
    assert not ledger.lots


def test_cannot_spend_cash_twice():
    ledger = Ledger(cash=100_000, fee_rate=0.0)
    ledger.buy("A", "2020-01-02", "2020-01-01", 10, 100_000, 90)
    with pytest.raises(ValueError):
        ledger.buy("B", "2020-01-02", "2020-01-01", 10, 1, 90)


def test_buy_needs_earlier_signal_and_sale_needs_maturity():
    ledger = Ledger(cash=100_000, fee_rate=0.0)
    with pytest.raises(ValueError):
        ledger.buy("A", "2020-01-02", "2020-01-02", 10, 1_000, 90)   # same-day signal
    lot = ledger.buy("A", "2020-01-02", "2020-01-01", 10, 1_000, 90)
    with pytest.raises(ValueError):
        ledger.sell(lot, "2020-02-01", 10)                            # day 30, too early


# ---------- full daily loop on a tiny market ----------

def one_stock_setup():
    entry = CAL[1]
    due = entry + pd.Timedelta(days=90)
    price = pd.Series(100.0, index=CAL)
    price[CAL >= due] = 120.0
    prices = make_prices(A=price)
    scores = make_scores({CAL[0]: {"A": 1.0}})
    strat = TopNStrategy(top_n=1, cohort_fraction=0.5, entry_every=10_000, max_name_weight=1.0)
    return prices, scores, strat, entry, due


def test_entry_next_session_exit_at_90_days_and_cash():
    prices, scores, strat, entry, due = one_stock_setup()
    res = run_backtest(prices, scores, strat, CAL[1], CAL[-1], FREE)
    buy, sell = res.trades.iloc[0], res.trades.iloc[1]
    assert buy.side == "BUY" and buy.date == entry and buy.signal_date == CAL[0]
    assert buy.units == pytest.approx(500_000 / 100)                  # half of $1,000,000 at $100
    first_session_on_or_after_due = CAL[CAL >= due][0]
    assert sell.side == "SELL" and sell.date == first_session_on_or_after_due
    assert res.nav.cash.iloc[-1] == pytest.approx(500_000 + 5_000 * 120)   # $1,100,000
    assert res.open_lots.empty
    assert all(audit(res, FREE).values())


def test_no_lookahead_in_signals():
    # B has the best score only on the ENTRY day itself, so it must not be bought.
    prices = make_prices(A=100.0, B=100.0)
    scores = make_scores({CAL[0]: {"A": 1.0, "B": 0.0}, CAL[1]: {"A": 0.0, "B": 99.0}})
    strat = TopNStrategy(top_n=1, cohort_fraction=0.5, entry_every=10_000, max_name_weight=1.0)
    res = run_backtest(prices, scores, strat, CAL[1], CAL[30], FREE)
    assert set(res.trades.symbol) == {"A"}


def test_open_positions_are_valued_at_the_end_and_not_sold():
    prices, scores, strat, entry, due = one_stock_setup()
    end = CAL[CAL < due][-1]                                           # last session before maturity
    res = run_backtest(prices, scores, strat, CAL[1], end, FREE)
    assert (res.trades.side == "SELL").sum() == 0
    assert len(res.open_lots) == 1
    assert res.nav.holdings.iloc[-1] == pytest.approx(5_000 * 100)
    assert res.nav.nav.iloc[-1] == pytest.approx(1_000_000)


def test_missing_quote_on_due_date_delays_the_sale_without_inventing_a_fill():
    prices, scores, strat, entry, due = one_stock_setup()
    first_due_session = CAL[CAL >= due][0]
    prices.loc[first_due_session, "A"] = np.nan
    res = run_backtest(prices, scores, strat, CAL[1], CAL[-1], FREE)
    sell = res.trades[res.trades.side == "SELL"].iloc[0]
    assert sell.date == CAL[CAL > first_due_session][0]
    assert (res.events.event == "sale_delayed_no_quote").sum() == 1
    assert res.events.iloc[0].date == first_due_session


def test_stale_mark_keeps_last_price_for_nav():
    prices, scores, strat, entry, due = one_stock_setup()
    prices.loc[CAL[5], "A"] = np.nan                                   # one missing quote while held
    res = run_backtest(prices, scores, strat, CAL[1], CAL[10], FREE)
    assert res.nav.loc[CAL[5], "holdings"] == pytest.approx(5_000 * 100)
    assert res.nav.loc[CAL[5], "stale_marks"] == 1


def test_entering_every_day_never_overspends_and_audit_passes():
    prices = make_prices(A=100.0, B=50.0, C=20.0)
    rows = {d: {"A": 3.0, "B": 2.0, "C": 1.0} for d in CAL[:-1]}
    cfg = Config(fee_bps=10)
    strat = TopNStrategy(top_n=3, cohort_fraction=1.0, entry_every=1, max_name_weight=1.0)
    res = run_backtest(prices, make_scores(rows), strat, CAL[1], CAL[-1], cfg)
    assert res.nav.cash.min() >= -1e-6
    assert all(audit(res, cfg).values())


def test_name_cap_limits_one_stock():
    prices = make_prices(A=100.0)
    scores = make_scores({CAL[0]: {"A": 1.0}})
    strat = TopNStrategy(top_n=1, cohort_fraction=1.0, entry_every=10_000, max_name_weight=0.05)
    res = run_backtest(prices, scores, strat, CAL[1], CAL[3], FREE)
    assert res.trades.iloc[0].notional == pytest.approx(50_000)        # 5% of $1,000,000


# ---------- metrics ----------

def test_max_drawdown_by_hand():
    nav = pd.Series([100, 120, 90, 110.0], index=pd.bdate_range("2020-01-01", periods=4))
    assert performance(nav).max_drawdown == pytest.approx(90 / 120 - 1)   # -25%


def test_cagr_by_hand():
    idx = pd.DatetimeIndex(["2020-01-01", "2020-07-01", "2021-01-01"])
    nav = pd.Series([100.0, 150.0, 200.0], index=idx)
    years = 366 / 365.25
    assert performance(nav).cagr == pytest.approx(2 ** (1 / years) - 1)
    assert performance(nav).final_value == 200.0


def test_sharpe_by_hand():
    nav = pd.Series([100, 110, 99, 108.9], index=pd.bdate_range("2020-01-01", periods=4))
    r = np.array([0.1, -0.1, 0.1])
    rf = 1.02 ** (1 / 252) - 1
    expected = np.sqrt(252) * (r.mean() - rf) / r.std(ddof=1)
    assert performance(nav).sharpe == pytest.approx(expected)


def test_benchmark_buys_at_first_test_close():
    idx = pd.bdate_range("2020-01-01", periods=4)                       # idx[0] is the initial row
    bench = pd.Series([90.0, 100.0, 110.0, 99.0], index=idx)
    curve = benchmark_curve(bench, idx, 1_000_000)
    assert curve.iloc[0] == 1_000_000 and curve.iloc[1] == 1_000_000
    assert curve.iloc[2] == pytest.approx(1_100_000) and curve.iloc[3] == pytest.approx(990_000)


# ---------- the border between strategy and backtester ----------

class FixedOrder(Strategy):
    """A strategy that always asks for the same dollar order. It ignores the scores."""

    def __init__(self, dollars, symbol="A"):
        self.dollars, self.symbol = dollars, symbol

    def orders(self, signal_date, session_number, scores, portfolio):
        return {self.symbol: self.dollars} if session_number == 0 else {}


def test_backtester_refuses_orders_larger_than_cash():
    prices = make_prices(A=100.0)
    with pytest.raises(ValueError):
        run_backtest(prices, make_scores({CAL[0]: {"A": 1.0}}), FixedOrder(2_000_000), CAL[1], CAL[5], FREE)


def test_backtester_refuses_negative_orders():
    prices = make_prices(A=100.0)
    with pytest.raises(ValueError):
        run_backtest(prices, make_scores({CAL[0]: {"A": 1.0}}), FixedOrder(-1_000), CAL[1], CAL[5], FREE)


def test_strategy_sees_only_the_previous_session_row():
    seen = []

    class Recorder(Strategy):
        def orders(self, signal_date, session_number, scores, portfolio):
            seen.append((signal_date, scores.name, portfolio.cash))
            return {}

    prices = make_prices(A=100.0)
    rows = {d: {"A": float(i)} for i, d in enumerate(CAL)}
    run_backtest(prices, make_scores(rows), Recorder(), CAL[3], CAL[6], FREE)
    assert [s[0] for s in seen] == list(CAL[2:6])                     # the session before each test day
    assert all(s[0] == s[1] for s in seen)                            # the row is that day's row only


def test_any_strategy_plugs_into_the_same_backtester():
    prices = make_prices(A=100.0)
    res = run_backtest(prices, make_scores({CAL[0]: {"A": 1.0}}), FixedOrder(10_000), CAL[1], CAL[5], FREE)
    assert res.trades.iloc[0].notional == pytest.approx(10_000)
    assert all(audit(res, FREE).values())


# ---------- round 2: the strategy knows how old its positions are ----------

def view(days_since_buy=None, cash=1_000_000.0, nav=1_000_000.0, exposure=None):
    from backtester import PortfolioView
    return PortfolioView(cash=cash, nav=nav, exposure=exposure or {}, days_since_buy=days_since_buy or {})


def test_default_strategy_ignores_position_age():
    # Without a wait rule, the best scores are bought again even when the position is new.
    scores = pd.Series({"A": 3.0, "B": 2.0, "C": 1.0})
    strat = TopNStrategy(top_n=2, cohort_fraction=0.4, entry_every=1, max_name_weight=1.0)
    orders = strat.orders(CAL[0], 0, scores, view(days_since_buy={"A": 1}))
    assert set(orders) == {"A", "B"}


def test_stock_bought_yesterday_is_skipped_and_the_next_best_takes_its_place():
    scores = pd.Series({"A": 3.0, "B": 2.0, "C": 1.0})
    strat = TopNStrategy(top_n=2, cohort_fraction=0.4, entry_every=1, max_name_weight=1.0, rebuy_wait_days=45)
    orders = strat.orders(CAL[0], 0, scores, view(days_since_buy={"A": 1}))
    assert set(orders) == {"B", "C"}                                  # A skipped, C moves up
    assert sum(orders.values()) == pytest.approx(400_000)            # the whole budget is still spent


def test_stock_can_be_bought_again_when_its_newest_lot_has_aged():
    scores = pd.Series({"A": 3.0, "B": 2.0, "C": 1.0})
    strat = TopNStrategy(top_n=2, cohort_fraction=0.4, entry_every=1, max_name_weight=1.0, rebuy_wait_days=45)
    assert set(strat.orders(CAL[0], 0, scores, view(days_since_buy={"A": 44}))) == {"B", "C"}
    assert set(strat.orders(CAL[0], 0, scores, view(days_since_buy={"A": 45}))) == {"A", "B"}
    assert set(strat.orders(CAL[0], 0, scores, view(days_since_buy={"A": 85}))) == {"A", "B"}


def test_ledger_reports_the_age_of_the_newest_lot():
    ledger = Ledger(cash=1_000_000, fee_rate=0.0)
    ledger.buy("A", "2020-01-02", "2020-01-01", 10, 1_000, 90)
    ledger.buy("A", "2020-02-03", "2020-01-31", 10, 1_000, 90)
    ledger.buy("B", "2020-01-02", "2020-01-01", 10, 1_000, 90)
    age = ledger.days_since_buy("2020-02-10")
    assert age == {"A": 7, "B": 39}                                   # newest A lot is 7 days old


def test_full_run_does_not_stack_a_stock_bought_five_sessions_ago():
    # Scores always favor A, then B, then C. Entries come every 5 sessions and only one name is bought.
    prices = make_prices(A=100.0, B=100.0, C=100.0)
    rows = {d: {"A": 3.0, "B": 2.0, "C": 1.0} for d in CAL[:-1]}
    strat = TopNStrategy(top_n=1, cohort_fraction=0.1, entry_every=5, max_name_weight=1.0, rebuy_wait_days=30)
    res = run_backtest(prices, make_scores(rows), strat, CAL[1], CAL[22], FREE)
    buys = res.trades[res.trades.side == "BUY"].sort_values("date")
    # Entries at sessions 0, 5, 10, 15, 20: A, then B (A is new), then C, then nothing (all three are
    # younger than 30 calendar days), then nothing. No name is stacked.
    assert list(buys.symbol) == ["A", "B", "C"]
    assert all(audit(res, FREE).values())


def test_default_run_is_unchanged_by_the_new_field():
    # Same market as the first full-loop test. The new view field must not change old results.
    prices, scores, strat, entry, due = one_stock_setup()
    res = run_backtest(prices, scores, strat, CAL[1], CAL[-1], FREE)
    assert res.nav.cash.iloc[-1] == pytest.approx(500_000 + 5_000 * 120)
