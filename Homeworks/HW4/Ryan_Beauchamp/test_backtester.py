"""Hand-checkable tests for backtester.py. Each docstring states the expected
number, worked out by hand before running.

Synthetic market: weekday sessions in 2015 (pd.bdate_range), fee 10 bps per
side, $1,000,000 start, 90-calendar-day holds, max 5 stale sessions.

Run from the repository root (no pytest needed):
    .venv/Scripts/python -B Homeworks/HW4/Ryan_Beauchamp/test_backtester.py

Credits: written with AI assistance (Claude Code) and reviewed by the author;
the constant-price round trip (-$199.80) was also checked by hand by the author.
"""

import math
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from backtester import (Account, Backtester, FixedOrdersStrategy, InsufficientCashError,  # noqa: E402
                        ScoreStrategy, StaleQuoteError, TrailingReturnStrategy, accounting_checks,
                        benchmark_nav, performance)

FEE = 0.001
DATES = pd.bdate_range("2015-01-02", "2015-06-30")
T = pd.Timestamp


def flat_prices(dates=DATES, symbols=("A",), price=100.0):
    return pd.DataFrame(price, index=dates, columns=list(symbols), dtype=float)


def run(prices, orders, **kwargs):
    kwargs.setdefault("entry_every", 1)
    return Backtester(prices, FixedOrdersStrategy(orders), fee_bps=10, **kwargs).run()


def close(a, b, tol=1e-6):
    assert abs(a - b) <= tol, f"expected {b!r}, got {a!r}"


# --- tests ---------------------------------------------------------------------
def test_constant_price_round_trip():
    """Budget 100,000 at constant $100: net = 100,000 * 0.999/1.001 - 100,000 = -199.8002 (final NAV 999,800.1998)."""
    r = run(flat_prices(), {"2015-01-02": {"A": 100_000}})
    expected = 100_000 * (1 - FEE) / (1 + FEE) - 100_000
    close(expected, -199.80019980, 1e-6)
    close(r.closed_lots.pnl.iloc[0], expected)
    close(r.history.nav.iloc[-1], 1_000_000 + expected)
    assert all(accounting_checks(r).values())


def test_monday_entry_sunday_maturity_sells_monday():
    """Signal Fri 2015-01-02 -> fill Mon 2015-01-05; due Sun 2015-04-05 -> sell Mon 2015-04-06."""
    r = run(flat_prices(), {"2015-01-02": {"A": 100_000}})
    buy, sell = r.fills.iloc[0], r.fills.iloc[1]
    assert buy.date == T("2015-01-05") and buy.date.day_name() == "Monday"
    assert buy.due_date == T("2015-04-05") and buy.due_date.day_name() == "Sunday"
    assert sell.date == T("2015-04-06") and sell.date.day_name() == "Monday"


def test_two_lots_same_ticker_different_maturities():
    """Lots of 10,000 (fill 01-05) and 20,000 (fill 01-19) in A sell on 04-06 and 04-20.
    Between them one lot of 20,000/100.1 = 199.8002 units stays open.
    Final cash = 1,000,000 - 30,000 * 0.002/1.001 = 999,940.0599."""
    r = run(flat_prices(), {"2015-01-02": {"A": 10_000}, "2015-01-16": {"A": 20_000}})
    sells = r.fills.loc[r.fills.side.eq("sell")].set_index("lot_id")
    assert sells.loc[0, "date"] == T("2015-04-06") and sells.loc[1, "date"] == T("2015-04-20")
    assert r.history.loc["2015-04-10", "open_lots"] == 1
    close(r.history.loc["2015-04-10", "holdings"], 20_000 / 100.1 * 100)
    close(sells.loc[1, "units"], 199.80019980)
    close(r.history.cash.iloc[-1], 1_000_000 - 30_000 * 0.002 / 1.001)
    close(r.history.cash.iloc[-1], 999_940.05994006)


def test_missing_entry_quote_rejects_and_keeps_cash():
    """B has no quote on fill day 2015-01-05: order rejected, cash and NAV stay 1,000,000 every day."""
    prices = flat_prices(symbols=("A", "B"))
    prices.loc["2015-01-05", "B"] = np.nan
    r = run(prices, {"2015-01-02": {"B": 50_000}})
    assert r.orders.status.iloc[0] == "rejected: no entry quote"
    assert len(r.fills) == 0
    assert (r.history.cash == 1_000_000).all() and (r.history.nav == 1_000_000).all()


def test_missing_quote_on_due_date_sells_next_quote():
    """A has no quote on due session Mon 2015-04-06: sale moves to Tue 2015-04-07; one stale row, past due."""
    prices = flat_prices()
    prices.loc["2015-04-06", "A"] = np.nan
    r = run(prices, {"2015-01-02": {"A": 100_000}})
    sell = r.fills.loc[r.fills.side.eq("sell")].iloc[0]
    assert sell.date == T("2015-04-07")
    assert len(r.stale) == 1 and r.stale.iloc[0].date == T("2015-04-06") and bool(r.stale.iloc[0].past_due)
    close(r.closed_lots.pnl.iloc[0], -199.80019980)


def test_insufficient_cash_raises():
    """Orders of 1,500,000 against 1,000,000 cash raise InsufficientCashError; so does Account.buy(2,000,000)."""
    try:
        run(flat_prices(), {"2015-01-02": {"A": 1_500_000}})
        raise AssertionError("no error raised for orders above cash")
    except InsufficientCashError:
        pass
    try:
        Account(1_000_000, 10).buy("A", 2_000_000, 100.0, T("2015-01-05"), T("2015-01-02"))
        raise AssertionError("no error raised by Account.buy")
    except InsufficientCashError:
        pass


def test_no_orders_nav_flat():
    """No orders: NAV = 1,000,000 on all 128 sessions (weekdays 2015-01-02 to 06-30: 21+20+22+22+21+22)."""
    r = run(flat_prices(), {})
    assert len(r.history) == len(DATES) == 128
    assert (r.history.nav == 1_000_000).all()


def test_lot_open_at_end_kept_at_last_mark():
    """Window ends 2015-03-31 at price 110 (100 before). Lot of 100,000 bought 01-05 at 100 is still open:
    units = 100,000/100.1 = 999.000999; final NAV = 900,000 + 999.000999*110 = 1,009,890.1099."""
    dates = pd.bdate_range("2015-01-02", "2015-03-31")
    prices = flat_prices(dates)
    prices.iloc[-1] = 110.0
    r = run(prices, {"2015-01-02": {"A": 100_000}})
    assert len(r.open_lots) == 1 and (r.fills.side == "sell").sum() == 0
    close(r.history.nav.iloc[-1], 1_009_890.10989011)
    assert all(accounting_checks(r).values())


def test_price_gap_longer_than_limit_raises():
    """Held A missing 6 sessions (limit 5) raises StaleQuoteError; missing 5 sessions runs with 5 stale rows."""
    six = flat_prices()
    six.loc["2015-02-02":"2015-02-09", "A"] = np.nan          # Mon..Mon = 6 sessions
    assert six.A.isna().sum() == 6
    try:
        run(six, {"2015-01-02": {"A": 100_000}})
        raise AssertionError("no StaleQuoteError for a 6-session gap")
    except StaleQuoteError:
        pass
    five = flat_prices()
    five.loc["2015-02-02":"2015-02-06", "A"] = np.nan
    r = run(five, {"2015-01-02": {"A": 100_000}})
    assert len(r.stale) == 5 and r.stale.stale_sessions.max() == 5


def random_market(seed=7, n_symbols=20, dates=DATES):
    rng = np.random.default_rng(seed)
    steps = rng.normal(0.0003, 0.02, size=(len(dates), n_symbols))
    prices = pd.DataFrame(50 * np.exp(np.cumsum(steps, axis=0)), index=dates,
                          columns=[f"S{i:02d}" for i in range(n_symbols)])
    # Isolated one-session gaps (never two in a row), so the stale limit is not hit.
    holes = rng.random(prices.shape) < 0.03
    holes[1:] &= ~holes[:-1]
    prices = prices.mask(holes)
    return prices


def momentum_run(prices, strategy=None):
    strategy = strategy or TrailingReturnStrategy(lookback=21, top_n=5, cohort_fraction=1 / 3, per_name_cap=0.1)
    return Backtester(prices, strategy, fee_bps=10, entry_every=5).run()


def test_cash_rebuilt_and_nav_identity_on_random_market():
    """Random 20-stock market with gaps: every accounting identity holds (all checks True)."""
    r = momentum_run(random_market())
    checks = accounting_checks(r)
    assert all(checks.values()), checks
    assert (r.fills.side == "buy").sum() > 20 and (r.fills.side == "sell").sum() > 5


def test_timing_signal_before_fill_and_sale_after_due():
    """Every buy fills exactly one session after its signal; every sale is on/after its due date."""
    r = momentum_run(random_market())
    buys = r.fills.loc[r.fills.side.eq("buy")]
    position = {d: i for i, d in enumerate(DATES)}
    assert all(position[f] - position[s] == 1 for s, f in zip(buys.signal_date, buys.date))
    sells = r.fills.loc[r.fills.side.eq("sell")]
    assert (sells.date >= sells.due_date).all()


def test_future_prices_do_not_change_past_decisions():
    """Scaling all prices after 2015-03-16 by random factors leaves every order planned on/before 03-16 unchanged."""
    prices = random_market()
    cutoff = T("2015-03-16")
    changed = prices.copy()
    later = changed.index > cutoff
    changed.loc[later] *= np.random.default_rng(1).uniform(0.5, 2.0, size=changed.loc[later].shape)
    a, b = momentum_run(prices).orders, momentum_run(changed).orders
    a, b = a.loc[a.signal_date <= cutoff].reset_index(drop=True), b.loc[b.signal_date <= cutoff].reset_index(drop=True)
    assert len(a) > 0
    pd.testing.assert_frame_equal(a[["signal_date", "symbol", "budget"]], b[["signal_date", "symbol", "budget"]])
    assert not momentum_run(prices).orders.equals(momentum_run(changed).orders)  # later decisions do change


class VandalStrategy(TrailingReturnStrategy):
    """Ranks honestly, then overwrites the price window it was given."""

    def rank(self, snapshot, window):
        scores = super().rank(snapshot, window)
        if window is not None:
            window.iloc[:, :] = 1e9
        return scores


def test_strategy_editing_its_window_cannot_change_quotes():
    """A strategy overwriting its window with 1e9 gives a run identical to the honest one; quotes unchanged."""
    prices = random_market()
    bt = Backtester(prices, VandalStrategy(lookback=21, top_n=5, cohort_fraction=1 / 3, per_name_cap=0.1),
                    fee_bps=10, entry_every=5)
    before = bt._values.copy()
    vandal = bt.run()
    assert np.array_equal(before, bt._values, equal_nan=True)
    honest = momentum_run(prices)
    pd.testing.assert_frame_equal(vandal.history, honest.history)
    pd.testing.assert_frame_equal(vandal.fills, honest.fills)


def test_metrics_on_tiny_series():
    """NAV 100, 110, 99, 121 on 2015-01-01, 04-01, 08-01, 2016-01-01 (365 days):
    final 121; CAGR = 1.21^(365.25/365) - 1 = 21.016%; max drawdown = 99/110 - 1 = -10%;
    daily returns +10%, -10%, +22.22%; rf/day = 1.02^(1/252) - 1;
    Sharpe = mean(excess)/std(excess, ddof=1)*sqrt(252) = 7.2211."""
    nav = pd.Series([100.0, 110.0, 99.0, 121.0],
                    index=pd.to_datetime(["2015-01-01", "2015-04-01", "2015-08-01", "2016-01-01"]))
    m = performance(nav)
    close(m.final_value, 121.0)
    close(m.CAGR, 1.21 ** (365.25 / 365) - 1, 1e-12)
    close(m.CAGR, 0.21016, 1e-5)
    close(m.max_drawdown, -0.10, 1e-12)
    returns = [0.10, -0.10, 121 / 99 - 1]
    rf = 1.02 ** (1 / 252) - 1
    excess = [x - rf for x in returns]
    mean = sum(excess) / 3
    sd = math.sqrt(sum((x - mean) ** 2 for x in excess) / 2)
    close(m.sharpe, mean / sd * math.sqrt(252), 1e-12)
    close(m.sharpe, 7.2211, 1e-3)


def test_benchmark_invests_at_first_fill_close():
    """Benchmark levels 50, 100, 110: $1M held as cash on day 0, invested at day 1 close -> 1.0M, 1.0M, 1.1M."""
    dates = pd.bdate_range("2015-01-02", periods=3)
    nav = benchmark_nav(pd.Series([50.0, 100.0, 110.0], index=dates), dates, dates[1])
    assert np.allclose(nav.to_numpy(), [1_000_000, 1_000_000, 1_100_000])


def test_per_name_cap_limits_total_position():
    """Top 1 name A, cohort 4% of NAV, cap 5%. Day 1 order 40,000. After it fills at $100, A is worth
    40,000/100.1*100 = 39,960.04 and NAV = 999,960.04, so the next order is capped at
    0.05*999,960.04 - 39,960.04 = 10,037.96 instead of 4% of NAV (39,998.40)."""
    scores = {T("2015-01-02"): pd.Series({"A": 1.0}), T("2015-01-05"): pd.Series({"A": 1.0})}
    strategy = ScoreStrategy(scores, top_n=1, cohort_fraction=0.04, per_name_cap=0.05)
    r = Backtester(flat_prices(), strategy, fee_bps=10, entry_every=1).run()
    budgets = r.orders.set_index("signal_date").budget
    close(budgets[T("2015-01-02")], 40_000.0)
    close(budgets[T("2015-01-05")], 0.05 * (1_000_000 - 40_000 + 40_000 / 100.1 * 100) - 40_000 / 100.1 * 100)
    close(budgets[T("2015-01-05")], 10_037.96204, 1e-4)


def test_long_gap_hold_policy_keeps_last_mark():
    """Policy "hold": A missing 6 sessions (limit 5) from 02-02 to 02-09 does not stop the run; one long-gap row
    (last quote 2015-01-30, flagged 02-09); quotes resume and the lot sells as usual on 2015-04-06, pnl -199.80."""
    six = flat_prices()
    six.loc["2015-02-02":"2015-02-09", "A"] = np.nan
    r = run(six, {"2015-01-02": {"A": 100_000}}, long_gap_policy="hold")
    assert len(r.long_gaps) == 1 and r.long_gaps.iloc[0].last_quote_date == T("2015-01-30")
    assert r.long_gaps.iloc[0].flagged_on == T("2015-02-09") and not r.long_gaps.iloc[0].still_open_at_end
    close(r.closed_lots.pnl.iloc[0], -199.80019980)
    assert r.fills.loc[r.fills.side.eq("sell"), "date"].iloc[0] == T("2015-04-06")


def test_long_gap_hold_series_ends_lot_stays_at_last_mark():
    """Policy "hold": A stops quoting after 2015-02-27 at $120. The lot is never sold (no invented fill) and stays in NAV:
    units 999.000999 * 120 = 119,880.12; final NAV = 900,000 + 119,880.12 = 1,019,880.12; value_at_end 119,880.12."""
    prices = flat_prices()
    prices.loc["2015-02-27", "A"] = 120.0
    prices.loc["2015-03-02":, "A"] = np.nan
    r = run(prices, {"2015-01-02": {"A": 100_000}}, long_gap_policy="hold")
    assert (r.fills.side == "sell").sum() == 0 and len(r.open_lots) == 1
    close(r.history.nav.iloc[-1], 1_019_880.11988012)
    close(r.long_gaps.value_at_end.sum(), 119_880.11988012)
    assert all(accounting_checks(r).values())


def test_jump_flag_records_but_does_not_alter():
    """Held A jumps 100 -> 600 on 2015-02-02 (log move 1.79 > ln 5 = 1.61): one jump row, NAV uses 600."""
    prices = flat_prices()
    prices.loc["2015-02-02":, "A"] = 600.0
    r = run(prices, {"2015-01-02": {"A": 100_000}})
    assert len(r.jumps) == 1 and r.jumps.iloc[0].date == T("2015-02-02")
    close(r.history.loc["2015-02-02", "holdings"], 100_000 / 100.1 * 600)


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in globals().items() if name.startswith("test_") and callable(fn)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception:  # noqa: BLE001 - report every failure, keep going
            failed += 1
            print(f"FAIL  {name}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} tests passed")
    sys.exit(1 if failed else 0)
