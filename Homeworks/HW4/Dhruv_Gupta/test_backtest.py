"""Hand checks for the ledger. Prices and fees below are small enough to recompute."""

import numpy as np
import pandas as pd

from backtest import (
    FEE_BPS,
    HOLDING_DAYS,
    Ledger,
    allocate_cohort,
    check_accounting,
    run_backtest,
)


def _calendar(start, periods):
    return pd.bdate_range(start, periods=periods)


def test_fee_is_reserved_inside_the_budget():
    ledger = Ledger(cash=1000, fee_bps=FEE_BPS)
    price = 10.0
    ledger.buy(1000, price, "AAA", "2016-04-04", "2016-04-01")
    units = 1000 / (price * (1 + ledger.fee_rate))
    lot = next(iter(ledger.lots.values()))
    assert np.isclose(lot.units, units)
    assert np.isclose(ledger.cash, 0.0)
    assert np.isclose(lot.entry_fee, units * price * ledger.fee_rate)
    assert lot.due_date == pd.Timestamp("2016-04-04") + pd.Timedelta(days=HOLDING_DAYS)


def test_sale_returns_cash_net_of_the_exit_fee():
    ledger = Ledger(cash=1000, fee_bps=0)
    ledger.buy(1000, 10, "AAA", "2016-04-04", "2016-04-01")
    ledger.sell(1, 12, "2016-07-05")
    assert np.isclose(ledger.cash, 1200)
    assert ledger.lots == {}
    pnl = 100 * (12 - 10)
    assert np.isclose(ledger.fills[-1]["cash_change"], 1200)
    assert np.isclose(pnl, 200)


def test_two_budgets_cannot_spend_the_same_cash():
    scores = pd.Series({"AAA": 0.2, "BBB": 0.1})
    budgets = allocate_cohort(scores, cash=1000, nav=1000, held_symbols=set(),
                              top_n=2, cohort_fraction=1, name_cap=1, fee_rate=0)
    assert np.isclose(budgets.sum(), 1000)
    try:
        allocate_cohort(scores, cash=1000, nav=5000, held_symbols=set(),
                        top_n=2, cohort_fraction=1, name_cap=1, fee_rate=0)
    except Exception:
        raise AssertionError("A cohort larger than cash must be cut down to cash, not rejected later")
    capped = allocate_cohort(scores, cash=1000, nav=5000, held_symbols=set(),
                             top_n=2, cohort_fraction=1, name_cap=1, fee_rate=0)
    assert np.isclose(capped.sum(), 1000)


def test_sale_proceeds_are_not_spent_by_the_same_close():
    days = pd.DatetimeIndex(["2016-03-31", "2016-04-01", "2016-06-30"])
    quotes = pd.DataFrame(
        {"AAA": [10.0, 10.0, 10.0], "BBB": [8.0, 8.0, 8.0]},
        index=days,
    )
    # April 1 buys AAA with the whole book. June 30 is both AAA's maturity and
    # a new entry date. The June 29 close still has the cash tied up in AAA,
    # so the June 30 sale cannot fund a BBB purchase.
    quotes.loc[pd.Timestamp("2016-06-29")] = {"AAA": 10.0, "BBB": 8.0}
    days = pd.DatetimeIndex(["2016-03-31", "2016-04-01", "2016-06-29", "2016-06-30"])
    quotes = quotes.reindex(days)
    scores = {
        pd.Timestamp("2016-03-31"): pd.Series({"AAA": 1.0, "BBB": 0.0}),
        pd.Timestamp("2016-06-29"): pd.Series({"AAA": 0.0, "BBB": 1.0}),
    }
    history, fills, open_lots, _ = run_backtest(
        quotes, days, scores, entry_dates=["2016-04-01", "2016-06-30"],
        start="2016-04-01", end="2016-06-30", initial_cash=1000, fee_bps=0,
        cohort_fraction=1, name_cap=1, top_n=1,
    )
    assert list(fills["symbol"]) == ["AAA", "AAA"]
    assert list(fills["side"]) == ["buy", "sell"]
    assert np.isclose(history.loc[pd.Timestamp("2016-06-30"), "cash"], 1000)
    assert len(open_lots) == 0


def test_missing_entry_quote_is_not_filled():
    days = pd.bdate_range("2016-03-31", periods=3)
    quotes = pd.DataFrame({"AAA": [10.0, np.nan, 10.0]}, index=days)
    scores = {days[0]: pd.Series({"AAA": 1.0})}
    history, fills, open_lots, diagnostics = run_backtest(
        quotes, days, scores, entry_dates=[days[1]],
        start=days[1], end=days[-1], initial_cash=1000, fee_bps=0,
    )
    assert fills.empty
    assert len(open_lots) == 0
    assert np.isclose(history["cash"].iloc[-1], 1000)
    assert (diagnostics["event"] == "rejected_missing_quote").any()


def test_weekend_maturity_sells_on_the_next_session():
    entry = pd.Timestamp("2016-04-01")  # Friday
    signal = pd.Timestamp("2016-03-31")
    due = entry + pd.Timedelta(days=90)  # June 30, 2016, Thursday
    # Force the due date off the calendar and put the next session the day after.
    sessions = pd.DatetimeIndex([signal, entry, due + pd.Timedelta(days=1)])
    quotes = pd.DataFrame({"AAA": [10.0, 10.0, 13.0]}, index=sessions)
    history, fills, open_lots, _ = run_backtest(
        quotes, sessions, {signal: pd.Series({"AAA": 1.0})}, entry_dates=[entry],
        start=entry, end=sessions[-1], initial_cash=1000, fee_bps=0,
        cohort_fraction=1, name_cap=1,
    )
    sale = fills.loc[fills["side"].eq("sell")].iloc[0]
    assert sale["date"] == sessions[-1]
    assert np.isclose(history["nav"].iloc[-1], 1300)
    assert len(open_lots) == 0


def test_open_positions_are_marked_at_the_sample_end():
    days = pd.bdate_range("2016-03-31", periods=4)
    quotes = pd.DataFrame({"AAA": [10.0, 10.0, 11.0, 12.0]}, index=days)
    # A later price exists in this frame only if we fail to cut the sample.
    history, fills, open_lots, _ = run_backtest(
        quotes.iloc[:3], days[:3], {days[0]: pd.Series({"AAA": 1.0})},
        entry_dates=[days[1]],         start=days[1], end=days[2], initial_cash=1000, fee_bps=0,
        cohort_fraction=1, name_cap=1,
    )
    assert len(fills) == 1 and fills.iloc[0]["side"] == "buy"
    assert len(open_lots) == 1
    assert np.isclose(history["nav"].iloc[-1], 1100)
    assert history.index.max() == days[2]


def test_name_cap_leaves_the_unused_budget_in_cash():
    scores = pd.Series({"AAA": 2.0, "BBB": 1.0})
    budgets = allocate_cohort(
        scores, cash=900, nav=900, held_symbols={"CCC"},
        top_n=2, cohort_fraction=1 / 3, name_cap=0.10, fee_rate=0.0005,
    )
    assert "CCC" not in budgets.index
    max_spend = 0.10 * 900 * 1.0005
    assert np.all(budgets <= max_spend + 1e-8)
    assert budgets.sum() < 900 / 3
    assert np.isclose(budgets["AAA"], max_spend)


def test_full_path_accounting_identity():
    days = pd.bdate_range("2016-03-31", periods=70)
    entry = days[1]
    quotes = pd.DataFrame({
        "AAA": 10 + 0.01 * np.arange(len(days)),
        "BBB": 20 + 0.02 * np.arange(len(days)),
    }, index=days)
    scores = {days[0]: pd.Series({"AAA": 0.3, "BBB": 0.2})}
    history, fills, open_lots, _ = run_backtest(
        quotes, days, scores, entry_dates=[entry],
        start=entry, end=days[-1], initial_cash=1000, fee_bps=FEE_BPS,
        cohort_fraction=1, name_cap=1,
    )
    check_accounting(history, fills, open_lots, 1000)
    assert history["open_lots"].iloc[-1] == 0
    assert fills["side"].tolist() == ["buy", "buy", "sell", "sell"]


def main():
    tests = [
        test_fee_is_reserved_inside_the_budget,
        test_sale_returns_cash_net_of_the_exit_fee,
        test_two_budgets_cannot_spend_the_same_cash,
        test_sale_proceeds_are_not_spent_by_the_same_close,
        test_missing_entry_quote_is_not_filled,
        test_weekend_maturity_sells_on_the_next_session,
        test_open_positions_are_marked_at_the_sample_end,
        test_name_cap_leaves_the_unused_budget_in_cash,
        test_full_path_accounting_identity,
    ]
    for test in tests:
        test()
        print(f"passed {test.__name__}")
    print(f"passed {len(tests)} accounting checks")


if __name__ == "__main__":
    main()
