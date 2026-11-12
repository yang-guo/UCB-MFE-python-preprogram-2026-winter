"""Sanity checks on the backtester.

First half: tiny made-up markets small enough to work out on paper, so the
expected numbers below come from arithmetic, not from running the code.
Second half: rules that the saved final run in results/ has to obey.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.resolve()))
from portfolio_sim import StaggeredBook  # noqa: E402

OUT_DIR = Path(__file__).parent.resolve() / "results"


def paper_market():
    """Two stocks, one tranche, $1,000, 10 bps fee, 5 day hold.

    Signal Jan 1, buy Jan 2 with $500 per stock. After the fee $499.5005 is
    actually invested, so A (at $10) gets 49.95005 shares and B (at $20)
    gets 24.975025. Both are due Jan 7. A sells that day at $12 for
    49.95005 * 12 * 0.999 = 598.80120. B has no Jan 7 price, so it waits
    and sells Jan 8 at $18 for 24.975025 * 18 * 0.999 = 449.10090.
    Cash at the end should be 1047.90210.
    """
    jan = pd.date_range("2026-01-01", periods=8, freq="D")
    closes = pd.DataFrame({"A": [9, 10, 10.5, 11, 11.5, 11, 12, 12.5],
                           "B": [19, 20, 21, 20, 19, 19, np.nan, 18]}, index=jan, dtype=float)
    book = StaggeredBook(closes, jan[0], jan[-1], start_cash=1000, tranches=1, basket=2,
                         fee_bps=10, horizon_days=5)
    run = book.run({jan[0]: pd.Series({"A": 2.0, "B": 1.0})})
    led, daily = run["fills"], run["daily"]
    entry = led[led.side == "buy"].set_index("symbol")
    exit_ = led[led.side == "sell"].set_index("symbol")

    shares_a, shares_b = 499.5005 / 10, 499.5005 / 20
    paid_a, paid_b = shares_a * 12 * 0.999, shares_b * 18 * 0.999
    return [
        ("buys happen the session after the signal", bool((entry.date == jan[1]).all())),
        ("shares = slice / (1 + fee) / price",
         np.isclose(entry.at["A", "shares"], shares_a, atol=1e-6)
         and np.isclose(entry.at["B", "shares"], shares_b, atol=1e-6)),
        ("A sells on Jan 7 for 598.80",
         exit_.at["A", "date"] == jan[6] and np.isclose(exit_.at["A", "cash_delta"], paid_a, atol=1e-4)),
        ("B has no Jan 7 price and sells Jan 8 for 449.10",
         exit_.at["B", "date"] == jan[7] and np.isclose(exit_.at["B", "cash_delta"], paid_b, atol=1e-4)),
        ("ending cash is 1047.90210", np.isclose(daily.cash.iloc[-1], 1047.90210, atol=1e-4)),
        ("Jan 3 NAV is shares x close with zero cash",
         np.isclose(daily.nav.loc[jan[2]], shares_a * 10.5 + shares_b * 21, atol=1e-6)),
        ("B is carried at $19 on Jan 7 while it has no price",
         np.isclose(daily.holdings.loc[jan[6]], shares_b * 19, atol=1e-6)),
    ]


def vanishing_stock():
    """One stock prints $10, 10, 11, 11 then goes silent. With a 3 day silence
    limit it should be closed on Mar 7 at $11, well before its 30 day date.
    $100 at $10 is 10 shares, so cash ends at $110 with no fee."""
    mar = pd.date_range("2026-03-01", periods=8, freq="D")
    closes = pd.DataFrame({"A": [10, 10, 11, 11] + [np.nan] * 4}, index=mar, dtype=float)
    book = StaggeredBook(closes, mar[0], mar[-1], start_cash=100, tranches=1, basket=1,
                         fee_bps=0, horizon_days=30, silent_limit=3)
    run = book.run({mar[0]: pd.Series({"A": 1.0})})
    closed = run["fills"][run["fills"].side == "buyout_close"]
    ok = len(closed) == 1 and closed.date.iloc[0] == mar[6] and np.isclose(run["daily"].cash.iloc[-1], 110)
    return [("a stock silent for 3 sessions is closed at its last print", ok)]


def audit_saved_run(folder):
    folder = Path(folder)
    daily = pd.read_csv(folder / "daily_nav.csv", index_col="date", parse_dates=True)
    led = pd.read_csv(folder / "fills.csv", parse_dates=["date"])
    lots = pd.read_csv(folder / "lots.csv", parse_dates=["picked_on", "bought_on", "sell_after", "sold_on"])

    replayed_cash = led.groupby("date").cash_delta.sum().reindex(daily.index).fillna(0).cumsum() + 1_000_000
    exits = led[led.side.isin(["sell", "buyout_close"])]
    matured = lots[lots.ticket.isin(led.ticket[led.side == "sell"])]
    buyouts = led[led.side == "buyout_close"]
    days_over = (matured.sold_on - matured.sell_after).dt.days

    return [
        ("cash is never below zero", bool(daily.cash.min() >= -1e-6)),
        ("NAV equals cash plus holdings on every day", np.allclose(daily.cash + daily.holdings, daily.nav)),
        ("cash replayed from the fills file matches daily cash", np.allclose(replayed_cash, daily.cash, atol=1e-4)),
        ("no lot is closed twice", not exits.ticket.duplicated().any()),
        ("no ordinary sale before the 90 day date", bool((matured.sold_on >= matured.sell_after).all())),
        (f"closed by the silence rule: {len(buyouts)} lots in {buyouts.symbol.nunique()} stocks", True),
        ("every sell date is exactly 90 days after purchase",
         bool(((lots.sell_after - lots.bought_on).dt.days == 90).all())),
        ("every purchase comes after its signal", bool((lots.bought_on > lots.picked_on).all())),
        ("nothing trades after Sep 30, 2026", led.date.max() <= pd.Timestamp("2026-09-30")),
        (f"ordinary sales more than 7 days late: {int((days_over > 7).sum())} of {len(matured)}", True),
    ]


def render(results, heading):
    return "\n".join([heading] + [f"  [{'PASS' if ok else 'FAIL'}] {what}" for what, ok in results])


if __name__ == "__main__":
    text = render(paper_market() + vanishing_stock(), "Paper market")
    if (OUT_DIR / "daily_nav.csv").exists():
        text += "\n\n" + render(audit_saved_run(OUT_DIR), "Saved final run")
    print(text)
    if OUT_DIR.exists():
        (OUT_DIR / "accounting_checks.txt").write_text(text + "\n", encoding="utf-8")
