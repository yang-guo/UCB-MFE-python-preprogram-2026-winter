"""Hand-checked accounting tests for backtester.py.

Each test uses tiny made-up prices where every number can be worked out on
paper. The "by hand" arithmetic is written next to the value the code produced,
and the comparison is saved to accounting_evidence.md as homework evidence.

Run from Homeworks/HW4/Ananya_Das/:
    ../../../.venv/bin/python -m unittest -v tests.test_backtester
"""

import math
from pathlib import Path
import sys
import unittest

import pandas as pd

HW_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HW_DIR))

import backtester as bt  # noqa: E402

EVIDENCE_FILE = HW_DIR / "accounting_evidence.md"
SECTIONS = []  # Filled by the tests, written by tearDownModule.

# Settings used by every test: $1,000,000 start, 10 bps per side, 90 days.
CAPITAL, FEE_BPS = 1_000_000, 10
D = pd.Timestamp


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def calendar(holidays=()):
    """Weekdays from Dec 31, 2009 to Apr 30, 2010, minus Jan 1 and any extra holidays."""
    days = pd.bdate_range("2009-12-31", "2010-04-30")
    return days.difference(pd.to_datetime(["2010-01-01", *holidays]))


def price_path(symbol, cal, base=100.0, changes=None, missing=()):
    """Price `base` every session, except `changes` {from_date: new_price} and `missing` dates."""
    changes = {D(k): v for k, v in (changes or {}).items()}
    missing = set(pd.to_datetime(list(missing)))
    rows, price = [], base
    for day in cal:
        price = changes.get(day, price)
        if day not in missing:
            rows.append({"date": day, "symbol": symbol, "adj_close": price})
    return pd.DataFrame(rows)


class FixedOrders:
    """A test strategy: on a given signal date, buy fixed dollar budgets."""

    def __init__(self, orders_by_signal_date):
        self.orders = {D(k): v for k, v in orders_by_signal_date.items()}

    def generate_orders(self, signal_date, execution_date, snapshot):
        return pd.Series(self.orders.get(signal_date, {}), dtype=float)


def fmt(value):
    if isinstance(value, pd.Timestamp):
        return f"{value.date()} ({value.day_name()[:3]})"
    if isinstance(value, bool) or isinstance(value, str):
        return str(value)
    if isinstance(value, (int, float)):
        if abs(value - round(value)) < 1e-6:
            return f"{round(value):,}"
        return f"{value:,.4f}"
    return str(value)


class Section:
    """Collects hand-vs-code rows for one test and asserts each one."""

    def __init__(self, test, title, setup):
        self.test, self.rows = test, []
        SECTIONS.append((title, setup, self.rows))

    def check(self, item, by_hand, expected, actual):
        if isinstance(expected, (int, float)) and not isinstance(expected, bool):
            ok = math.isclose(expected, actual, rel_tol=0, abs_tol=1e-6)
        else:
            ok = expected == actual
        self.rows.append((item, by_hand, fmt(expected), fmt(actual), "✅" if ok else "❌"))
        self.test.assertTrue(ok, f"{item}: expected {expected!r}, code gave {actual!r}")


def sells(result):
    return result.fills[result.fills.side.eq("SELL")]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestAccounting(unittest.TestCase):

    def test_1_buy_and_sell_with_costs(self):
        s = Section(self, "Test 1: one buy and one sale, with 10 bps costs",
                    "Signal Mon 2010-01-04, buy AAA at the Tue 2010-01-05 close for a $100,100 "
                    "budget (fee included) at $100. The price stays $100 until it jumps to $110 "
                    "on Mon 2010-04-05, which is exactly 90 days later.")
        cal = calendar()
        prices = price_path("AAA", cal, changes={"2010-04-05": 110.0})
        r = bt.Backtester(prices, cal, CAPITAL, FEE_BPS).run(
            FixedOrders({"2010-01-04": {"AAA": 100_100}}), "2010-01-04", "2010-04-09")
        buy, sell = r.fills.iloc[0], sells(r).iloc[0]
        h = r.history

        s.check("Starting NAV (Dec 31, before trading)", "initial capital", 1_000_000, h.nav.iloc[0])
        s.check("Units bought", "100,100 / (100 × 1.001) = 1,000", 1_000, buy.units)
        s.check("Buy fee", "1,000 × 100 × 0.001 = 100", 100, buy.fee)
        s.check("Cash after buy", "1,000,000 − 100,000 − 100 = 899,900", 899_900, h.cash.loc["2010-01-05"])
        s.check("NAV at entry close", "899,900 + 1,000 × 100 = 999,900", 999_900, h.nav.loc["2010-01-05"])
        s.check("Maturity date", "2010-01-05 + 90 days = 2010-04-05", D("2010-04-05"), buy.due_date)
        s.check("NAV on Fri 2010-04-02 (still held)", "899,900 + 1,000 × 100 = 999,900",
                999_900, h.nav.loc["2010-04-02"])
        s.check("Sale date", "maturity is a trading day", D("2010-04-05"), sell.date)
        s.check("Sale fee", "1,000 × 110 × 0.001 = 110", 110, sell.fee)
        s.check("Sale proceeds", "110,000 − 110 = 109,890", 109_890, sell.cash_change)
        s.check("Cash after sale", "899,900 + 109,890 = 1,009,790", 1_009_790, h.cash.loc["2010-04-05"])
        s.check("Realized PnL", "1,000 × (110 − 100) − 100 − 110 = 9,790", 9_790, sell.realized_pnl)
        s.check("Total fees", "100 + 110 = 210", 210, r.summary.total_fees)
        s.check("Final NAV (all cash)", "1,000,000 + 9,790 = 1,009,790", 1_009_790, h.nav.iloc[-1])
        s.check("Open lots at end", "sold", 0, int(h.open_lots.iloc[-1]))

    def test_2_maturity_on_closed_day(self):
        s = Section(self, "Test 2: 90-day maturity lands on a day the market is closed",
                    "Lot A: buy Mon 2010-01-04, so it is due Sun 2010-04-04 (a weekend). "
                    "Lot B: buy Thu 2010-01-07, so it is due Wed 2010-04-07, which this test "
                    "calendar marks as a market holiday. Both budgets are $100,100 at a constant "
                    "$100, so each lot is 1,000 units.")
        cal = calendar(holidays=["2010-04-07"])
        prices = price_path("A", cal)._append(price_path("B", cal), ignore_index=True)
        r = bt.Backtester(prices, cal, CAPITAL, FEE_BPS).run(
            FixedOrders({"2009-12-31": {"A": 100_100}, "2010-01-06": {"B": 100_100}}),
            "2010-01-04", "2010-04-30")
        sold = sells(r).set_index("symbol")
        h = r.history

        s.check("Lot A due date", "2010-01-04 + 90 days", D("2010-04-04"), sold.loc["A", "due_date"])
        s.check("Lot A sold", "Sunday is closed, so the next session is Monday",
                D("2010-04-05"), sold.loc["A", "date"])
        s.check("Lot A calendar days held", "Jan 4 to Apr 5 = 91", 91,
                (sold.loc["A", "date"] - D("2010-01-04")).days)
        s.check("Lot B due date", "2010-01-07 + 90 days", D("2010-04-07"), sold.loc["B", "due_date"])
        s.check("Lot B sold", "Apr 7 is a holiday, so the next session is Thu Apr 8",
                D("2010-04-08"), sold.loc["B", "date"])
        s.check("Lot B calendar days held", "Jan 7 to Apr 8 = 91", 91,
                (sold.loc["B", "date"] - D("2010-01-07")).days)
        s.check("Not sold early: open lots on Fri 2010-04-02", "both lots before maturity", 2,
                int(h.open_lots.loc["2010-04-02"]))
        s.check("Open lots on Tue 2010-04-06", "A sold, B not yet due", 1, int(h.open_lots.loc["2010-04-06"]))
        s.check("Weekend/holiday exit counted as on time", "first session on/after due date",
                False, bool(sold.exit_delayed.any()))
        s.check("Final cash", "1,000,000 − 2 × 100,100 + 2 × (100,000 − 100) = 999,600",
                999_600, h.cash.iloc[-1])
        s.check("Total fees", "4 trades × 100 = 400", 400, r.summary.total_fees)

    def test_3_buy_exceeding_cash(self):
        s = Section(self, "Test 3: a buy that needs more cash than is available",
                    "(a) A $10,000 account buys $6,006, then tries $6,006 again. "
                    "(b) Over a full run, a strategy asks for $600,000 + $600,000 on one day. "
                    "(c) A strategy spends $600,000 on day 1, then asks for $600,000 more on day 2 "
                    "while only $400,000 is left.")
        # (a) Directly on the account.
        pf = bt.Portfolio(10_000, FEE_BPS)
        pf.buy("AAA", 6_006, 100.0, "2010-01-05", "2010-01-04")
        s.check("(a) Cash after first buy", "10,000 − 60 × 100 − 6 = 3,994", 3_994, pf.cash)
        with self.assertRaises(bt.InsufficientCashError):
            pf.buy("BBB", 6_006, 100.0, "2010-01-05", "2010-01-04")
        s.check("(a) Second $6,006 buy refused, cash unchanged", "3,994 < 6,006, so refused", 3_994, pf.cash)
        s.check("(a) Lots after refused buy", "only the first", 1, len(pf.lots))
        pf.buy("CCC", 3_994, 100.0, "2010-01-05", "2010-01-04")
        s.check("(a) Buying exactly the remaining cash works", "3,994 / 100.1 = 39.9001 units",
                3_994 / 100.1, pf.lots[1].units)
        s.check("(a) Cash after spending it all", "3,994 − 3,994 = 0", 0, pf.cash)

        cal = calendar()
        prices = price_path("AAA", cal)._append(price_path("BBB", cal), ignore_index=True)
        # (b) Two orders on one day that add up to more than the account holds.
        b = bt.Backtester(prices, cal, CAPITAL, FEE_BPS)
        with self.assertRaises(bt.OrderError):
            b.run(FixedOrders({"2010-01-04": {"AAA": 600_000, "BBB": 600_000}}), "2010-01-04", "2010-04-30")
        s.check("(b) $1.2M of orders against $1M: run stops", "1,200,000 > 1,000,000", 0, len(b.portfolio.fills))
        s.check("(b) Cash untouched", "no fills", 1_000_000, b.portfolio.cash)

        # (c) The same cash cannot be spent twice on consecutive days.
        c = bt.Backtester(prices, cal, CAPITAL, FEE_BPS)
        with self.assertRaises(bt.OrderError):
            c.run(FixedOrders({"2010-01-04": {"AAA": 600_000}, "2010-01-05": {"BBB": 600_000}}),
                  "2010-01-04", "2010-04-30")
        s.check("(c) Day-1 buy filled", "one fill", 1, len(c.portfolio.fills))
        s.check("(c) Cash left after day 1", "1,000,000 − 600,000 = 400,000", 400_000, c.portfolio.cash)

    def test_4_stock_stops_quoting(self):
        s = Section(self, "Test 4: a held stock stops quoting (max 5 stale sessions)",
                    "Each case buys 1,000 units at the Tue 2010-01-05 close ($100,100 budget at "
                    "$100), so cash is 899,900 afterwards. Maturity is Mon 2010-04-05. "
                    "AAA: last quote Mon 2010-02-01, then none ever again (a delisting with "
                    "unknown proceeds). BBB: no quote on Apr 5 or Apr 6, then $105 on Apr 7. "
                    "CCC: no quotes Feb 2–10 (7 sessions), then quotes resume at $90.")
        cal = calendar()
        orders = lambda sym: FixedOrders({"2010-01-04": {sym: 100_100}})  # noqa: E731
        after_feb1 = cal[cal > D("2010-02-01")]

        # AAA: disappears for good.
        r = bt.Backtester(price_path("AAA", cal, missing=after_feb1), cal, CAPITAL, FEE_BPS).run(
            orders("AAA"), "2010-01-04", "2010-04-30")
        d, h = r.diagnostics, r.history
        s.check("AAA stale-mark sessions", "Feb 2, 3, 4, 5, 8 = 5", 5, int(d.event.eq("stale_mark").sum()))
        s.check("AAA frozen on", "6th session without a quote: Tue Feb 9", D("2010-02-09"),
                d.loc[d.event.eq("frozen"), "date"].iloc[0])
        s.check("AAA never sold (no quote to sell at)", "no fill is invented", 0, len(sells(r)))
        s.check("AAA still open at the end", "lot carried", 1, int(h.open_lots.iloc[-1]))
        s.check("Final NAV, frozen at last price", "899,900 + 1,000 × 100 = 999,900", 999_900, h.nav.iloc[-1])
        s.check("Final frozen value", "1,000 × 100", 100_000, h.frozen_value.iloc[-1])
        s.check("Final NAV if frozen lot is worth $0", "999,900 − 100,000 = 899,900",
                899_900, h.nav_frozen_at_zero.iloc[-1])

        # BBB: missing exactly on its maturity date.
        bbb = price_path("BBB", cal, changes={"2010-04-07": 105.0}, missing=["2010-04-05", "2010-04-06"])
        r = bt.Backtester(bbb, cal, CAPITAL, FEE_BPS).run(orders("BBB"), "2010-01-04", "2010-04-30")
        sale = sells(r).iloc[0]
        s.check("BBB exit delay logged on", "due Mon Apr 5, no quote", D("2010-04-05"),
                r.diagnostics.loc[r.diagnostics.event.eq("exit_delayed"), "date"].iloc[0])
        s.check("BBB sold on", "first real quote after maturity: Wed Apr 7", D("2010-04-07"), sale.date)
        s.check("BBB sale flagged as delayed", "Apr 7 > Apr 5", True, bool(sale.exit_delayed))
        s.check("BBB sale proceeds", "1,000 × 105 − 105 = 104,895", 104_895, sale.cash_change)
        s.check("BBB final cash", "899,900 + 104,895 = 1,004,795", 1_004_795, r.history.cash.iloc[-1])

        # CCC: a long gap, then quotes come back.
        gap = cal[(cal >= D("2010-02-02")) & (cal <= D("2010-02-10"))]
        ccc = price_path("CCC", cal, changes={"2010-02-11": 90.0}, missing=gap)
        r = bt.Backtester(ccc, cal, CAPITAL, FEE_BPS).run(orders("CCC"), "2010-01-04", "2010-04-30")
        d, h = r.diagnostics, r.history
        s.check("CCC frozen on", "6th missing session: Tue Feb 9", D("2010-02-09"),
                d.loc[d.event.eq("frozen"), "date"].iloc[0])
        s.check("CCC NAV while frozen (Feb 10)", "899,900 + 1,000 × 100 = 999,900", 999_900, h.nav.loc["2010-02-10"])
        s.check("CCC unfrozen on", "quotes resume Thu Feb 11", D("2010-02-11"),
                d.loc[d.event.eq("unfrozen"), "date"].iloc[0])
        s.check("CCC NAV after quotes resume", "899,900 + 1,000 × 90 = 989,900", 989_900, h.nav.loc["2010-02-11"])
        s.check("CCC sold at normal maturity", "Mon Apr 5", D("2010-04-05"), sells(r).iloc[0].date)
        s.check("CCC final cash", "899,900 + 90,000 − 90 = 989,810", 989_810, h.cash.iloc[-1])


def tearDownModule():
    lines = ["# Backtester accounting evidence", "",
             "Generated by `tests/test_backtester.py` (run: `../../../.venv/bin/python -m unittest -v "
             "tests.test_backtester` from this folder). Each row compares arithmetic done by hand with "
             "the value produced by `backtester.py`. Settings: $1,000,000 starting capital, 10 bps "
             "per side (fee included in each buy budget), 90-calendar-day holding period, at most 5 "
             "stale sessions before a stock is frozen. All prices are made up so the arithmetic is "
             "easy to check.", ""]
    for title, setup, rows in sorted(SECTIONS):
        lines += [f"## {title}", "", setup, "",
                  "| Check | By hand | Expected | Code | OK |", "|---|---|---:|---:|:-:|"]
        lines += [f"| {item} | {hand} | {exp} | {got} | {ok} |" for item, hand, exp, got, ok in rows]
        lines.append("")
    total = sum(len(rows) for _, _, rows in SECTIONS)
    passed = sum(row[-1] == "✅" for _, _, rows in SECTIONS for row in rows)
    lines.append(f"**{passed} of {total} checks match.**")
    EVIDENCE_FILE.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    unittest.main()
