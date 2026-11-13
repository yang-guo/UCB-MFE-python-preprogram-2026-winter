"""Unit tests for hw4_data.py (label timing + audit).

Run from the repo root:
    uv run python -m unittest discover -s Homeworks/HW4/Antonio_Rivera -v

Part A: toy calendar and prices small enough to verify by hand.
Part B: mutation tests — break a correct label and confirm the audit catches it.
Part C: integration — the audit passes on the real class database.
"""

import unittest

import numpy as np
import pandas as pd

import hw4_data as hd

# Toy market calendar: weekdays Jan 2 – Jun 28, 2024, with Tue Apr 2 a "holiday".
HOLIDAY = pd.Timestamp("2024-04-02")
CAL = pd.bdate_range("2024-01-02", "2024-06-28").drop(HOLIDAY)


def toy_prices():
    """Three stocks, adj_close = 100 everywhere except the hand-set cells below.

    HOL : exit on the day after the holiday maturity is 110.
    GAP : no quotes Apr 3–4; next quote Apr 5 is 120 (2 sessions late, allowed).
    DARK: no quotes Apr 3–12 (8 sessions, > MAX_EXIT_DELAY) -> censored.
    """
    rows = []
    for sym in ["HOL", "GAP", "DARK"]:
        for d in CAL:
            if sym == "GAP" and d in (pd.Timestamp("2024-04-03"), pd.Timestamp("2024-04-04")):
                continue
            if sym == "DARK" and pd.Timestamp("2024-04-03") <= d <= pd.Timestamp("2024-04-12"):
                continue
            px = 100.0
            if sym == "HOL" and d == pd.Timestamp("2024-04-03"):
                px = 110.0
            if sym == "GAP" and d == pd.Timestamp("2024-04-05"):
                px = 120.0
            rows.append((sym, d, px))
    return pd.DataFrame(rows, columns=["symbol", "date", "adj_close"])


def toy_panel():
    """Label every quoted (symbol, date) with the cited course make_labels."""
    prices = toy_prices()
    panel = prices[["symbol", "date"]].copy()
    panel = hd.cd.make_labels(panel, prices, CAL, hd.HOLDING_DAYS, hd.MAX_EXIT_DELAY)
    return panel, prices


def row(panel, sym, date):
    return panel.loc[panel.symbol.eq(sym) & panel.date.eq(pd.Timestamp(date))].iloc[0]


class TestToyLabels(unittest.TestCase):
    """Part A: hand-verifiable cases."""

    @classmethod
    def setUpClass(cls):
        cls.panel, cls.prices = toy_panel()

    def test_holiday_maturity_rolls_to_next_session(self):
        # Signal Tue Jan 2 -> entry Wed Jan 3 -> due Tue Apr 2 (holiday) -> exit Wed Apr 3.
        r = row(self.panel, "HOL", "2024-01-02")
        self.assertEqual(r.entry_date, pd.Timestamp("2024-01-03"))
        self.assertEqual(r.scheduled_exit, HOLIDAY)
        self.assertEqual(r.label_end, pd.Timestamp("2024-04-03"))
        self.assertAlmostEqual(r.target, 110 / 100 - 1)  # 0.10

    def test_friday_signal_enters_monday(self):
        # Signal Fri Jan 5 -> entry Mon Jan 8 (weekend skipped).
        r = row(self.panel, "HOL", "2024-01-05")
        self.assertEqual(r.entry_date, pd.Timestamp("2024-01-08"))

    def test_missing_exit_quote_uses_first_later_quote(self):
        # Signal Wed Jan 3 -> entry Thu Jan 4 -> due Wed Apr 3; no quote Apr 3–4 -> Fri Apr 5.
        r = row(self.panel, "GAP", "2024-01-03")
        self.assertEqual(r.scheduled_exit, pd.Timestamp("2024-04-03"))
        self.assertEqual(r.label_end, pd.Timestamp("2024-04-05"))
        self.assertAlmostEqual(r.target, 120 / 100 - 1)  # 0.20

    def test_long_gap_is_censored_not_filled(self):
        # Same dates as GAP, but the next quote is 8 sessions late -> no label.
        r = row(self.panel, "DARK", "2024-01-03")
        self.assertTrue(pd.isna(r.label_end))
        self.assertTrue(np.isnan(r.target))

    def test_no_label_uses_price_after_data_end(self):
        lab = self.panel.loc[self.panel.target.notna()]
        self.assertLessEqual(lab.label_end.max(), CAL[-1])
        # Signals whose maturity is past the last session must be censored.
        late = self.panel.scheduled_exit > CAL[-1]
        self.assertTrue(late.any())
        self.assertTrue(self.panel.loc[late, "target"].isna().all())

    def test_last_session_has_no_entry(self):
        r = row(self.panel, "HOL", CAL[-1])
        self.assertTrue(pd.isna(r.entry_date))

    def test_audit_passes_on_correct_toy_labels(self):
        stats = hd.audit_label_timing(self.panel, self.prices, CAL)
        self.assertGreater(stats["exit_rolled_forward"], 0)
        self.assertGreater(stats["censored_or_missing"], 0)


class TestAuditCatchesBugs(unittest.TestCase):
    """Part B: each test injects one timing bug; the audit must raise."""

    def setUp(self):
        self.panel, self.prices = toy_panel()
        self.i = row(self.panel, "HOL", "2024-01-02").name  # index of a labeled row

    def assert_audit_fails(self, msg=None):
        if msg:
            with self.assertRaisesRegex(AssertionError, msg):
                hd.audit_label_timing(self.panel, self.prices, CAL)
        else:
            with self.assertRaises(AssertionError):
                hd.audit_label_timing(self.panel, self.prices, CAL)

    def test_same_day_entry_is_caught(self):
        # Look-ahead bug: buying at the close the features already used.
        self.panel.loc[self.i, "entry_date"] = self.panel.loc[self.i, "date"]
        self.assert_audit_fails("next session")

    def test_exit_before_maturity_is_caught(self):
        self.panel.loc[self.i, "label_end"] = pd.Timestamp("2024-04-01")
        self.assert_audit_fails("before maturity")

    def test_skipped_quote_is_caught(self):
        # Exit one quote late, with price and target made self-consistent,
        # so only check 5 ("first quote") can detect it.
        later = pd.Timestamp("2024-04-04")
        self.panel.loc[self.i, "label_end"] = later
        self.panel.loc[self.i, "exit_price"] = 100.0
        self.panel.loc[self.i, "target"] = 0.0
        self.assert_audit_fails("skipped a quote")

    def test_wrong_target_is_caught(self):
        self.panel.loc[self.i, "target"] += 0.01
        self.assert_audit_fails()

    def test_wrong_holding_period_is_caught(self):
        self.panel.loc[self.i, "scheduled_exit"] += pd.Timedelta(days=1)
        self.assert_audit_fails()


@unittest.skipUnless(hd.CLASS_DB.is_file(), "class database not downloaded")
class TestClassDatabase(unittest.TestCase):
    """Part C: the full audit on the real 30-stock development database."""

    def test_audit_passes(self):
        panel, prices, calendar = hd.build_panel(hd.CLASS_DB, end="2015-12-31")
        stats = hd.audit_label_timing(panel, prices, calendar)
        self.assertEqual(stats["rows"], len(panel))
        self.assertLessEqual(pd.Timestamp(stats["last_label_end"]), pd.Timestamp("2015-12-31"))

    def test_aapl_hand_checked_row(self):
        # Verified from raw SQL: entry Mon 2012-07-02 adj 17.74, exit Mon 2012-10-01 adj 19.83.
        panel, _, _ = hd.build_panel(hd.CLASS_DB, end="2015-12-31")
        r = row(panel, "AAPL", "2012-06-29")
        self.assertEqual(r.entry_date, pd.Timestamp("2012-07-02"))
        self.assertEqual(r.label_end, pd.Timestamp("2012-10-01"))
        self.assertAlmostEqual(r.target, 19.83 / 17.74 - 1, places=6)


if __name__ == "__main__":
    unittest.main()
