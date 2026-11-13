"""Hand-verifiable accounting tests for the backtester.

Run from the repo root:
    uv run python -m unittest discover -s Homeworks/HW4/Antonio_Rivera -v

All numbers below can be checked with a calculator. Default fee_bps = 10 per side.
"""

import unittest

import numpy as np
import pandas as pd

import hw4_backtest as bt

T = pd.Timestamp


class TestOneBuyOneSale(unittest.TestCase):
    """Budget 100,100 at $100 with 10 bps -> exactly 1,000 units and a $100 fee."""

    def setUp(self):
        self.om = bt.OrderManager(1_000_000, fee_bps=10)
        self.lot = self.om.buy("AAA", 100_100, 100.0, "2024-01-02")

    def test_buy_by_hand(self):
        self.assertAlmostEqual(self.lot.units, 1_000.0)          # 100,100 / (100 × 1.001)
        self.assertAlmostEqual(self.lot.entry_fee, 100.0)       # 1,000 × 100 × 0.001
        self.assertAlmostEqual(self.om.cash, 899_900.0)          # 1,000,000 − 100,100
        self.assertEqual(self.lot.due_date, T("2024-04-01"))     # Jan 2 + 90 days

    def test_nav_after_buy_reflects_entry_fee(self):
        # Price unchanged: NAV = 899,900 + 1,000 × 100 = 999,900 (lost only the $100 fee).
        self.assertAlmostEqual(self.om.nav({"AAA": 100.0}), 999_900.0)

    def test_maturity_sale_by_hand(self):
        t = self.om.sell(self.lot.lot_id, 110.0, "2024-04-01")
        # gross 110,000; fee 110; proceeds 109,890
        self.assertAlmostEqual(self.om.cash, 899_900 + 109_890)  # 1,009,790
        self.assertAlmostEqual(t["pnl"], 9_790.0)                # 1,000×10 − 100 − 110
        self.assertAlmostEqual(t["fees"], 210.0)
        self.assertAlmostEqual(t["return"], 9_790 / 100_100)
        self.assertEqual(t["days_held"], 90)
        self.assertEqual(self.om.open_lots, {})
        self.assertAlmostEqual(self.om.fees_paid, 210.0)
        self.assertAlmostEqual(self.om.nav({}), 1_009_790.0)

    def test_fill_log(self):
        self.om.sell(self.lot.lot_id, 110.0, "2024-04-01")
        f = self.om.fills_frame()
        self.assertEqual(f.side.tolist(), ["BUY", "SELL"])
        self.assertEqual(f.cash_change.round(2).tolist(), [-100_100.0, 109_890.0])
        self.assertEqual(f.cash_after.round(2).tolist(), [899_900.0, 1_009_790.0])


class TestTwoLotsSameTicker(unittest.TestCase):
    """Two AAA lots with different maturities are tracked and closed independently."""

    def setUp(self):
        self.om = bt.OrderManager(1_000_000, fee_bps=10)
        self.a = self.om.buy("AAA", 100_100, 100.0, "2024-01-02")   # 1,000 units, due Apr 1
        self.b = self.om.buy("AAA", 50_050, 50.0, "2024-02-01")     # 1,000 units, due May 1

    def test_separate_lots_and_due_dates(self):
        self.assertNotEqual(self.a.lot_id, self.b.lot_id)
        self.assertEqual(self.a.due_date, T("2024-04-01"))
        self.assertEqual(self.b.due_date, T("2024-05-01"))
        self.assertAlmostEqual(self.om.cash, 1_000_000 - 100_100 - 50_050)  # 849,850

    def test_first_sale_leaves_second_lot_intact(self):
        t = self.om.sell(self.a.lot_id, 80.0, "2024-04-01")
        # proceeds = 80,000 − 80 = 79,920 ; pnl = 79,920 − 100,100 = −20,180
        self.assertAlmostEqual(t["pnl"], -20_180.0)
        self.assertEqual(list(self.om.open_lots), [self.b.lot_id])
        self.assertAlmostEqual(self.om.open_lots[self.b.lot_id].entry_price, 50.0)
        # NAV at a common mark of 80: cash 929,770 + 1,000 × 80
        self.assertAlmostEqual(self.om.nav({"AAA": 80.0}), 849_850 + 79_920 + 80_000)

    def test_unrealized_pnl_uses_each_lot_basis(self):
        # At 80: lot a = 80,000 − 100,100 = −20,100 ; lot b = 80,000 − 50,050 = +29,950
        self.assertAlmostEqual(self.om.unrealized_pnl({"AAA": 80.0}), 9_850.0)


class TestCashGuards(unittest.TestCase):

    def setUp(self):
        self.om = bt.OrderManager(100_000, fee_bps=10)

    def test_cannot_spend_more_than_cash(self):
        with self.assertRaises(ValueError):
            self.om.buy("AAA", 100_000.01, 10.0, "2024-01-02")
        self.assertEqual(self.om.cash, 100_000)           # nothing changed
        self.assertEqual(self.om.fills, [])

    def test_cannot_spend_same_cash_twice(self):
        self.om.buy("AAA", 60_000, 10.0, "2024-01-02")
        with self.assertRaises(ValueError):
            self.om.buy("BBB", 60_000, 10.0, "2024-01-02")  # only 40,000 left
        self.om.buy("BBB", 40_000, 10.0, "2024-01-02")       # exactly the rest is fine
        self.assertEqual(self.om.cash, 0.0)

    def test_cannot_sell_twice_or_unknown_lot(self):
        lot = self.om.buy("AAA", 10_000, 10.0, "2024-01-02")
        self.om.sell(lot.lot_id, 10.0, "2024-04-01")
        with self.assertRaises(KeyError):
            self.om.sell(lot.lot_id, 10.0, "2024-04-02")
        with self.assertRaises(KeyError):
            self.om.sell(999, 10.0, "2024-04-02")

    def test_cannot_sell_on_entry_day(self):
        lot = self.om.buy("AAA", 10_000, 10.0, "2024-01-02")
        with self.assertRaises(ValueError):
            self.om.sell(lot.lot_id, 11.0, "2024-01-02")

    def test_bad_prices_rejected(self):
        for p in [0.0, -1.0, np.nan, np.inf]:
            with self.assertRaises(ValueError):
                self.om.buy("AAA", 1_000, p, "2024-01-02")

    def test_missing_mark_is_an_error_not_zero(self):
        self.om.buy("AAA", 10_000, 10.0, "2024-01-02")
        with self.assertRaises(KeyError):
            self.om.nav({})


class TestAccountingIdentity(unittest.TestCase):
    """Random trading: cash never negative, and the books always balance:
        NAV = initial cash + realized PnL + unrealized PnL
        fees_paid = sum of fill fees
    """

    def test_random_sequence(self):
        rng = np.random.default_rng(42)
        om = bt.OrderManager(1_000_000, fee_bps=10)
        syms = ["AAA", "BBB", "CCC"]
        day = T("2024-01-02")
        for _ in range(400):
            day += pd.Timedelta(days=1)
            marks = {s: float(rng.uniform(5, 200)) for s in syms}
            if om.open_lots and rng.random() < 0.4:
                lot_id = rng.choice(list(om.open_lots))
                lot = om.open_lots[lot_id]
                om.sell(lot_id, marks[lot.symbol], day)
            elif om.cash > 1:
                s = syms[rng.integers(3)]
                om.buy(s, om.cash * rng.uniform(0.05, 1.0), marks[s], day)
            self.assertGreaterEqual(om.cash, 0.0)
            nav = om.nav(marks)
            self.assertAlmostEqual(nav, om.initial_cash + om.realized_pnl() + om.unrealized_pnl(marks),
                                   places=4)
        self.assertAlmostEqual(om.fees_paid, om.fills_frame().fee.sum(), places=6)
        self.assertGreater(len(om.closed), 50)       # the test really exercised sales


class TestFeeSetting(unittest.TestCase):
    """fee_bps is a per-run trading-cost assumption."""

    def round_trip(self, fee_bps):
        om = bt.OrderManager(1_000_000, fee_bps=fee_bps)
        lot = om.buy("AAA", 100_000, 100.0, "2024-01-02")
        return om, lot, om.sell(lot.lot_id, 110.0, "2024-04-01")

    def test_default_is_10_bps(self):
        om = bt.OrderManager()
        self.assertEqual(om.fee_bps, 10.0)
        self.assertAlmostEqual(om.fee_rate, 0.001)

    def test_zero_fee_by_hand(self):
        om, lot, t = self.round_trip(0)
        self.assertAlmostEqual(lot.units, 1_000.0)          # 100,000 / 100
        self.assertAlmostEqual(t["pnl"], 10_000.0)          # 1,000 × (110 − 100)
        self.assertEqual(om.fees_paid, 0.0)

    def test_25_bps_by_hand(self):
        om, lot, t = self.round_trip(25)
        units = 100_000 / (100 * 1.0025)                     # 997.506...
        self.assertAlmostEqual(lot.units, units)
        self.assertAlmostEqual(lot.entry_fee, units * 100 * 0.0025)
        exit_fee = units * 110 * 0.0025
        self.assertAlmostEqual(t["pnl"], units * 110 - exit_fee - 100_000)

    def test_higher_fee_never_helps(self):
        pnls = [self.round_trip(b)[2]["pnl"] for b in (0, 5, 10, 25, 50)]
        self.assertEqual(pnls, sorted(pnls, reverse=True))

    def test_change_allowed_before_first_fill(self):
        om = bt.OrderManager(fee_bps=10)
        om.fee_bps = 5
        self.assertAlmostEqual(om.fee_rate, 0.0005)
        lot = om.buy("AAA", 100_050, 100.0, "2024-01-02")
        self.assertAlmostEqual(lot.units, 1_000.0)           # 100,050 / (100 × 1.0005)

    def test_change_blocked_after_first_fill(self):
        om = bt.OrderManager(fee_bps=10)
        om.buy("AAA", 10_000, 10.0, "2024-01-02")
        with self.assertRaises(RuntimeError):
            om.fee_bps = 0
        self.assertEqual(om.fee_bps, 10.0)                   # unchanged

    def test_invalid_fee_rejected(self):
        for bad in (-1, np.nan, np.inf):
            with self.assertRaises(ValueError):
                bt.OrderManager(fee_bps=bad)


class TestDailyBooks(unittest.TestCase):
    """OrderManager.record: cash, NAV, realized and unrealized PnL per day, by hand."""

    def setUp(self):
        om = bt.OrderManager(1_000_000, fee_bps=10)
        om.record("2024-01-02", {})                                   # before any trade
        lot = om.buy("AAA", 100_100, 100.0, "2024-01-03")             # 1,000 units, fee 100
        om.record("2024-01-03", {"AAA": 100.0})
        om.record("2024-01-04", {"AAA": 105.0}, stale_symbols=["AAA"])
        om.sell(lot.lot_id, 110.0, "2024-04-03")                      # proceeds 109,890
        om.record("2024-04-03", {})
        self.om, self.d = om, om.daily_frame()

    def row(self, day):
        return self.d.loc[T(day)]

    def test_initial_row(self):
        r = self.row("2024-01-02")
        self.assertEqual((r.cash, r.nav, r.realized_pnl, r.unrealized_pnl), (1_000_000, 1_000_000, 0, 0))

    def test_after_buy(self):
        r = self.row("2024-01-03")
        self.assertAlmostEqual(r.cash, 899_900)
        self.assertAlmostEqual(r.unrealized_pnl, -100)                # 100,000 − 100,100 basis
        self.assertEqual(r.realized_pnl, 0)
        self.assertAlmostEqual(r.bought, 100_000)
        self.assertAlmostEqual(r.fees, 100)

    def test_mark_up_and_stale_flag(self):
        r = self.row("2024-01-04")
        self.assertAlmostEqual(r.unrealized_pnl, 105_000 - 100_100)   # 4,900
        self.assertAlmostEqual(r.nav, 899_900 + 105_000)
        self.assertEqual(r.stale_symbols, 1)
        self.assertEqual(r.bought, 0)                                 # yesterday's buy not repeated

    def test_after_sale_pnl_moves_to_realized(self):
        r = self.row("2024-04-03")
        self.assertAlmostEqual(r.realized_pnl, 9_790)
        self.assertEqual(r.unrealized_pnl, 0)
        self.assertAlmostEqual(r.cash, 1_009_790)
        self.assertAlmostEqual(r.sold, 110_000)
        self.assertAlmostEqual(r.fees_cum, 210)

    def test_identity_every_day(self):
        lhs = self.d.nav
        rhs = self.om.initial_cash + self.d.realized_pnl + self.d.unrealized_pnl
        np.testing.assert_allclose(lhs, rhs)

    def test_dates_must_increase(self):
        with self.assertRaises(ValueError):
            self.om.record("2024-04-03", {})

    def test_exposure_sums_lots_in_one_symbol(self):
        om = bt.OrderManager(1_000_000, fee_bps=0)
        om.buy("AAA", 1_000, 10.0, "2024-01-02")
        om.buy("AAA", 2_000, 20.0, "2024-01-03")                      # 100 + 100 units
        om.buy("BBB", 500, 5.0, "2024-01-03")
        self.assertEqual(om.exposure({"AAA": 30.0, "BBB": 5.0}), {"AAA": 6_000.0, "BBB": 500.0})


# =============================================================================
# Strategy layer
# =============================================================================

def toy_panel(n_days=30, symbols=("AAA", "BBB", "CCC", "DDD", "EEE")):
    """Business days from 2024-01-02; score = position in `symbols` (EEE highest)."""
    dates = pd.bdate_range("2024-01-02", periods=n_days)
    rows = [{"date": d, "symbol": s, "score": float(i), "eligible": True, "volume": 1_000}
            for d in dates for i, s in enumerate(symbols)]
    return pd.DataFrame(rows), dates


def state(cash=1_000_000.0, nav=1_000_000.0, exposure=None):
    return bt.PortfolioState(cash, nav, exposure or {}, ())


class TestMarketView(unittest.TestCase):

    def setUp(self):
        self.panel, self.dates = toy_panel()
        self.view = bt.MarketView(self.panel, self.dates[10])

    def test_today_is_only_the_cutoff_date(self):
        self.assertEqual(set(self.view.today().date), {self.dates[10]})

    def test_history_stops_at_cutoff(self):
        self.assertEqual(self.view.history().date.max(), self.dates[10])
        self.assertEqual(len(self.view.history()), 11 * 5)

    def test_future_access_raises(self):
        with self.assertRaises(bt.LookAheadError):
            self.view.at(self.dates[11])
        with self.assertRaises(bt.LookAheadError):
            self.view.history(start=self.dates[11])

    def test_returned_data_is_a_copy(self):
        t = self.view.today()
        t["score"] = -999.0
        self.assertNotIn(-999.0, self.panel.score.tolist())


class TestTopNSelection(unittest.TestCase):

    def setUp(self):
        self.panel, self.dates = toy_panel()

    def view(self, panel=None, i=25):
        return bt.MarketView(self.panel if panel is None else panel, self.dates[i])

    def test_picks_highest_scores(self):
        s = bt.TopNScoreStrategy(top_n=2)
        self.assertEqual(s.select(self.view()), ["EEE", "DDD"])

    def test_skips_ineligible_and_missing_scores(self):
        p = self.panel.copy()
        p.loc[p.symbol.eq("EEE"), "eligible"] = False
        p.loc[p.symbol.eq("DDD"), "score"] = np.nan
        self.assertEqual(bt.TopNScoreStrategy(top_n=2).select(self.view(p)), ["CCC", "BBB"])

    def test_ties_break_alphabetically(self):
        p = self.panel.copy()
        p["score"] = 1.0
        self.assertEqual(bt.TopNScoreStrategy(top_n=3).select(self.view(p)), ["AAA", "BBB", "CCC"])

    def test_illiquid_name_excluded(self):
        # EEE trades on only 17 of the last 21 sessions (< 18 required).
        p = self.panel.copy()
        last21 = self.dates[25 - 20: 26]
        zero_days = last21[:4]
        p.loc[p.symbol.eq("EEE") & p.date.isin(zero_days), "volume"] = 0
        self.assertNotIn("EEE", bt.TopNScoreStrategy(top_n=5).select(self.view(p)))
        # 3 zero days -> 18 traded days -> still liquid.
        p.loc[p.symbol.eq("EEE") & p.date.eq(zero_days[0]), "volume"] = 1_000
        self.assertIn("EEE", bt.TopNScoreStrategy(top_n=5).select(self.view(p)))

    def test_future_scores_cannot_influence_today(self):
        p = self.panel.copy()
        p.loc[p.date.gt(self.dates[25]) & p.symbol.eq("AAA"), "score"] = 1e9  # future spike
        self.assertEqual(bt.TopNScoreStrategy(top_n=1).select(self.view(p)), ["EEE"])


class TestFastPaths(unittest.TestCase):
    """Speed-ups must give exactly the same answers as the simple versions."""

    def setUp(self):
        self.panel, self.dates = toy_panel(n_days=40)
        rng = np.random.default_rng(3)
        self.panel["volume"] = np.where(rng.random(len(self.panel)) < 0.15, 0, 1_000)

    def test_marketview_positions_match_scan(self):
        pos = self.panel.groupby("date").indices
        for i in (0, 17, 39):
            slow = bt.MarketView(self.panel, self.dates[i]).today()
            fast = bt.MarketView(self.panel, self.dates[i], pos).today()
            pd.testing.assert_frame_equal(slow, fast)
            with self.assertRaises(bt.LookAheadError):
                bt.MarketView(self.panel, self.dates[i], pos).at(self.dates[i] + pd.Timedelta(days=1))

    def test_traded_days_matches_history_count(self):
        fast_panel = bt.add_traded_days(self.panel, self.dates)
        s = bt.TopNScoreStrategy(top_n=5, min_volume_days=17)
        for i in (20, 30, 39):
            slow = s.liquid_symbols(bt.MarketView(self.panel, self.dates[i]))
            fast = s.liquid_symbols(bt.MarketView(fast_panel, self.dates[i]))
            self.assertEqual(slow, fast)

    def test_traded_days_is_causal(self):
        base = bt.add_traded_days(self.panel, self.dates)
        p = self.panel.copy()
        p.loc[p.date.gt(self.dates[20]), "volume"] = 0          # future volume vanishes
        moved = bt.add_traded_days(p, self.dates)
        upto = base.date.le(self.dates[20])
        pd.testing.assert_series_equal(base.loc[upto, "traded_days"], moved.loc[upto, "traded_days"])

    def test_mcap_suspect_names_skipped(self):
        p = self.panel.assign(volume=1_000, mcap_suspect=False)
        p.loc[p.symbol.eq("EEE"), "mcap_suspect"] = True
        v = bt.MarketView(p, self.dates[30])
        self.assertEqual(bt.TopNScoreStrategy(top_n=1).select(v), ["DDD"])
        self.assertEqual(bt.TopNScoreStrategy(top_n=1, skip_suspect=False).select(v), ["EEE"])


class TestTopNSizing(unittest.TestCase):

    def setUp(self):
        self.panel, self.dates = toy_panel()
        self.view = bt.MarketView(self.panel, self.dates[0])     # Jan 2: first session of month

    def test_budget_split_by_hand(self):
        s = bt.TopNScoreStrategy(top_n=3, cohort_fraction=1 / 3, max_weight=1.0, min_volume_days=1)
        orders = s.plan_orders(self.dates[0], self.view, state())
        self.assertEqual(sorted(orders), ["CCC", "DDD", "EEE"])
        for b in orders.values():
            self.assertAlmostEqual(b, 1_000_000 / 3 / 3)          # 111,111.11

    def test_limited_by_cash(self):
        s = bt.TopNScoreStrategy(top_n=3, max_weight=1.0, min_volume_days=1)
        orders = s.plan_orders(self.dates[0], self.view, state(cash=90_000))
        self.assertEqual(list(orders.values()), [30_000.0] * 3)
        self.assertLessEqual(sum(orders.values()), 90_000)        # never more than cash

    def test_max_weight_cap_with_existing_exposure(self):
        # NAV 1M, cap 5% = 50,000. EEE already holds 45,000 -> room 5,000.
        s = bt.TopNScoreStrategy(top_n=2, max_weight=0.05, min_volume_days=1)
        orders = s.plan_orders(self.dates[0], self.view, state(exposure={"EEE": 45_000}))
        self.assertAlmostEqual(orders["EEE"], 5_000)
        self.assertAlmostEqual(orders["DDD"], 50_000)             # per-name 166,667 capped at 50,000

    def test_rank_weighting_by_hand(self):
        # 3 picks, cohort 600 -> weights 3:2:1 of 6 -> 300, 200, 100 (best = EEE).
        s = bt.TopNScoreStrategy(top_n=3, cohort_fraction=1.0, max_weight=1.0,
                                 min_volume_days=1, weighting="rank")
        orders = s.plan_orders(self.dates[0], self.view, state(cash=600, nav=600))
        self.assertEqual({k: round(v, 6) for k, v in orders.items()},
                         {"EEE": 300.0, "DDD": 200.0, "CCC": 100.0})

    def test_weights_sum_to_one(self):
        for mode in ("equal", "rank"):
            w = bt.TopNScoreStrategy(weighting=mode).weights(100)
            self.assertAlmostEqual(w.sum(), 1.0)
            self.assertTrue((np.diff(w) <= 0).all())         # never more to a worse rank

    def test_full_name_gets_no_order(self):
        s = bt.TopNScoreStrategy(top_n=1, max_weight=0.05, min_volume_days=1)
        orders = s.plan_orders(self.dates[0], self.view, state(exposure={"EEE": 50_000}))
        self.assertEqual(orders, {})

    def test_orders_only_on_first_session_of_month(self):
        p, dates = toy_panel(n_days=45)                            # Jan 2 .. Mar 4
        s = bt.TopNScoreStrategy(top_n=1, min_volume_days=1)
        signal_days = [d for d in dates
                       if s.plan_orders(d, bt.MarketView(p, d), state())]
        self.assertEqual(signal_days, [T("2024-01-02"), T("2024-02-01"), T("2024-03-01")])

    def test_weekly_frequency_signals_first_session_of_each_week(self):
        p, dates = toy_panel(n_days=15)                       # Tue Jan 2 .. Mon Jan 22
        s = bt.TopNScoreStrategy(top_n=1, min_volume_days=1, frequency="W")
        days = [d for d in dates if s.plan_orders(d, bt.MarketView(p, d), state())]
        self.assertEqual(days, [T("2024-01-02"), T("2024-01-08"), T("2024-01-15"), T("2024-01-22")])

    def test_weekly_cohorts_keep_cash_low(self):
        """Engine run, flat prices: weekly 1/12 cohorts end up ~fully invested."""
        cal = pd.bdate_range("2023-01-02", "2024-06-28")
        rows = [{"date": d, "symbol": f"S{i}", "adj_close": 100.0, "volume": 1_000,
                 "eligible": True, "score": float(i)} for d in cal for i in range(30)]
        panel = pd.DataFrame(rows)
        res = {}
        for freq, frac in (("M", 1 / 3), ("W", 1 / 12)):
            strat = bt.TopNScoreStrategy(top_n=30, cohort_fraction=frac, max_weight=1.0, frequency=freq)
            run_ = bt.BacktestEngine(panel, cal, strat, bt.OrderManager(fee_bps=0),
                                     "2023-02-01", "2024-06-28").run()
            d = run_.daily.loc["2023-09-01":]                  # after the ramp-up
            res[freq] = (d.cash / d.nav).mean()
        self.assertLess(res["W"], 0.10)
        self.assertGreater(res["M"], res["W"] + 0.15)          # weekly fixes the drag

    def test_warm_up_no_orders_without_volume_history(self):
        # Default screen needs 18 traded days in the last 21: on day 1 nothing qualifies,
        # so the January cohort is skipped (by design, not retried later in the month).
        p, dates = toy_panel(n_days=45)
        s = bt.TopNScoreStrategy(top_n=1)
        signal_days = [d for d in dates if s.plan_orders(d, bt.MarketView(p, d), state())]
        self.assertEqual(signal_days, [T("2024-02-01"), T("2024-03-01")])


class TestSellRule(unittest.TestCase):

    def lot(self, lot_id, sym, due):
        return bt.Lot(lot_id, sym, 1.0, T("2024-01-01"), 10.0, 0.0, T(due))

    def test_due_and_quoted_is_sold(self):
        lots = [self.lot(1, "AAA", "2024-04-01")]
        self.assertEqual(bt.Strategy().lots_to_sell("2024-04-01", lots, {"AAA"}), [1])

    def test_not_yet_due_is_kept(self):
        lots = [self.lot(1, "AAA", "2024-04-02")]
        self.assertEqual(bt.Strategy().lots_to_sell("2024-04-01", lots, {"AAA"}), [])

    def test_weekend_maturity_sells_next_session(self):
        # Due Sunday 2024-03-31 -> first session Monday 2024-04-01 sells it.
        lots = [self.lot(1, "AAA", "2024-03-31")]
        self.assertEqual(bt.Strategy().lots_to_sell("2024-04-01", lots, {"AAA"}), [1])

    def test_due_but_no_quote_is_held(self):
        lots = [self.lot(1, "AAA", "2024-04-01"), self.lot(2, "BBB", "2024-04-01")]
        self.assertEqual(bt.Strategy().lots_to_sell("2024-04-01", lots, {"BBB"}), [2])

    def test_two_lots_same_ticker_sold_at_own_maturity(self):
        lots = [self.lot(1, "AAA", "2024-04-01"), self.lot(2, "AAA", "2024-05-01")]
        self.assertEqual(bt.Strategy().lots_to_sell("2024-04-01", lots, {"AAA"}), [1])
        self.assertEqual(bt.Strategy().lots_to_sell("2024-05-01", lots, {"AAA"}), [1, 2])


# =============================================================================
# Engine: hand-checked toy runs
# =============================================================================
#
# Calendar: weekdays 2024-01-02 .. 2024-06-28, with Tue 2024-04-02 a market holiday.
#   Signal Tue Jan 2 close -> fill Wed Jan 3 close -> due Jan 3 + 90 = Tue Apr 2
#   (holiday) -> sold at the next session, Wed Apr 3.
#
# AAA: 95 on Jan 2 (signal day: must NOT be the fill price), 100 from Jan 3 to
#      Apr 1, 110 from Apr 3 on.
# BBB: always 50, but no quotes Apr 3-4 (due-date gap) -> sold Fri Apr 5.
# CCC: no quote on Jan 3 -> any order for it is cancelled.

HOLIDAY = T("2024-04-02")
CAL = pd.bdate_range("2024-01-02", "2024-06-28").drop(HOLIDAY)


def engine_panel(after_end_price=None):
    rows = []
    for d in CAL:
        aaa = 95.0 if d == T("2024-01-02") else (100.0 if d < HOLIDAY else 110.0)
        rows.append({"date": d, "symbol": "AAA", "adj_close": aaa})
        if d not in (T("2024-04-03"), T("2024-04-04")):
            rows.append({"date": d, "symbol": "BBB", "adj_close": 50.0})
        if d != T("2024-01-03"):
            rows.append({"date": d, "symbol": "CCC", "adj_close": 20.0})
    p = pd.DataFrame(rows)
    if after_end_price is not None:              # poison every price after Jun 3
        p.loc[p.date.gt(T("2024-06-03")), "adj_close"] = after_end_price
    return p


class ScriptedStrategy(bt.Strategy):
    """Places preset orders on preset signal dates; inherits the maturity sell rule."""

    def __init__(self, script):
        self.script = {T(k): v for k, v in script.items()}
        self.calls = []

    def plan_orders(self, date, view, state):
        self.calls.append(date)
        return dict(self.script.get(date, {}))


def run(script, start="2024-01-02", end="2024-06-28", panel=None, fee_bps=10):
    strat = ScriptedStrategy(script)
    eng = bt.BacktestEngine(engine_panel() if panel is None else panel, CAL, strat,
                            bt.OrderManager(1_000_000, fee_bps=fee_bps), start, end)
    return eng.run(), strat


class TestEngineOneLot(unittest.TestCase):
    """One AAA lot: every number checkable by hand."""

    @classmethod
    def setUpClass(cls):
        cls.res, cls.strat = run({"2024-01-02": {"AAA": 100_100}})
        cls.d = cls.res.daily

    def test_first_row_is_initial_capital(self):
        self.assertEqual(self.d.index[0], T("2024-01-02"))
        self.assertEqual(self.d.nav.iloc[0], 1_000_000)
        self.assertEqual(self.d.open_lots.iloc[0], 0)

    def test_fill_is_next_session_close_not_signal_close(self):
        buy = self.res.fills.iloc[0]
        self.assertEqual(buy.date, T("2024-01-03"))
        self.assertEqual(buy.price, 100.0)                   # not the 95 signal-day close
        self.assertAlmostEqual(buy.units, 1_000.0)            # 100,100 / (100 × 1.001)

    def test_nav_after_fill(self):
        self.assertAlmostEqual(self.d.loc[T("2024-01-03"), "cash"], 899_900)
        self.assertAlmostEqual(self.d.loc[T("2024-01-03"), "nav"], 999_900)

    def test_holiday_maturity_sold_next_session(self):
        sell = self.res.fills.iloc[1]
        self.assertEqual(sell.side, "SELL")
        self.assertEqual(sell.date, T("2024-04-03"))          # due Apr 2 = holiday
        self.assertEqual(sell.price, 110.0)
        self.assertNotIn(HOLIDAY, self.d.index)               # no row on a closed day
        self.assertEqual(self.d.loc[T("2024-04-01"), "open_lots"], 1)   # held to maturity

    def test_trade_pnl_and_final_cash(self):
        t = self.res.trades.iloc[0]
        self.assertAlmostEqual(t.pnl, 9_790.0)                 # 1,000 × 10 − 100 − 110
        self.assertEqual(t.days_held, 91)                      # Jan 3 -> Apr 3
        self.assertAlmostEqual(self.d.nav.iloc[-1], 1_009_790)
        self.assertAlmostEqual(self.d.cash.iloc[-1], 1_009_790)

    def test_realized_and_unrealized_pnl_from_the_books(self):
        self.assertAlmostEqual(self.d.loc[T("2024-01-03"), "unrealized_pnl"], -100)   # entry fee
        self.assertAlmostEqual(self.d.loc[T("2024-04-01"), "unrealized_pnl"], -100)
        self.assertAlmostEqual(self.d.loc[T("2024-04-03"), "realized_pnl"], 9_790)
        self.assertEqual(self.d.loc[T("2024-04-03"), "unrealized_pnl"], 0)
        np.testing.assert_allclose(self.d.nav, 1_000_000 + self.d.realized_pnl + self.d.unrealized_pnl)

    def test_nav_equals_cash_plus_marked_holdings_every_day(self):
        for d, row in self.d.iterrows():
            aaa = 95.0 if d == T("2024-01-02") else (100.0 if d < HOLIDAY else 110.0)
            expected = row.cash + (1_000 * aaa if row.open_lots else 0.0)
            self.assertAlmostEqual(row.nav, expected, places=6, msg=str(d.date()))

    def test_no_planning_on_last_session(self):
        self.assertEqual(self.strat.calls[-1], CAL[-2])


class TestEngineMissingQuotes(unittest.TestCase):

    def test_due_lot_without_quote_held_then_sold_on_next_quote(self):
        res, _ = run({"2024-01-02": {"BBB": 50_050}})          # 1,000 units at 50
        sell = res.fills.loc[res.fills.side.eq("SELL")].iloc[0]
        self.assertEqual(sell.date, T("2024-04-05"))           # Apr 3, 4 have no quote
        d = res.daily
        for day in ("2024-04-03", "2024-04-04"):               # marked at last real quote
            self.assertAlmostEqual(d.loc[T(day), "holdings"], 50_000)
            self.assertEqual(d.loc[T(day), "stale_symbols"], 1)
        due = res.diagnostics.loc[res.diagnostics.event.eq("due_lot_no_quote")]
        self.assertEqual(due.date.tolist(), [T("2024-04-03"), T("2024-04-04")])

    def test_order_without_fill_quote_is_cancelled(self):
        res, _ = run({"2024-01-02": {"CCC": 10_000, "AAA": 10_000}})
        self.assertEqual(res.fills.symbol.tolist()[:1], ["AAA"])   # only AAA filled
        self.assertNotIn("CCC", res.fills.symbol.tolist())
        cancel = res.diagnostics.loc[res.diagnostics.event.eq("order_cancelled_no_quote")]
        self.assertEqual(cancel.symbol.tolist(), ["CCC"])
        self.assertAlmostEqual(res.daily.loc[T("2024-01-03"), "cash"], 990_000)  # CCC cash kept


class TestEngineEndOfTest(unittest.TestCase):

    def test_open_lot_valued_at_end_close_and_no_later_prices(self):
        # End Jun 3, while a lot bought Apr 2's next session is still open.
        clean, _ = run({"2024-04-01": {"AAA": 110_110}}, end="2024-06-03")
        poisoned, _ = run({"2024-04-01": {"AAA": 110_110}}, end="2024-06-03",
                          panel=engine_panel(after_end_price=1e6))
        pd.testing.assert_frame_equal(clean.daily, poisoned.daily)   # post-end prices unused
        lot = clean.open_lots.iloc[0]
        self.assertEqual(lot.due_date, T("2024-07-02"))              # Apr 3 + 90: after end
        self.assertAlmostEqual(lot.end_value, 1_000 * 110)           # valued at Jun 3 close
        self.assertEqual(clean.daily.index[-1], T("2024-06-03"))


class TestEngineGuards(unittest.TestCase):

    def test_overspending_plan_rejected(self):
        with self.assertRaises(ValueError):
            run({"2024-01-02": {"AAA": 600_000, "BBB": 600_000}})

    def test_early_sale_rejected(self):
        class SellEarly(ScriptedStrategy):
            def lots_to_sell(self, date, open_lots, quoted):
                return [l.lot_id for l in open_lots]           # ignores maturity
        eng = bt.BacktestEngine(engine_panel(), CAL, SellEarly({"2024-01-02": {"AAA": 1_000}}),
                                bt.OrderManager(), "2024-01-02", "2024-06-28")
        with self.assertRaises(RuntimeError):
            eng.run()

    def test_strategy_peeking_raises(self):
        class Peeker(bt.Strategy):
            def plan_orders(self, date, view, state):
                view.at(date + pd.Timedelta(days=1))           # tomorrow
                return {}
        eng = bt.BacktestEngine(engine_panel(), CAL, Peeker(), bt.OrderManager(),
                                "2024-01-02", "2024-06-28")
        with self.assertRaises(bt.LookAheadError):
            eng.run()

    def test_off_calendar_rows_rejected(self):
        p = pd.concat([engine_panel(), pd.DataFrame([{"date": HOLIDAY, "symbol": "AAA",
                                                      "adj_close": 1.0}])])
        with self.assertRaises(ValueError):
            bt.BacktestEngine(p, CAL, bt.Strategy(), bt.OrderManager(), "2024-01-02", "2024-06-28")


class TestEngineWithTopN(unittest.TestCase):
    """Random prices + the real TopN strategy: the books balance every day."""

    def test_invariants_on_random_run(self):
        rng = np.random.default_rng(7)
        cal = pd.bdate_range("2023-01-02", "2024-12-31")
        syms = [f"S{i:02d}" for i in range(12)]
        rows = []
        for s in syms:
            px = 50 * np.cumprod(1 + rng.normal(0.0004, 0.02, len(cal)))
            for d, p in zip(cal, px):
                if rng.random() > 0.01:                        # ~1% random missing quotes
                    rows.append({"date": d, "symbol": s, "adj_close": p, "volume": 1_000,
                                 "eligible": True, "score": rng.normal()})
        panel = pd.DataFrame(rows)
        om = bt.OrderManager(1_000_000, fee_bps=10)
        strat = bt.TopNScoreStrategy(top_n=4, max_weight=0.15)
        res = bt.BacktestEngine(panel, cal, strat, om, "2023-03-01", "2024-12-31").run()
        d = res.daily
        self.assertTrue((d.cash >= 0).all())
        self.assertGreater(len(res.trades), 20)                # many lots opened and closed
        # Cash ledger: initial + Σ fill cash changes = final cash.
        self.assertAlmostEqual(1_000_000 + res.fills.cash_change.sum(), d.cash.iloc[-1], places=4)
        # Every closed lot held at least 90 calendar days.
        self.assertTrue((res.trades.days_held >= 90).all())
        # Fees: 10 bps of every notional traded.
        self.assertAlmostEqual(res.fills.fee.sum(), 0.001 * res.fills.notional.sum(), places=4)
        # Final NAV = cash + open lots at their end marks.
        end_val = res.open_lots.end_value.sum() if len(res.open_lots) else 0.0
        self.assertAlmostEqual(d.nav.iloc[-1], d.cash.iloc[-1] + end_val, places=4)


# =============================================================================
# Analytics
# =============================================================================

class TestMetrics(unittest.TestCase):

    def test_max_drawdown_by_hand(self):
        nav = pd.Series([100, 110, 99, 121.0], index=pd.bdate_range("2024-01-02", periods=4))
        m = bt.performance_metrics(nav)
        self.assertAlmostEqual(m["max_drawdown"], 99 / 110 - 1)        # −10%
        self.assertEqual(m["dd_peak"], nav.index[1].date())
        self.assertEqual(m["dd_trough"], nav.index[2].date())
        self.assertAlmostEqual(m["total_return"], 0.21)
        self.assertEqual(m["final_value"], 121.0)

    def test_cagr_by_hand(self):
        nav = pd.Series([100.0, 121.0], index=[T("2020-01-01"), T("2022-01-01")])  # 731 days
        expected = 1.21 ** (365.25 / 731) - 1                                       # ≈ 9.99%
        self.assertAlmostEqual(bt.performance_metrics(nav)["cagr"], expected)

    def test_sharpe_matches_independent_formula(self):
        r = np.array([0.01, -0.005, 0.02, 0.0, 0.003])
        nav = pd.Series(100 * np.cumprod(np.r_[1, 1 + r]), index=pd.bdate_range("2024-01-02", periods=6))
        rf_d = 1.02 ** (1 / 252) - 1
        expected = (r.mean() - rf_d) / r.std(ddof=1) * np.sqrt(252)
        self.assertAlmostEqual(bt.performance_metrics(nav)["sharpe"], expected)

    def test_flat_nav_has_no_sharpe(self):
        nav = pd.Series(100.0, index=pd.bdate_range("2024-01-02", periods=10))
        m = bt.performance_metrics(nav)
        self.assertTrue(np.isnan(m["sharpe"]))
        self.assertEqual(m["max_drawdown"], 0.0)


class TestBenchmark(unittest.TestCase):

    def test_invested_at_first_close_by_hand(self):
        px = pd.Series([200.0, 210.0, 190.0, 999.0], index=pd.bdate_range("2024-01-02", periods=4))
        b = bt.benchmark_nav(px, px.index[:3])
        self.assertEqual(b.tolist(), [1_000_000.0, 1_050_000.0, 950_000.0])  # 999 is after the end

    def test_missing_benchmark_close_is_an_error(self):
        px = pd.Series([200.0, 210.0], index=[T("2024-01-02"), T("2024-01-04")])
        with self.assertRaises(ValueError):
            bt.benchmark_nav(px, [T("2024-01-02"), T("2024-01-03")])


class TestReportOnToyRun(unittest.TestCase):
    """One AAA round trip (see engine tests): 1,000,000 -> 1,009,790."""

    @classmethod
    def setUpClass(cls):
        res, _ = run({"2024-01-02": {"AAA": 100_100}})
        bench_px = pd.Series(np.linspace(100, 110, len(CAL)), index=CAL)   # +10% over the run
        cls.rep = bt.PerformanceReport(res, bench_px)

    def test_summary_values(self):
        s = self.rep.summary()
        self.assertAlmostEqual(s.loc["final_value", "Strategy"], 1_009_790)
        self.assertAlmostEqual(s.loc["final_value", "Benchmark"], 1_100_000)
        self.assertEqual(s.loc["start", "Strategy"], s.loc["start", "Benchmark"])  # same first close

    def test_verdict(self):
        v = self.rep.verdict()
        self.assertFalse(v["beat_benchmark"])
        self.assertAlmostEqual(v["excess_final_value"], 1_009_790 - 1_100_000)

    def test_trading_stats_by_hand(self):
        tr = self.rep.trading()
        self.assertAlmostEqual(tr["fees_total"], 210)
        self.assertAlmostEqual(tr["fees_pct_of_start"], 210 / 1_000_000)
        self.assertEqual(tr["lots_closed"], 1)
        self.assertEqual(tr["win_rate"], 1.0)
        self.assertAlmostEqual(tr["mean_lot_return"], 9_790 / 100_100)
        self.assertAlmostEqual(tr["realized_pnl"], 9_790)
        # Cash weight: 100% before the buy and after the sale; ~90% while holding.
        self.assertAlmostEqual(tr["max_cash_weight_after_first_buy"], 1.0)
        self.assertGreater(tr["mean_cash_weight"], 0.9)

    def test_chart_is_written(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            path = self.rep.plot(f"{tmp}/chart.png")
            with open(path, "rb") as f:
                self.assertEqual(f.read(8), b"\x89PNG\r\n\x1a\n")


if __name__ == "__main__":
    unittest.main()
