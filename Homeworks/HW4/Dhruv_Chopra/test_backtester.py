"""Hand-checkable tests of the backtester's accounting and timing.

Run:  ../../../.venv/bin/python -m unittest test_backtester -v
(from this folder). No database is needed: every case uses tiny synthetic
prices whose answers can be worked out on paper.
"""

import unittest

import numpy as np
import pandas as pd

from backtester import Backtester, Ledger, performance
from data import MarketData, clean_prices
from features import labels

FEE = 10 / 10_000


def make_data(prices, start="2024-01-02"):
    """prices: dict symbol -> list of adjusted closes (NaN = no quote)."""
    T = len(next(iter(prices.values())))
    calendar = pd.bdate_range(start, periods=T)   # weekdays only
    symbols = pd.Index(sorted(prices))
    px = np.array([prices[s] for s in symbols], dtype=np.float32).T
    return MarketData(
        calendar=calendar, symbols=symbols, adj_close=px,
        dollar_volume=np.where(np.isfinite(px), 1e9, np.nan).astype(np.float32),
        member=np.ones_like(px, dtype=bool), sector=pd.Series("Tech", index=symbols),
        benchmark=pd.Series(np.linspace(100, 110, T), index=calendar),
        fundamentals=pd.DataFrame(), repairs=pd.DataFrame())


class ScriptedStrategy:
    """Places pre-written orders: {signal session index: {symbol index: budget}}."""

    def __init__(self, script):
        self.script = script
        self.seen = []

    def plan(self, t, snap):
        self.seen.append((t, snap.cash, snap.nav))
        return self.script.get(t, {})


class LedgerArithmetic(unittest.TestCase):
    def test_buy_and_sell_by_hand(self):
        ledger = Ledger(1_000_000, fee_bps=10)
        d0, d1 = pd.Timestamp("2024-01-03"), pd.Timestamp("2024-04-02")
        lot = ledger.buy(0, "A", 10_000, 10.0, d0, d0 - pd.Timedelta(days=1), 90)
        units = 10_000 / (10 * 1.001)                     # 999.000999...
        self.assertAlmostEqual(lot.units, units, places=9)
        self.assertAlmostEqual(ledger.cash, 990_000.0, places=6)   # budget includes the fee
        self.assertAlmostEqual(lot.entry_fee, units * 10 * FEE, places=9)
        self.assertEqual(lot.due_date, d0 + pd.Timedelta(days=90))
        pnl = ledger.sell(lot.lot_id, 12.0, d1)
        proceeds = units * 12 * (1 - FEE)
        self.assertAlmostEqual(ledger.cash, 990_000 + proceeds, places=6)
        self.assertAlmostEqual(pnl, proceeds - 10_000, places=6)   # PnL net of both fees
        self.assertEqual(len(ledger.lots), 0)

    def test_cannot_spend_same_cash_twice(self):
        ledger = Ledger(1_000, fee_bps=10)
        d = pd.Timestamp("2024-01-03")
        ledger.buy(0, "A", 600, 10.0, d, d - pd.Timedelta(days=1), 90)
        with self.assertRaises(ValueError):
            ledger.buy(1, "B", 600, 10.0, d, d - pd.Timedelta(days=1), 90)

    def test_signal_must_precede_fill(self):
        ledger = Ledger(1_000)
        d = pd.Timestamp("2024-01-03")
        with self.assertRaises(ValueError):
            ledger.buy(0, "A", 100, 10.0, d, d, 90)


class BacktestTiming(unittest.TestCase):
    def setUp(self):
        T = 80
        a = np.full(T, 10.0)
        a[40:] = 12.0                                  # A rises 20% on session 40
        b = np.full(T, 20.0)
        b[3] = np.nan                                  # B has no quote on session 3
        self.data = make_data({"A": list(a), "B": list(b)})

    def test_fill_exit_and_nav(self):
        data = self.data
        # Signal at session 1 (close) -> fill at session 2's close.
        strat = ScriptedStrategy({1: {0: 100_000.0}})
        res = Backtester(data, strat, fee_bps=10).run(data.calendar[2], data.calendar[-1])
        buy = res.fills.iloc[0]
        self.assertEqual(buy.date, data.calendar[2])
        self.assertEqual(buy.signal_date, data.calendar[1])
        units = 100_000 / (10 * 1.001)
        self.assertAlmostEqual(buy.units, units, places=6)
        # Entry Thu 2024-01-04 + 90 days = Wed 2024-04-03, a session -> sell there at 12.
        sell = res.fills.iloc[1]
        self.assertEqual(sell.date, pd.Timestamp("2024-04-03"))
        self.assertAlmostEqual(sell.cash_change, units * 12 * (1 - FEE), places=6)
        # NAV on a day in between = cash + units * mark.
        day = data.calendar[45]
        self.assertAlmostEqual(res.daily.loc[day, "nav"], 900_000 + units * 12, places=4)
        # After the exit everything is cash.
        self.assertAlmostEqual(res.daily.nav.iloc[-1], 900_000 + units * 12 * (1 - FEE), places=4)

    def test_weekend_maturity_rolls_to_next_session(self):
        data = self.data
        # Fill Mon 2024-01-08 (session 4) -> due Sun 2024-04-07 (2024 is a leap year)
        # -> no session that day, so the lot is sold at Monday 2024-04-08's close.
        strat = ScriptedStrategy({3: {0: 50_000.0}})
        res = Backtester(data, strat).run(data.calendar[4], data.calendar[-1])
        self.assertEqual(res.fills.iloc[0].date, pd.Timestamp("2024-01-08"))
        self.assertEqual(res.fills.iloc[1].date, pd.Timestamp("2024-04-08"))

    def test_buy_without_quote_is_rejected_not_invented(self):
        data = self.data
        strat = ScriptedStrategy({2: {1: 10_000.0}})      # B has no quote on session 3
        res = Backtester(data, strat).run(data.calendar[3], data.calendar[10])
        self.assertTrue(res.fills.empty)
        self.assertEqual(res.events.event.tolist(), ["buy_rejected_no_quote"])
        self.assertAlmostEqual(res.daily.nav.iloc[-1], 1_000_000.0)

    def test_missing_exit_quote_waits_for_next_quote(self):
        T = 80
        a = np.full(T, 10.0)
        data0 = make_data({"A": list(a)})
        due_session = data0.calendar.get_loc(pd.Timestamp("2024-04-03"))
        a[due_session:due_session + 2] = np.nan            # no quote on the due date and the next
        data = make_data({"A": list(a)})
        res = Backtester(data, ScriptedStrategy({1: {0: 10_000.0}})).run(data.calendar[2], data.calendar[-1])
        self.assertEqual(res.fills.iloc[1].date, data.calendar[due_session + 2])
        self.assertIn("exit_waiting_for_quote", set(res.events.event))
        # While unquoted, the lot is marked at its last price and counted as stale.
        self.assertEqual(res.daily.loc[data.calendar[due_session], "stale_names"], 1)

    def test_reused_ticker_after_long_gap_is_frozen_not_sold(self):
        T = 120
        a = np.full(T, 10.0)
        a[20:80] = np.nan                                  # acquired: quotes stop for 60 sessions...
        a[80:] = 500.0                                     # ...then the ticker is reused by another stock
        data = make_data({"A": list(a)})
        res = Backtester(data, ScriptedStrategy({1: {0: 10_000.0}})).run(data.calendar[2], data.calendar[-1])
        self.assertEqual((res.fills.side == "SELL").sum(), 0)           # never sold at 500
        self.assertEqual(len(res.frozen), 1)
        units = 10_000 / (10 * 1.001)
        self.assertAlmostEqual(res.daily.nav.iloc[-1], 990_000 + units * 10, places=4)
        self.assertAlmostEqual(res.open_lots.last_mark.iloc[0], 10.0)

    def test_orders_above_signal_cash_raise(self):
        data = self.data
        with self.assertRaises(ValueError):
            Backtester(data, ScriptedStrategy({1: {0: 2_000_000.0}})).run(data.calendar[2], data.calendar[-1])

    def test_strategy_sees_only_signal_close_state(self):
        data = self.data
        strat = ScriptedStrategy({1: {0: 100_000.0}})
        Backtester(data, strat).run(data.calendar[2], data.calendar[50])
        # Decision for session 41 is made at session 40's close: A is already 12 there,
        # but the decision for session 40 (made at 39's close) must still value A at 10.
        nav_at = {t: nav for t, _, nav in strat.seen}
        units = 100_000 / (10 * 1.001)
        self.assertAlmostEqual(nav_at[39], 900_000 + units * 10, places=4)
        self.assertAlmostEqual(nav_at[40], 900_000 + units * 12, places=4)


class LabelAlignment(unittest.TestCase):
    def test_label_matches_backtest_lot_return(self):
        rng = np.random.default_rng(0)
        T = 120
        a = 50 * np.cumprod(1 + rng.normal(0, 0.01, T))
        data = make_data({"A": list(a)})
        target, label_end = labels(data)
        res = Backtester(data, ScriptedStrategy({5: {0: 10_000.0}}), fee_bps=0).run(data.calendar[6], data.calendar[-1])
        buy, sell = res.fills.iloc[0], res.fills.iloc[1]
        self.assertAlmostEqual(target[5, 0], sell.price / buy.price - 1, places=5)
        self.assertEqual(data.calendar[int(label_end[5, 0])], sell.date)


class PriceRepairs(unittest.TestCase):
    CAL = pd.bdate_range("2024-01-02", periods=80)

    def test_unadjusted_split_is_back_adjusted_and_spike_removed(self):
        T = 60
        cal = self.CAL[:T]
        px = np.full((T, 2), 100.0, dtype=np.float32)
        close = px.copy()
        vol = np.full((T, 2), 1e6, dtype=np.float32)
        px[30:, 0] = 25.0; close[30:, 0] = 25.0; vol[30:, 0] = 4e6     # 4:1 split, not adjusted
        px[45, 1] = 1.0; close[45, 1] = 1.0                            # one-day bad print
        out, log = clean_prices(px, close, vol, cal, pd.Index(["S", "Z"]))
        # The split-day quote and the next one are unconfirmed (no quote); the
        # split is confirmed on the second following quote, from then on the
        # history is rescaled so there is no fake -75% return.
        self.assertTrue(np.isnan(out[30:32, 0]).all())
        self.assertTrue(np.allclose(np.delete(out[:, 0], [30, 31]), 25.0))
        self.assertTrue(np.isnan(out[45, 1]))
        self.assertEqual(sorted(log.event), ["spike_removed", "split_adjusted"])

    def test_genuine_crash_with_volume_spike_is_not_a_split(self):
        T = 60
        cal = self.CAL[:T]
        px = np.full((T, 1), 100.0, dtype=np.float32)
        vol = np.full((T, 1), 1e6, dtype=np.float32)
        px[30:] = 45.0                      # -55%: not within 6% of 1/2
        vol[30:] = 8e6
        out, log = clean_prices(px, px.copy(), vol, cal, pd.Index(["C"]))
        self.assertEqual(log.event.tolist(), ["large_move_confirmed"])
        self.assertEqual(out[29, 0], 100.0)                   # history not rescaled
        self.assertEqual(out[32, 0], 45.0)                    # crash accepted once confirmed

    def test_cleaning_is_prefix_invariant(self):
        """Cleaning only the data available up to day k must give the same
        quote mask and the same returns as cleaning the full history (prices
        may differ by a constant factor from a later split rescaling)."""
        T = 80
        raw = np.full((T, 5), 100.0, dtype=np.float32)
        vol = np.full((T, 5), 1e6, dtype=np.float32)
        raw[30:33, 0] = 10.0                          # 3-day fall then recovery
        raw[20:, 1] = 50.0; vol[20:, 1] = 2e6         # 2:1 split, never adjusted
        raw[40, 2] = 400.0                            # one-day spike
        raw[25:, 3] = 40.0; vol[25:, 3] = 9e6         # genuine -60% crash
        raw[50:, 4] = 700.0                           # jump that never confirms cleanly...
        raw[51::2, 4] = 300.0                         # ...because it keeps oscillating
        close = raw.copy()
        close[60:, 2] = raw[60:, 2] / 3               # adjustment-factor break on day 60
        symbols = pd.Index(list("ABCDE"))
        full, _ = clean_prices(raw, close, vol, self.CAL[:T], symbols)
        for k in range(2, T + 1):
            part, _ = clean_prices(raw[:k], close[:k], vol[:k], self.CAL[:k], symbols)
            np.testing.assert_array_equal(np.isnan(part), np.isnan(full[:k]), err_msg=f"mask, prefix {k}")
            for j in range(raw.shape[1]):
                ok = np.flatnonzero(np.isfinite(part[:, j]))
                # Return from every quote to the latest one is identical.
                np.testing.assert_allclose(part[ok, j] / part[ok[-1], j], full[ok, j] / full[ok[-1], j],
                                           rtol=1e-6, err_msg=f"returns, prefix {k}, stock {j}")

    def test_newer_filing_with_missing_field_is_not_backfilled(self):
        from features import fundamental_features
        cal = pd.bdate_range("2024-01-02", periods=200)
        rows = [  # same company, two quarterly filings; the second lacks gross profit
            dict(fiscal_date="2023-12-31", filing_date="2024-02-01", gross_profit=20.0),
            dict(fiscal_date="2024-03-31", filing_date="2024-05-01", gross_profit=np.nan),
        ]
        f = pd.DataFrame(rows).assign(
            symbol="A", period="quarter", revenue=100.0, operating_income=5.0, net_income=4.0,
            operating_cash_flow=6.0, free_cash_flow=3.0, total_assets=100.0, total_debt=10.0,
            total_equity=50.0, cash_and_equivalents=8.0, dividends_paid=np.nan, common_stock_issued=np.nan,
            common_stock_repurchased=np.nan, shares_outstanding=np.nan, long_term_debt=np.nan,
            short_term_debt=np.nan, total_current_assets=np.nan, total_current_liabilities=np.nan,
            interest_expense=np.nan)
        for c in ["fiscal_date", "filing_date"]:
            f[c] = pd.to_datetime(f[c])
        data = make_data({"A": [10.0] * len(cal)}, start=str(cal[0].date()))
        data.fundamentals = f
        out = fundamental_features(data)
        gp = pd.Series(out["gross_profitability"][:, 0], index=data.calendar)
        self.assertAlmostEqual(gp.loc["2024-04-30"], 0.2, places=6)     # first filing known
        self.assertTrue(np.isnan(gp.loc["2024-05-02":]).all())          # newer filing: missing
        age = pd.Series(out["filing_age_days"][:, 0], index=data.calendar)
        self.assertEqual(age.loc["2024-05-02"], 0)                      # age follows the new filing


class IdeaFeatures(unittest.TestCase):
    """The v2 features from my skew research (features.idea_features)."""

    def test_close_location_by_hand(self):
        from data import close_location
        close = [10.0, 12.0, 5.0, 13.0, np.nan]
        high = [12.0, 12.0, 5.0, 12.0, 12.0]
        low = [8.0, 8.0, 5.0, 8.0, 8.0]
        # mid-range, at the high, empty range, close above the high, missing close
        np.testing.assert_array_equal(close_location(close, high, low), [0.5, 1.0, np.nan, np.nan, np.nan])

    def _data(self, T=320, seed=0):
        rng = np.random.default_rng(seed)
        px = 100 * np.exp(np.cumsum(rng.standard_t(3, size=(T, 3)) * 0.02, axis=0))
        px[150:160, 1] = np.nan                                      # a quote gap
        data = make_data({s: px[:, j].tolist() for j, s in enumerate("ABC")})
        data.close_location = rng.uniform(0, 1, size=(T, 3)).astype(np.float32)
        return data

    def test_skew_and_kurtosis_match_scipy(self):
        from scipy import stats
        from features import price_features
        data = self._data()
        f = price_features(data)
        r = np.diff(data.adj_close[:, 0].astype(float)) / data.adj_close[:-1, 0]
        t = len(r)                                                   # last row of the price arrays
        self.assertAlmostEqual(f["skew_63"][t, 0], stats.skew(r[-63:], bias=False), places=4)
        self.assertAlmostEqual(f["kurt_63"][t, 0], stats.kurtosis(r[-63:], bias=False), places=3)
        self.assertAlmostEqual(f["dist_high_20"][t, 0],
                               data.adj_close[t, 0] / data.adj_close[t - 19:t + 1, 0].max() - 1, places=5)
        self.assertAlmostEqual(f["clv_21"][t, 0], data.close_location[t - 20:t + 1, 0].mean(), places=5)

    def test_idea_features_use_only_past_data(self):
        """Features at row t must not change when the data after t is removed."""
        from features import IDEA_FEATURES, price_features
        data = self._data()
        full = price_features(data)
        for k in (70, 155, 260):
            part = self._data()
            part.calendar, part.adj_close = part.calendar[:k], part.adj_close[:k]
            part.dollar_volume, part.close_location = part.dollar_volume[:k], part.close_location[:k]
            cut = price_features(part)
            for name in IDEA_FEATURES:
                np.testing.assert_allclose(cut[name], full[name][:k], rtol=1e-5, atol=1e-6,
                                           equal_nan=True, err_msg=f"{name}, prefix {k}")


def quarterly_filings(dividends_paid, symbol="A", skip=()):
    """One company's quarterly filings; dividends_paid uses the cash-flow sign (paid < 0)."""
    fiscal = [d for i, d in enumerate(pd.date_range("2021-03-31", periods=len(dividends_paid), freq="QE"))
              if i not in skip]
    paid = [v for i, v in enumerate(dividends_paid) if i not in skip]
    f = pd.DataFrame({"fiscal_date": fiscal, "dividends_paid": paid}).assign(
        symbol=symbol, period="quarter", revenue=100.0, gross_profit=40.0, operating_income=10.0,
        net_income=8.0, operating_cash_flow=20.0, free_cash_flow=12.0, total_assets=1000.0, total_debt=100.0,
        total_equity=500.0, cash_and_equivalents=50.0, common_stock_issued=5.0,
        common_stock_repurchased=-15.0, shares_outstanding=100.0, long_term_debt=80.0, short_term_debt=20.0,
        total_current_assets=300.0, total_current_liabilities=150.0, interest_expense=2.0)
    f["filing_date"] = f.fiscal_date + pd.Timedelta(days=30)
    return f


class PayoutFeatures(unittest.TestCase):
    """Dividend and issuance features: TTM sums and the data-hole rule, by hand."""

    def events(self, paid, **kw):
        from features import financial_events
        return financial_events(quarterly_filings(paid, **kw)).reset_index(drop=True)

    def test_hole_is_filled_and_confirmed_omission_counts_as_zero(self):
        nan = np.nan
        e = self.events([-10, -10, -10, -10, -10, nan, -10, -10, nan, nan])
        # Q6 is blank between paying quarters -> a data hole, filled with 10.
        self.assertAlmostEqual(e.div_to_assets[5], 40 / 1000)
        self.assertAlmostEqual(e.div_change_yoy[7], 0.0)
        # Q9 blank: not yet confirmed (could be a hole) -> still 40 TTM.
        self.assertAlmostEqual(e.div_to_assets[8], 40 / 1000)
        # Q10 blank again: Q9 and Q10 are now zeros -> TTM 20 vs 40 a year ago.
        self.assertAlmostEqual(e.div_to_assets[9], 20 / 1000)
        self.assertAlmostEqual(e.div_change_yoy[9], -0.5)

    def test_omission_and_initiation(self):
        nan = np.nan
        stop = self.events([-10, -10, -10, -10, nan, nan, nan, nan])
        self.assertEqual(stop.div_change_yoy.iloc[-1], -1.0)
        self.assertEqual(stop.div_payer.iloc[-1], 0.0)
        start = self.events([0, 0, 0, 0, -5, -5, -5, -5])
        self.assertEqual(start.div_change_yoy.iloc[-1], 1.0)
        self.assertEqual(start.div_payer.iloc[-1], 1.0)
        self.assertAlmostEqual(start.div_payout_cfo.iloc[-1], 20 / 80)     # 4x5 paid / 4x20 CFO

    def test_missing_quarter_breaks_the_ttm(self):
        e = self.events([-10] * 9, skip=(5,))       # the 6th quarter's filing is absent
        # Rows that would need the absent quarter in their four-quarter window are NaN.
        self.assertTrue(np.isnan(e.div_to_assets[5:8]).all())
        self.assertAlmostEqual(e.div_to_assets[4], 40 / 1000)

    def test_issuance_and_buybacks(self):
        e = self.events([-10] * 8)
        self.assertAlmostEqual(e.issued_to_assets.iloc[-1], 20 / 1000)
        self.assertAlmostEqual(e.buyback_to_assets.iloc[-1], 60 / 1000)
        self.assertAlmostEqual(e.net_payout_to_assets.iloc[-1], (40 + 60 - 20) / 1000)
        self.assertAlmostEqual(e.net_issuance_yoy.iloc[-1], 0.0)


class HealthAndLongMomentum(unittest.TestCase):
    def test_health_ratios_by_hand(self):
        from features import financial_events
        f = quarterly_filings([-10] * 2)
        f.loc[1, ["long_term_debt", "short_term_debt", "interest_expense"]] = [0.0, 0.0, 0.0]   # debt-free
        e = financial_events(f).reset_index(drop=True)
        self.assertAlmostEqual(e.operating_margin[0], 10 / 100)
        self.assertAlmostEqual(e.cash_to_ltd[0], 50 / (50 + 80))
        self.assertAlmostEqual(e.net_cash_to_assets[0], (50 - 100) / 1000)
        self.assertAlmostEqual(e.current_ratio[0], 2.0)
        self.assertAlmostEqual(e.interest_coverage[0], 5.0)
        self.assertAlmostEqual(e.st_debt_share[0], 0.2)
        self.assertEqual(e.cash_to_ltd[1], 1.0)            # no long-term debt
        self.assertEqual(e.interest_coverage[1], 50.0)     # no interest to cover: best value
        self.assertEqual(e.st_debt_share[1], 0.0)

    def test_long_momentum_uses_completed_month_ends(self):
        from features import long_momentum
        cal = pd.bdate_range("2020-01-01", "2023-03-31")
        px = 100 * 1.01 ** np.arange(len(cal))
        data = make_data({"A": px.tolist()}, start="2020-01-01")
        out = long_momentum(data, lags=((24, 12),), avg_windows=(252,))
        s = pd.Series(px, index=cal)
        me = s.resample("ME").last()
        t = cal.get_loc(pd.Timestamp("2023-03-15"))           # latest completed month: Feb 2023
        expected = me["2022-02-28"] / me["2021-02-28"] - 1
        self.assertAlmostEqual(out["mom_504_252"][t, 0], expected, places=4)
        self.assertAlmostEqual(out["avg_ret_252"][t, 0], 0.01, places=6)
        # Two years of month-ends are needed; earlier rows stay NaN.
        self.assertTrue(np.isnan(out["mom_504_252"][cal.get_loc(pd.Timestamp("2021-06-15")), 0]))
        # No look-ahead: dropping everything after t leaves row t unchanged.
        part = make_data({"A": px[:t + 1].tolist()}, start="2020-01-01")
        self.assertAlmostEqual(long_momentum(part, lags=((24, 12),))["mom_504_252"][t, 0],
                               out["mom_504_252"][t, 0], places=6)


class Decomposition(unittest.TestCase):
    def test_leave_one_out_peer_return(self):
        from features import peer_returns
        ret = np.array([[0.01, 0.02, 0.03, 0.05]])
        member = np.array([[True, True, True, False]])       # the 4th stock is not a member
        out = peer_returns(ret, member, np.array(["X", "X", "X", "X"]), min_peers=1)
        np.testing.assert_allclose(out[0], [0.025, 0.02, 0.015, 0.02])

    def _data(self, T=330, N=8, seed=1):
        rng = np.random.default_rng(seed)
        rm = rng.normal(0, 0.01, T)
        rets = 1.3 * rm[:, None] + rng.normal(0, 0.02, (T, N))
        px = 100 * np.cumprod(1 + rets, axis=0)
        data = make_data({f"S{j}": px[:, j].tolist() for j in range(N)})
        data.benchmark = pd.Series(100 * np.cumprod(1 + rm), index=data.calendar)
        data.industry = pd.Series("Ind", index=data.symbols)
        return data

    def test_rolling_regression_matches_least_squares(self):
        from features import decomposition_features, peer_returns
        data = self._data()
        out = decomposition_features(data, window=252, resid_skip=21, ind_window=126)
        y = pd.DataFrame(data.adj_close.astype(float)).pct_change().to_numpy()
        rm = data.benchmark.pct_change().to_numpy()
        x2 = peer_returns(y, data.member, data.industry.to_numpy()) - rm[:, None]
        t, j = len(y) - 1, 3
        rows = slice(t - 251, t + 1)
        X = np.column_stack([np.ones(252), rm[rows], x2[rows, j]])
        coef, *_ = np.linalg.lstsq(X, y[rows, j], rcond=None)
        resid = y[rows, j] - X @ coef
        self.assertAlmostEqual(out["beta_mkt_252"][t, j], coef[1], places=4)
        r2 = 1 - (resid ** 2).sum() / ((y[rows, j] - y[rows, j].mean()) ** 2).sum()
        self.assertAlmostEqual(out["r2_252"][t, j], r2, places=4)
        self.assertAlmostEqual(out["idio_vol_252"][t, j], np.sqrt((resid ** 2).sum() / 249 * 252), places=4)
        self.assertAlmostEqual(out["resid_mom_252_21"][t, j], resid[:252 - 21].sum(), places=4)
        self.assertAlmostEqual(out["ind_mom_126"][t, j], x2[t - 125:t + 1, j].sum(), places=4)
        # Squared residuals with sector peers (here every stock is in one sector).
        data.sector = pd.Series("Tech", index=data.symbols)
        sq = decomposition_features(data, peers="sector", resid_sq_windows=(21,))
        xs = peer_returns(y, data.member, data.sector.to_numpy()) - rm[:, None]
        Xs = np.column_stack([np.ones(252), rm[rows], xs[rows, j]])
        cs, *_ = np.linalg.lstsq(Xs, y[rows, j], rcond=None)
        es = y[rows, j] - Xs @ cs
        self.assertAlmostEqual(sq["sec_resid_sq_21"][t, j] / (es[-21:] ** 2).mean(), 1.0, places=4)
        self.assertAlmostEqual(sq["sec_resid_sq_252"][t, j] / (es ** 2).mean(), 1.0, places=4)
        self.assertAlmostEqual(sq["sec_resid_sq_ratio_21_252"][t, j],
                               (es[-21:] ** 2).mean() / (es ** 2).mean(), places=3)

    def test_decomposition_uses_only_past_data(self):
        from features import decomposition_features
        full = decomposition_features(self._data())
        for k in (200, 300):
            part = self._data()
            part.calendar, part.adj_close = part.calendar[:k], part.adj_close[:k]
            part.member, part.benchmark = part.member[:k], part.benchmark.iloc[:k]
            cut = decomposition_features(part)
            for name, a in cut.items():
                np.testing.assert_allclose(a, full[name][:k], rtol=1e-4, atol=1e-6, equal_nan=True,
                                           err_msg=f"{name}, prefix {k}")


class Weighting(unittest.TestCase):
    def test_cohort_weights_by_hand(self):
        from strategy import cohort_weights
        np.testing.assert_allclose(cohort_weights("equal", 4), [0.25] * 4)
        np.testing.assert_allclose(cohort_weights("rank", 4), [0.4, 0.3, 0.2, 0.1])     # 4,3,2,1 / 10
        np.testing.assert_allclose(cohort_weights("inverse_vol", 2, [0.2, 0.4]), [2 / 3, 1 / 3])

    def test_capped_money_is_redistributed_not_left_in_cash(self):
        from strategy import capped_allocation
        # 100 split 50/30/20, but the first name has room for only 20:
        # its extra 30 is re-split 3:2 over the others -> 48 and 32.
        a, b = capped_allocation(100, [0.5, 0.3, 0.2], [20, 1e9, 1e9])
        np.testing.assert_allclose(a, [20, 48, 32])
        # Everyone capped: leftovers go to backups, up to backup_size each.
        a, b = capped_allocation(100, [0.5, 0.5], [30, 30], backup_rooms=[1e9, 5], backup_size=25)
        np.testing.assert_allclose(a, [30, 30])
        np.testing.assert_allclose(b, [25, 5])           # 10 stays in cash: nowhere to put it


class SignalCutoff(unittest.TestCase):
    def test_weak_slots_stay_in_cash(self):
        from types import SimpleNamespace
        from strategy import CohortTopNStrategy
        data = make_data({s: [10.0] * 5 for s in "ABCD"})
        data.sector = pd.Series(["S1", "S2", "S3", "S4"], index=data.symbols)
        shape = data.adj_close.shape
        feats = {"vol_63": np.full(shape, 0.3, np.float32), "adv_63": np.full(shape, 1e9, np.float32)}

        class Fixed:
            def scores(self, t):
                return np.arange(4), np.array([0.70, 0.60, 0.50, 0.40])

        snap = SimpleNamespace(nav=1000.0, cash=1000.0, open_lots=[], date=data.calendar[2], holdings={})
        plain = CohortTopNStrategy(data, feats, Fixed(), top_n=4, n_cohorts=1, name_cap=1.0, sector_cap=1.0)
        self.assertEqual(plain.plan(2, snap), {0: 250.0, 1: 250.0, 2: 250.0, 3: 250.0})
        cut = CohortTopNStrategy(data, feats, Fixed(), top_n=4, n_cohorts=1, name_cap=1.0, sector_cap=1.0,
                                 min_score=0.55, redistribute=True)
        self.assertEqual(cut.plan(2, snap), {0: 250.0, 1: 250.0})     # 500 stays in cash
        none = CohortTopNStrategy(data, feats, Fixed(), top_n=4, n_cohorts=1, name_cap=1.0, sector_cap=1.0,
                                  min_score=0.9)
        self.assertEqual(none.plan(2, snap), {})                      # no signal strong enough


class SectorNeutral(unittest.TestCase):
    def test_best_name_in_each_sector(self):
        from types import SimpleNamespace
        from strategy import CohortTopNStrategy
        data = make_data({s: [10.0] * 5 for s in "ABCDE"})
        data.sector = pd.Series(["Tech", "Tech", "Tech", "Energy", "Energy"], index=data.symbols)
        shape = data.adj_close.shape
        feats = {"vol_63": np.full(shape, 0.3, np.float32), "adv_63": np.full(shape, 1e9, np.float32)}

        class Fixed:
            def scores(self, t):     # all three Tech names outscore both Energy names
                return np.arange(5), np.array([0.9, 0.8, 0.7, 0.6, 0.5])

        snap = SimpleNamespace(nav=1000.0, cash=1000.0, open_lots=[], date=data.calendar[2], holdings={})
        one = CohortTopNStrategy(data, feats, Fixed(), n_cohorts=1, name_cap=1.0, per_sector=1)
        self.assertEqual(one.plan(2, snap), {0: 500.0, 3: 500.0})        # best Tech + best Energy
        two = CohortTopNStrategy(data, feats, Fixed(), n_cohorts=1, name_cap=1.0, per_sector=2)
        self.assertEqual(sorted(two.plan(2, snap)), [0, 1, 3, 4])


class IndustryRotation(unittest.TestCase):
    def test_buys_top_stocks_of_best_industry(self):
        from types import SimpleNamespace
        from strategy import CohortTopNStrategy
        T = 70
        prices = {}
        for k in range(4):                      # "Hot" industry: +20% over the lookback
            prices[f"H{k}"] = np.linspace(100, 120, T).tolist()
        for k in range(4):                      # "Cold" industry: flat
            prices[f"C{k}"] = [100.0] * T
        data = make_data(prices)
        data.industry = pd.Series({sym: "Hot" if sym.startswith("H") else "Cold" for sym in prices})
        data.sector = pd.Series("S", index=data.symbols)
        shape = data.adj_close.shape
        feats = {"vol_63": np.full(shape, 0.3, np.float32), "adv_63": np.full(shape, 1e9, np.float32)}
        cols_hot = [data.symbols.get_loc(f"H{k}") for k in range(4)]
        cols_cold = [data.symbols.get_loc(f"C{k}") for k in range(4)]

        class Fixed:                            # the model prefers the cold stocks
            def scores(self, t):
                sc = np.zeros(8)
                sc[cols_cold] = [0.9, 0.8, 0.7, 0.6]
                sc[cols_hot] = [0.4, 0.3, 0.2, 0.1]
                return np.arange(8), sc

        snap = SimpleNamespace(nav=1000.0, cash=1000.0, open_lots=[], date=data.calendar[-1], holdings={})
        s = CohortTopNStrategy(data, feats, Fixed(), n_cohorts=1, name_cap=1.0, sector_cap=1.0,
                               rotation={"n_industries": 1, "lookback": 63, "per_industry": 2, "min_members": 3})
        orders = s.plan(T - 1, snap)
        self.assertEqual(sorted(orders), sorted(cols_hot[:2]))          # best 2 Hot stocks by score
        self.assertEqual(s.log[-1]["industries"], "Hot")


class CorrelationCap(unittest.TestCase):
    def test_skips_a_stock_that_moves_with_the_portfolio(self):
        from types import SimpleNamespace
        from strategy import CohortTopNStrategy
        rng = np.random.default_rng(0)
        T = 260
        a = 100 * np.cumprod(1 + rng.normal(0, 0.02, T))
        b = a * 1.5                                                   # same returns as A
        c = 100 * np.cumprod(1 + rng.normal(0, 0.02, T))              # independent
        data = make_data({"A": a.tolist(), "B": b.tolist(), "C": c.tolist()})
        data.sector = pd.Series(["S1", "S2", "S3"], index=data.symbols)
        shape = data.adj_close.shape
        feats = {"vol_63": np.full(shape, 0.3, np.float32), "adv_63": np.full(shape, 1e9, np.float32)}

        class Fixed:
            def scores(self, t):
                return np.arange(3), np.array([0.9, 0.8, 0.7])          # A > B > C

        snap = SimpleNamespace(nav=1000.0, cash=1000.0, open_lots=[], date=data.calendar[-1], holdings={})
        s = CohortTopNStrategy(data, feats, Fixed(), top_n=2, n_cohorts=1, name_cap=1.0, sector_cap=1.0,
                               corr_cap=0.5, corr_min_obs=100)
        self.assertEqual(sorted(s.plan(T - 1, snap)), [0, 2])           # B rejected: corr 1 with A
        # A stock already held counts as part of the portfolio.
        held = SimpleNamespace(nav=1000.0, cash=500.0, open_lots=[], date=data.calendar[-1], holdings={0: 500.0})
        s2 = CohortTopNStrategy(data, feats, Fixed(), top_n=2, n_cohorts=1, name_cap=1.0, sector_cap=1.0,
                                corr_cap=0.5, corr_min_obs=100)
        self.assertNotIn(1, s2.plan(T - 1, held))


class DuplicateTickers(unittest.TestCase):
    def test_second_ticker_of_same_company_is_flagged_point_in_time(self):
        from data import duplicate_copies
        rng = np.random.default_rng(3)
        T = 300
        a = 100 * np.cumprod(1 + rng.normal(0, 0.02, T))
        b = np.full(T, np.nan); b[150:] = a[150:] * 0.5          # new ticker, same company, from day 150
        c = 100 * np.cumprod(1 + rng.normal(0, 0.02, T))          # unrelated
        px = np.column_stack([a, b, c]).astype(np.float32)
        copy, pairs = duplicate_copies(px, pd.Index(["OLD", "NEW", "OTH"]), min_hits=10)
        self.assertEqual(pairs, [(0, 1)])
        self.assertFalse(copy[:, 0].any() or copy[:, 2].any())    # the older ticker is kept
        self.assertFalse(copy[:170, 1].any())                     # not enough shared history yet
        self.assertTrue(copy[200:, 1].all())                      # flagged once 63 days match
        # Point in time: flags up to day k do not change when later data is removed.
        for k in (180, 240):
            part, _ = duplicate_copies(px[:k], pd.Index(["OLD", "NEW", "OTH"]), min_hits=10)
            np.testing.assert_array_equal(part, copy[:k])


class PanelColumns(unittest.TestCase):
    def test_panel_features_are_unique(self):
        from features import PANEL_FEATURES
        self.assertEqual(len(PANEL_FEATURES), len(set(PANEL_FEATURES)))


class Agreement(unittest.TestCase):
    def test_agreement_by_hand(self):
        from model import RidgeOnRanks
        m = RidgeOnRanks()
        m.model.coef_ = np.array([0.2, -0.1, 0.0, 0.0])        # 2 features + 2 missing indicators
        m.model.intercept_ = 0.5
        X = np.array([[0.9, 0.1],      # +0.08 and +0.04: full agreement
                      [0.9, 0.9]])     # +0.08 and -0.04: 2/3 agree
        c = m.contributions(X)
        np.testing.assert_allclose(c, [[0.08, 0.04], [0.08, -0.04]])
        agree = np.clip(c, 0, None).sum(1) / np.abs(c).sum(1)
        np.testing.assert_allclose(agree, [1.0, 2 / 3])
        np.testing.assert_allclose(m.predict(X), 0.5 + c.sum(1))   # contributions add up to the score


class Metrics(unittest.TestCase):
    def test_cagr_and_drawdown(self):
        idx = pd.to_datetime(["2020-01-01", "2020-07-01", "2021-01-01"])
        nav = pd.Series([100.0, 80.0, 121.0], index=idx)
        m = performance(nav)
        self.assertAlmostEqual(m.max_drawdown, -0.2)
        years = (idx[-1] - idx[0]).days / 365.25
        self.assertAlmostEqual(m.CAGR, 1.21 ** (1 / years) - 1)


if __name__ == "__main__":
    unittest.main()
