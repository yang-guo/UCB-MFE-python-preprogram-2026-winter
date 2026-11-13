"""Unit tests for hw4_features.py (features A, B, D and their timing).

Run from the repo root:
    uv run python -m unittest discover -s Homeworks/HW4/Antonio_Rivera -v
"""

import unittest

import numpy as np
import pandas as pd

import hw4_data as hd
import hw4_features as hf

T = pd.Timestamp


def toy_fund(rows, extra=None):
    """rows: (fiscal_date, filing_date, gross_profit, total_assets) for symbol AAA.

    extra: optional {column: list of values} for the other fundamentals; any
    column not given is NaN (so module-2 tests ignore module-4a fields).
    """
    f = pd.DataFrame(rows, columns=["fiscal_date", "filing_date", "gross_profit", "total_assets"])
    f.insert(0, "symbol", "AAA")
    f["period"] = "quarter"
    f[["fiscal_date", "filing_date"]] = f[["fiscal_date", "filing_date"]].apply(pd.to_datetime)
    for col in hf.FUND_COLUMNS:
        if col not in f:
            f[col] = (extra or {}).get(col, np.nan)
    return f


def quarters(n, start="2019-03-31", lag_days=40):
    """n consecutive quarter-ends with filings `lag_days` later; GP/assets constant."""
    ends = pd.date_range(start, periods=n, freq="QE")
    return [(d.strftime("%Y-%m-%d"), (d + pd.Timedelta(days=lag_days)).strftime("%Y-%m-%d"),
             10.0, 100.0) for d in ends]


# Five clean quarters: assets 100 -> 150 over one year, gross profit always 10.
CLEAN = [
    ("2022-03-31", "2022-05-10", 10.0, 100.0),
    ("2022-06-30", "2022-08-09", 10.0, 110.0),
    ("2022-09-30", "2022-11-09", 10.0, 120.0),
    ("2022-12-31", "2023-02-09", 10.0, 130.0),
    ("2023-03-31", "2023-05-10", 10.0, 150.0),
]


def event(ev, fiscal):
    return ev.loc[ev.fiscal_date.eq(T(fiscal))].iloc[0]


class TestFinancialEvents(unittest.TestCase):
    """A and B computed per filing, by hand."""

    def test_gross_profitability_and_asset_growth(self):
        ev = hf.extra_financial_events(toy_fund(CLEAN))
        e = event(ev, "2023-03-31")
        self.assertAlmostEqual(e.gross_profitability, 10 / 150)
        self.assertAlmostEqual(e.asset_growth, 150 / 100 - 1)  # 0.50
        self.assertEqual(e.available_date, T("2023-05-11"))     # filing + 1 day

    def test_no_year_ago_quarter_means_missing_growth(self):
        ev = hf.extra_financial_events(toy_fund(CLEAN))
        self.assertTrue(np.isnan(event(ev, "2022-12-31").asset_growth))

    def test_52_53_week_fiscal_drift_still_matches(self):
        # Year-ago fiscal end 2022-03-26 is 6 days from 2023-04-01 - 365d: within ±20.
        rows = [("2022-03-26", "2022-05-01", 10.0, 100.0),
                ("2023-04-01", "2023-05-01", 10.0, 120.0)]
        ev = hf.extra_financial_events(toy_fund(rows))
        self.assertAlmostEqual(event(ev, "2023-04-01").asset_growth, 0.20)

    def test_year_ago_published_later_is_not_used(self):
        # Pathological: the year-ago record is filed AFTER the current one.
        rows = [("2022-03-31", "2023-06-01", 10.0, 100.0),
                ("2023-03-31", "2023-05-10", 10.0, 150.0)]
        ev = hf.extra_financial_events(toy_fund(rows))
        self.assertTrue(np.isnan(event(ev, "2023-03-31").asset_growth))

    def test_late_old_filing_does_not_replace_newer(self):
        # Fiscal 2022-06-30 refiled on 2023-06-01, after 2023-03-31 was known.
        rows = CLEAN + [("2022-06-30", "2023-06-01", 99.0, 999.0)]
        rows = [r for r in rows if not (r[0] == "2022-06-30" and r[1] == "2022-08-09")]
        ev = hf.extra_financial_events(toy_fund(rows))
        self.assertFalse(ev.available_date.eq(T("2023-06-02")).any())

    def test_invalid_filing_dates_excluded(self):
        rows = CLEAN + [("2023-06-30", "2023-06-30", 10.0, 160.0)]  # filing == fiscal end
        ev = hf.extra_financial_events(toy_fund(rows))
        self.assertFalse(ev.fiscal_date.eq(T("2023-06-30")).any())

    def test_implausible_values_blanked_and_flagged(self):
        rows = [("2022-03-31", "2022-05-10", 10.0, 100.0),
                ("2023-03-31", "2023-05-10", 500.0, 100_000.0)]  # assets x1000: unit error
        ev = hf.extra_financial_events(toy_fund(rows))
        e = event(ev, "2023-03-31")
        self.assertTrue(np.isnan(e.asset_growth))      # 999 is outside (-0.9, 9)
        self.assertTrue(e.asset_growth_blanked)
        self.assertAlmostEqual(e.gross_profitability, 500 / 100_000)  # 0.005 is plausible
        bad = hf.extra_financial_events(toy_fund([("2022-03-31", "2022-05-10", 500.0, 100.0)]))
        self.assertTrue(np.isnan(bad.gross_profitability.iloc[0]))  # 5.0 > 2.0


class TestJoinTiming(unittest.TestCase):
    """The as-of join must respect filing_date + 1 day and the 550-day staleness."""

    @classmethod
    def setUpClass(cls):
        ev = hf.extra_financial_events(toy_fund(CLEAN))
        dates = pd.to_datetime(["2023-05-10", "2023-05-11", "2024-10-01", "2024-10-03"])
        panel = pd.DataFrame({"symbol": "AAA", "date": dates})
        cls.out = hf.join_extra_financials(panel, ev).set_index("date")

    def test_not_usable_on_filing_date(self):
        # On 2023-05-10 the 2023-03-31 filing is not yet usable: previous quarter's value.
        self.assertAlmostEqual(self.out.loc[T("2023-05-10"), "gross_profitability"], 10 / 130)

    def test_usable_from_next_day(self):
        self.assertAlmostEqual(self.out.loc[T("2023-05-11"), "gross_profitability"], 10 / 150)

    def test_stale_after_550_days(self):
        # fiscal 2023-03-31 + 550 days = 2024-10-02.
        self.assertFalse(np.isnan(self.out.loc[T("2024-10-01"), "gross_profitability"]))
        self.assertTrue(np.isnan(self.out.loc[T("2024-10-03"), "gross_profitability"]))


class TestPeerRanks(unittest.TestCase):
    """D: ranks use only eligible names on the same date."""

    def panel(self):
        return pd.DataFrame({
            "date": [T("2024-01-02")] * 4 + [T("2024-01-03")] * 2,
            "symbol": ["A", "B", "C", "D", "A", "B"],
            "eligible": [True, True, True, False, True, True],
            "x": [3.0, 1.0, 2.0, 100.0, 5.0, 6.0],
        })

    def test_ranks_by_hand(self):
        r = hf.add_peer_ranks(self.panel(), ["x"]).x_rank.tolist()
        self.assertEqual(r[:3], [1.0, 1 / 3, 2 / 3])
        self.assertTrue(np.isnan(r[3]))           # ineligible: no rank
        self.assertEqual(r[4:], [0.5, 1.0])

    def test_ineligible_and_other_dates_do_not_leak(self):
        base = hf.add_peer_ranks(self.panel(), ["x"]).x_rank
        p = self.panel()
        p.loc[3, "x"] = -1e9                      # ineligible name changes
        p.loc[4, "x"] = 1e9                       # a different date changes
        moved = hf.add_peer_ranks(p, ["x"]).x_rank
        pd.testing.assert_series_equal(base[:3], moved[:3])

    def test_missing_value_stays_missing(self):
        p = self.panel()
        p.loc[0, "x"] = np.nan
        r = hf.add_peer_ranks(p, ["x"]).x_rank
        self.assertTrue(np.isnan(r[0]))
        self.assertEqual(r[1:3].tolist(), [0.5, 1.0])


class TestSUE(unittest.TestCase):
    """EPS seasonal change d_k = eps_k - eps_{k-4}; SUE_k = d_k / std(previous d's)."""

    @staticmethod
    def events(eps):
        return hf.extra_financial_events(toy_fund(quarters(len(eps)), {"eps": eps}))

    def test_sue_by_hand(self):
        # eps 0,0,0,0 then d = +1,-1,+1,-1 (k=4..7), then d = +3 at k=8.
        eps = [0, 0, 0, 0, 1, -1, 1, -1, 4]
        ev = self.events(eps)
        sigma = np.std([1, -1, 1, -1], ddof=1)          # 1.1547
        self.assertAlmostEqual(ev.sue.iloc[8], 3 / sigma)  # 2.598

    def test_needs_four_prior_changes(self):
        ev = self.events([0, 0, 0, 0, 1, -1, 1, -1, 4])
        self.assertTrue(ev.sue.iloc[:8].isna().all())     # k=7 has only 3 prior d's

    def test_zero_sigma_gives_missing_not_infinite(self):
        ev = self.events([0, 0, 0, 0, 1, 1, 1, 1, 2])     # previous d's all equal 1
        self.assertTrue(np.isnan(ev.sue.iloc[8]))


class TestTTM(unittest.TestCase):

    def test_ttm_sum_by_hand(self):
        ni = [1.0, 2.0, 3.0, 4.0, 5.0]
        ev = hf.extra_financial_events(toy_fund(quarters(5), {"net_income": ni}))
        self.assertTrue(ev.ttm_net_income.iloc[:3].isna().all())  # fewer than 4 quarters
        self.assertEqual(ev.ttm_net_income.iloc[3], 10.0)         # 1+2+3+4
        self.assertEqual(ev.ttm_net_income.iloc[4], 14.0)         # 2+3+4+5

    def test_missing_quarter_breaks_ttm(self):
        rows = quarters(6)
        del rows[2]                                              # gap: span becomes ~1 year
        ev = hf.extra_financial_events(toy_fund(rows, {"net_income": [1.0] * 5}))
        self.assertTrue(ev.ttm_net_income.iloc[3:5].isna().all())

    def test_any_missing_value_breaks_ttm(self):
        ev = hf.extra_financial_events(toy_fund(quarters(4), {"revenue": [1.0, np.nan, 1.0, 1.0]}))
        self.assertTrue(np.isnan(ev.ttm_revenue.iloc[3]))

    def test_older_quarter_filed_later_breaks_ttm(self):
        rows = quarters(4)
        rows[2] = (rows[2][0], "2020-06-01", 10.0, 100.0)        # Q3 filed after Q4
        ev = hf.extra_financial_events(toy_fund(rows, {"net_income": [1.0] * 4}))
        q4 = ev.loc[ev.fiscal_date.eq(T(rows[3][0]))]
        self.assertTrue(q4.ttm_net_income.isna().all())


class TestValueFeatures(unittest.TestCase):

    def frame(self, **kw):
        base = dict(close=10.0, shares_outstanding=100.0, total_equity=500.0, total_debt=200.0,
                    cash_and_equivalents=100.0, ttm_net_income=50.0, ttm_ocf=80.0,
                    ttm_ebitda=110.0, ttm_revenue=1000.0)
        base.update(kw)
        return pd.DataFrame([base])

    def test_ratios_by_hand(self):
        r = hf.add_value_features(self.frame()).iloc[0]
        self.assertEqual(r.mcap, 1000.0)                   # 10 x 100
        self.assertAlmostEqual(r.book_to_market, 0.5)      # 500 / 1000
        self.assertAlmostEqual(r.earnings_yield, 0.05)     # 50 / 1000
        self.assertAlmostEqual(r.cashflow_yield, 0.08)
        self.assertAlmostEqual(r.ebitda_to_ev, 110 / 1100)  # EV = 1000 + 200 - 100
        self.assertAlmostEqual(r.sales_to_price, 1.0)
        self.assertFalse(r.mcap_suspect)

    def test_negative_book_is_missing_but_not_suspect(self):
        r = hf.add_value_features(self.frame(total_equity=-5.0)).iloc[0]
        self.assertTrue(np.isnan(r.book_to_market))
        self.assertFalse(r.mcap_suspect)
        self.assertAlmostEqual(r.earnings_yield, 0.05)

    def test_inflated_mcap_blanks_whole_row(self):
        # Close restated for a reverse split: mcap x1000 -> P/S = 1000 > 100.
        r = hf.add_value_features(self.frame(close=10_000.0)).iloc[0]
        self.assertTrue(r.sales_to_price_blanked and r.mcap_suspect)
        self.assertTrue(np.isnan([r.mcap] + [r[c] for c in hf.VALUE_METRICS]).all())

    def test_deflated_mcap_blanks_whole_row(self):
        # Assets/earnings in the wrong units: E/P = 50 > 1.
        r = hf.add_value_features(self.frame(close=0.01)).iloc[0]
        self.assertTrue(r.mcap_suspect)
        self.assertTrue(np.isnan(r.book_to_market))


class TestValueComposite(unittest.TestCase):

    def test_composite_by_hand(self):
        # Two eligible names; A is cheaper on every metric except V5 (missing for B).
        p = pd.DataFrame({"date": [T("2024-01-02")] * 3, "symbol": ["A", "B", "C"],
                          "eligible": [True, True, False],
                          "book_to_market": [0.8, 0.2, 9.0], "earnings_yield": [0.1, 0.05, 9.0],
                          "cashflow_yield": [0.2, 0.1, 9.0], "ebitda_to_ev": [0.1, 0.2, 9.0],
                          "sales_to_price": [1.0, np.nan, 9.0]})
        out = hf.add_value_composite(p)
        # A ranks: 1, 1, 1, 0.5, 1 -> mean 0.9 ; B ranks: 0.5, 0.5, 0.5, 1 -> mean 0.625
        self.assertAlmostEqual(out.value_composite.iloc[0], 0.9)
        self.assertAlmostEqual(out.value_composite.iloc[1], 0.625)
        self.assertTrue(np.isnan(out.value_composite.iloc[2]))  # ineligible

    def test_needs_three_metrics(self):
        p = pd.DataFrame({"date": [T("2024-01-02")] * 2, "symbol": ["A", "B"],
                          "eligible": [True, True], "book_to_market": [0.8, 0.2],
                          "earnings_yield": [0.1, 0.05], "cashflow_yield": [np.nan, np.nan],
                          "ebitda_to_ev": [np.nan, np.nan], "sales_to_price": [np.nan, np.nan]})
        self.assertTrue(hf.add_value_composite(p).value_composite.isna().all())


class TestPriceBlock(unittest.TestCase):
    """beta / idio vol / MAX vs an independent numpy OLS on the same window."""

    N = 300

    def setUp(self):
        rng = np.random.default_rng(0)
        self.cal = pd.bdate_range("2020-01-01", periods=self.N)
        self.mkt = pd.Series(rng.normal(0, 0.01, self.N), index=self.cal, name="mkt")
        self.mkt.iloc[0] = np.nan                          # first day has no return
        self.noise = rng.normal(0, 0.02, self.N)

    def prices_from_returns(self, r, sym="S"):
        px = 100 * np.cumprod(1 + np.nan_to_num(r))
        return pd.DataFrame({"symbol": sym, "date": self.cal, "adj_close": px})

    def block(self, r):
        return hf.price_block(self.prices_from_returns(r), self.mkt).set_index("date")

    def test_pure_multiple_of_market(self):
        out = self.block(2.0 * self.mkt.to_numpy())
        self.assertAlmostEqual(out.beta_252.iloc[-1], 2.0, places=6)
        self.assertAlmostEqual(out.idio_vol_63.iloc[-1], 0.0, places=6)

    def test_matches_numpy_ols(self):
        r = 1.5 * self.mkt.to_numpy() + self.noise
        out = self.block(r)
        rs, ms = pd.Series(r, index=self.cal).iloc[-252:], self.mkt.iloc[-252:]
        slope = np.polyfit(ms, rs, 1)[0]
        self.assertAlmostEqual(out.beta_252.iloc[-1], slope, places=8)
        y, x = rs.iloc[-63:], ms.iloc[-63:]
        resid = y - np.polyval(np.polyfit(x, y, 1), x)
        expected = np.sqrt((resid ** 2).sum() / (63 - 1) * 252)
        self.assertAlmostEqual(out.idio_vol_63.iloc[-1], expected, places=8)

    def test_max_ret_by_hand(self):
        r = np.zeros(self.N)
        r[100] = 0.10                                       # one +10% day
        out = self.block(r)
        self.assertAlmostEqual(out.max_ret_21.iloc[100], 0.10)
        self.assertAlmostEqual(out.max_ret_21.iloc[120], 0.10)  # still inside 21 sessions
        self.assertAlmostEqual(out.max_ret_21.iloc[121], 0.0)   # dropped out

    def test_causal_future_prices_do_not_change_past(self):
        r = 1.2 * self.mkt.to_numpy() + self.noise
        base = self.block(r)
        r2 = r.copy()
        r2[260:] = 0.5                                      # wild future returns
        moved = self.block(r2)
        pd.testing.assert_frame_equal(base.iloc[:260], moved.iloc[:260])

    def test_price_gap_not_filled(self):
        prices = self.prices_from_returns(1.0 * self.mkt.to_numpy())
        prices = prices.drop(index=range(280, 290))         # 10 missing sessions
        out = hf.price_block(prices, self.mkt).set_index("date")
        self.assertNotIn(self.cal[285], out.index)          # no row on a gap day
        self.assertTrue(np.isnan(out.max_ret_21.loc[self.cal[295]]))  # window spans the gap
        self.assertTrue(np.isnan(out.idio_vol_63.loc[self.cal[295]]))
        self.assertAlmostEqual(out.beta_252.loc[self.cal[295]], 1.0, places=6)  # >= 189 days left

    def test_beta_needs_189_days(self):
        out = self.block(1.0 * self.mkt.to_numpy())
        self.assertTrue(np.isnan(out.beta_252.iloc[188]))   # 188 returns so far
        self.assertFalse(np.isnan(out.beta_252.iloc[189]))


@unittest.skipUnless(hd.CLASS_DB.is_file(), "class database not downloaded")
class TestClassDatabase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.panel, _, _ = hf.build_feature_panel(hd.CLASS_DB, end="2015-12-31")

    def test_no_future_filing_used(self):
        p = self.panel
        self.assertTrue((p.x_available_date.isna() | p.x_available_date.le(p.date)).all())

    def test_same_filing_as_course_features(self):
        # Our join must pick the same filing the course baseline picked.
        p = self.panel.dropna(subset=["fiscal_date", "x_fiscal_date"])
        self.assertTrue(p.fiscal_date.eq(p.x_fiscal_date).all())

    def test_ranks_in_unit_interval_and_eligible_only(self):
        p = self.panel
        ranks = p[hf.RANK_FEATURES]
        self.assertTrue(ranks[~p.eligible].isna().all().all())
        vals = ranks[p.eligible].stack()
        self.assertTrue(((vals > 0) & (vals <= 1)).all())

    def test_aapl_mcap_matches_reality(self):
        # AAPL on 2015-06-30 was worth roughly $720B; close x restated shares must agree.
        r = self.panel.loc[self.panel.symbol.eq("AAPL") & self.panel.date.eq(T("2015-06-30"))].iloc[0]
        self.assertTrue(650e9 < r.mcap < 800e9)
        self.assertFalse(r.mcap_suspect)

    def test_known_basis_errors_are_flagged(self):
        e = self.panel.loc[self.panel.eligible]
        share = e.groupby("symbol").mcap_suspect.mean()
        for sym in ["APDN", "BKNG", "PEGY", "BESS"]:
            self.assertGreater(share[sym], 0.8, sym)
        for sym in ["AAPL", "MSFT", "XOM", "JNJ"]:
            self.assertEqual(share[sym], 0.0, sym)

    def test_betas_are_economically_sensible(self):
        med = self.panel.loc[self.panel.eligible].groupby("symbol").beta_252.median()
        for sym in ["JNJ", "KO", "PG", "WMT"]:      # defensives
            self.assertLess(med[sym], 0.8, sym)
        for sym in ["FCX", "JPM"]:                  # cyclicals
            self.assertGreater(med[sym], 1.2, sym)

    def test_module1_audit_still_passes(self):
        # Adding features must not disturb labels.
        _, prices, calendar = hd.build_panel(hd.CLASS_DB)
        hd.audit_label_timing(self.panel, prices, calendar)


if __name__ == "__main__":
    unittest.main()
