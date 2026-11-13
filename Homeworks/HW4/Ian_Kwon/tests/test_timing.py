"""Information-timing tests on the real database (research mode only)."""

import unittest

import numpy as np
import pandas as pd

from quantlab import features as F
from quantlab import integrity as I
from quantlab import panel as P
from quantlab.data import DEFAULT_DB, load_market, load_reference
from quantlab.leakage import prefix_invariance
from quantlab.portfolio import ScoreBook
from quantlab.scoring import LeakageError, assert_matured, training_mask

HAVE_DB = DEFAULT_DB.is_file()


@unittest.skipUnless(HAVE_DB, "needs equities_top2000.duckdb")
class RealData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = load_market(verify=False)
        cls.r = load_reference()
        cls.audit = I.audit_prices(cls.m.adj, cls.m.close, cls.m.volume, cls.m.calendar, cls.m.symbols)
        cls.panel = P.build_panel(cls.m, cls.r, cls.audit, end="2015-12-31")
        cls.cols = [c for c in cls.panel.columns if c.endswith("_csr")] + F.PRICE_FEATURES

    def test_prefix_invariance_of_all_model_inputs(self):
        report = prefix_invariance(self.m, self.r, "2013-06-28", self.cols, panel_end="2015-12-31")
        bad = report.loc[report.max_abs_diff > 1e-9]
        self.assertTrue(bad.empty, f"future data changed past features:\n{bad}")

    def test_prefix_harness_catches_an_intentionally_leaked_feature(self):
        original = F.price_feature_matrices

        def leaky(px, close, volume):
            out = original(px, close, volume)
            future = np.full_like(px, np.nan)
            future[:-5] = px[5:]
            out["ret_21"] = future / px - 1  # uses the price 5 sessions AHEAD
            return out
        P.F.price_feature_matrices = leaky
        try:
            report = prefix_invariance(self.m, self.r, "2013-06-28", ["ret_21"], panel_end="2015-12-31")
        finally:
            P.F.price_feature_matrices = original
        self.assertGreater(report.set_index("column").loc["ret_21", "max_abs_diff"], 0)

    def test_label_matches_hand_calculation(self):
        row = self.panel.loc[self.panel.symbol.eq("AAPL") & self.panel.date.eq("2012-08-31")].iloc[0]
        cal, s = self.m.calendar, self.m.symbols.get_loc("AAPL")
        entry = cal[cal.get_loc(row.date) + 1]
        exit_ = cal[cal.searchsorted(entry + pd.Timedelta(days=90))]
        px = self.audit.clean[:, s]
        self.assertEqual(entry, pd.Timestamp("2012-09-04"))          # same worked example as Lecture 7.2
        self.assertEqual(exit_, pd.Timestamp("2012-12-03"))
        self.assertEqual(row.label_end, exit_)
        self.assertAlmostEqual(row.target, px[cal.get_loc(exit_)] / px[cal.get_loc(entry)] - 1, places=12)

    def test_financial_statement_used_only_after_filing_plus_one_day(self):
        # Lecture 7.2 example: AAPL quarter ending 2013-12-28 filed 2014-01-28 -> usable from 2014-01-29
        ev = F.financial_events(self.r.fundamentals)
        probe = pd.DataFrame({"symbol": "AAPL", "date": pd.to_datetime(["2014-01-28", "2014-01-29"]),
                              "log_mcap_proxy": np.log(5e11)})
        j = F.join_financials(probe, ev)
        self.assertLess(j.fiscal_date.iloc[0], pd.Timestamp("2013-12-28"))
        self.assertEqual(j.fiscal_date.iloc[1], pd.Timestamp("2013-12-28"))
        self.assertTrue((self.panel.available_date.isna() | (self.panel.available_date <= self.panel.date)).all())

    def test_trend_features_match_hand_calculation(self):
        # AAPL at 2014-08-29 uses the quarter ending 2014-06-28 (filed 2014-07-23) against 2013-06-29.
        f = self.r.fundamentals
        q = f.loc[f.symbol.eq("AAPL") & f.period.eq("quarter")].set_index("fiscal_date")
        now, ago = q.loc["2014-06-28"], q.loc["2013-06-29"]
        row = self.panel.loc[self.panel.symbol.eq("AAPL") & self.panel.date.eq("2014-08-29")].iloc[0]
        self.assertEqual(row.fiscal_date, pd.Timestamp("2014-06-28"))
        self.assertAlmostEqual(row.revenue_growth_yoy, now.revenue / ago.revenue - 1, places=12)
        self.assertAlmostEqual(row.asset_growth_yoy, now.total_assets / ago.total_assets - 1, places=12)
        self.assertAlmostEqual(row.net_issuance_yoy, np.log(now.shares_outstanding / ago.shares_outstanding), places=12)
        self.assertAlmostEqual(row.roa_yoy_change, now.operating_income / now.total_assets
                               - ago.operating_income / ago.total_assets, places=12)
        ttm = q.loc["2013-09-28":"2014-06-28"].net_income
        self.assertEqual(len(ttm), 4)
        self.assertAlmostEqual(row.earnings_yield_ttm, ttm.sum() / np.exp(row.log_mcap_proxy), places=9)

    def test_placeholder_filings_get_conservative_availability(self):
        ev = F.financial_events(self.r.fundamentals)
        imp = ev.loc[ev.availability_imputed.eq(1)]
        self.assertGreater(len(imp), 0)
        self.assertTrue((imp.available_date == imp.fiscal_date + pd.Timedelta(days=120)).all())

    def test_membership_is_entry_quarter_and_known_at_signal(self):
        p = self.panel
        self.assertTrue((p.quarter_start == p.entry_date.dt.to_period("Q").dt.start_time).all())
        self.assertTrue((p.selection_date <= p.date).all())
        # June 30 signal -> July 1 entry uses Q3 membership (Lecture 7.2 / data guide example)
        q = p.loc[p.date.eq("2014-06-30"), "quarter_start"].unique()
        self.assertEqual(list(q), [pd.Timestamp("2014-07-01")])

    def test_sector_peers_use_the_full_eligible_cross_section(self):
        p = self.panel
        day = p.loc[p.date.eq("2014-06-30") & p.eligible]
        tech = day.loc[day.sector.eq("Technology")]
        expected = tech.ret_63 - tech.ret_63.median()
        np.testing.assert_allclose(tech.ret_63_sr, expected, atol=1e-6)
        self.assertTrue(p.loc[~p.eligible, "ret_63_sr"].isna().all())

    def test_training_rows_are_matured_at_every_fit(self):
        F_ = pd.Timestamp("2014-01-03")
        tr = self.panel.loc[training_mask(self.panel, F_, "2010-01-01", "target_rank")]
        assert_matured(tr, F_)
        self.assertLess(tr.label_end.max(), F_)
        leaky = self.panel.loc[self.panel.date.lt(F_) & self.panel.label_end.ge(F_)]
        with self.assertRaises(LeakageError):
            assert_matured(leaky, F_)


class ScoreBookTiming(unittest.TestCase):
    def frame(self, **kw):
        base = {"date": ["2015-01-02"], "symbol": ["A"], "score": [1.0], "fit_date": ["2015-01-02"],
                "train_max_label_end": ["2014-12-30"]}
        base.update(kw)
        return pd.DataFrame(base)

    def test_accepts_valid_timing(self):
        ScoreBook(self.frame())

    def test_rejects_model_fitted_after_signal(self):
        with self.assertRaises(ValueError):
            ScoreBook(self.frame(fit_date=["2015-01-05"]))

    def test_rejects_training_target_not_complete_at_fit(self):
        with self.assertRaises(ValueError):
            ScoreBook(self.frame(train_max_label_end=["2015-01-02"]))


if __name__ == "__main__":
    unittest.main()
