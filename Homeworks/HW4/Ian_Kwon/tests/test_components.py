"""Unit tests: holdout protocol, price integrity, portfolio construction, metrics."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from quantlab import features as F
from quantlab import integrity as I
from quantlab import metrics as M
from quantlab import protocol
from quantlab.backtest import Backtester, OrderManagementSystem, PortfolioSnapshot
from quantlab.data import DEFAULT_DB, load_market
from quantlab.models import TimeTunedRidge, build_model
from quantlab.portfolio import (CohortBudget, CompositeStrategy, EqualWeight, InverseVolatility, NameCap,
                                RankWeight, ScoreBook, SectorCap, TopN, Weekly)
from quantlab.protocol import HoldoutViolation, Mode


class Protocol(unittest.TestCase):
    def test_dates_match_the_assignment(self):
        self.assertEqual(protocol.PROTOCOL["development_start"], "2010-01-01")
        self.assertEqual(protocol.PROTOCOL["development_end"], "2015-12-31")
        self.assertEqual(protocol.PROTOCOL["buffer_end"], "2016-03-31")
        self.assertEqual(protocol.PROTOCOL["test_start"], "2016-04-01")
        self.assertEqual(protocol.PROTOCOL["test_end"], "2026-09-30")

    def test_any_edit_to_the_protocol_changes_its_hash(self):
        edited = {**protocol.PROTOCOL, "test_start": "2017-01-01"}
        self.assertNotEqual(protocol.digest(edited), protocol.PROTOCOL_SHA256)
        self.assertEqual(protocol.digest(protocol.PROTOCOL), protocol.PROTOCOL_SHA256)

    def test_authorization_cannot_be_forged(self):
        with self.assertRaises(HoldoutViolation):
            protocol.FinalAuthorization("x", "now", 0, False)

    @unittest.skipUnless(DEFAULT_DB.is_file(), "needs database")
    def test_research_loads_stop_at_buffer_and_final_needs_authorization(self):
        m = load_market(mode=Mode.RESEARCH, verify=False)
        self.assertLessEqual(m.calendar.max(), protocol.BUFFER_END)
        with self.assertRaises(HoldoutViolation):
            load_market(mode=Mode.FINAL, verify=False)
        with self.assertRaises(HoldoutViolation):
            protocol.assert_research_safe([protocol.TEST_START])

    def test_final_authorization_requires_frozen_identical_config_and_logs_runs(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            with mock.patch.multiple(protocol, FROZEN_CONFIG=d / "f.json", FROZEN_LOCK=d / "F.lock",
                                     FINAL_RUN_LOG=d / "runs.jsonl"):
                cfg = {"name": "x", "alpha": 1}
                with self.assertRaises(HoldoutViolation):          # nothing frozen yet
                    protocol.authorize_final(cfg)
                (d / "f.json").write_text(json.dumps(cfg))
                (d / "F.lock").write_text(json.dumps({"config_sha256": protocol.digest(cfg)}))
                with self.assertRaises(HoldoutViolation):          # different config
                    protocol.authorize_final({**cfg, "alpha": 2})
                auth = protocol.authorize_final(cfg)
                self.assertEqual(auth.prior_final_runs, 0)
                self.assertEqual(len(protocol.read_final_log()), 1)
                # re-freezing a different strategy after a final run is blocked unless acknowledged
                cfg2 = {**cfg, "alpha": 3}
                (d / "f.json").write_text(json.dumps(cfg2))
                (d / "F.lock").write_text(json.dumps({"config_sha256": protocol.digest(cfg2)}))
                with self.assertRaises(HoldoutViolation):
                    protocol.authorize_final(cfg2)
                auth2 = protocol.authorize_final(cfg2, acknowledge_reselection=True)
                self.assertTrue(auth2.reselection_acknowledged)


def synthetic(px_col, vol_col, days=60):
    cal = pd.bdate_range("2020-01-01", periods=days)
    adj = np.asarray(px_col, float)[:, None]
    vol = np.asarray(vol_col, float)[:, None]
    return adj, adj.copy(), vol, cal, pd.Index(["X"])


class Integrity(unittest.TestCase):
    def test_unadjusted_split_on_normal_volume_is_repaired(self):
        p = np.r_[np.full(30, 500.0), np.full(30, 50.4)]   # 1:10 with a +0.8% market move
        v = np.r_[np.full(30, 1e5), np.full(30, 1e6)]
        adj, close, vol, cal, sym = synthetic(p, v)
        res = I.audit_prices(adj, close, vol, cal, sym)
        self.assertEqual(res.events.kind.tolist(), ["split"])
        self.assertAlmostEqual(res.clean[30, 0] / res.clean[29, 0] - 1, 0.008, places=9)

    def test_same_drop_on_panic_volume_is_kept_as_a_real_crash(self):
        p = np.r_[np.full(30, 500.0), np.full(30, 50.4)]
        v = np.r_[np.full(30, 1e5), [5e6], np.full(29, 1e6)]  # 100x dollar volume on the day
        adj, close, vol, cal, sym = synthetic(p, v)
        res = I.audit_prices(adj, close, vol, cal, sym)
        self.assertEqual(res.events.kind.tolist(), ["unclassified"])
        self.assertAlmostEqual(res.clean[30, 0] / res.clean[29, 0], 50.4 / 500)

    def test_bad_print_is_quarantined_not_filled(self):
        p = np.full(60, 20.0)
        p[40] = 20_000.0
        adj, close, vol, cal, sym = synthetic(p, np.full(60, 1e5))
        res = I.audit_prices(adj, close, vol, cal, sym)
        self.assertEqual(res.events.kind.tolist(), ["bad_print"])
        self.assertTrue(np.isnan(res.clean[40, 0]))
        self.assertEqual(res.clean[41, 0], 20.0)

    def test_level_shift_on_normal_volume_is_neutralized(self):
        p = np.r_[np.full(30, 28.0), np.full(30, 474.0)]   # x16.9 like LCI 2013-12-13
        adj, close, vol, cal, sym = synthetic(p, np.full(60, 1e5) * np.r_[np.ones(30), np.full(30, 1 / 16.9)])
        res = I.audit_prices(adj, close, vol, cal, sym)
        self.assertEqual(res.events.kind.tolist(), ["level_shift"])
        self.assertAlmostEqual(res.clean[30, 0] / res.clean[29, 0], 1.0)

    def test_neutralize_all_bound_zeroes_every_break(self):
        p = np.r_[np.full(30, 500.0), np.full(30, 50.4)]
        v = np.r_[np.full(30, 1e5), [5e6], np.full(29, 1e6)]
        adj, close, vol, cal, sym = synthetic(p, v)
        res = I.audit_prices(adj, close, vol, cal, sym)
        n = I.neutralized_prices(adj, cal, sym, res)
        self.assertAlmostEqual(n[30, 0] / n[29, 0], 1.0)


def snap(cash=100.0, nav=1000.0, holdings=None):
    return PortfolioSnapshot(pd.Timestamp("2020-01-03"), cash, nav, pd.Series(holdings or {}, dtype=float), 0.0)


class Ctx:
    def __init__(self, portfolio):
        self.portfolio = portfolio


class Portfolio(unittest.TestCase):
    info = pd.DataFrame({"score": [3.0, 2.0, 1.0], "vol_63": [0.2, 0.4, 0.8], "sector": ["T", "T", "E"]},
                        index=pd.Index(["A", "B", "C"], name="symbol"))

    def test_sizers_sum_to_one_with_expected_shapes(self):
        np.testing.assert_allclose(EqualWeight().weights(self.info, None), [1 / 3] * 3)
        np.testing.assert_allclose(RankWeight().weights(self.info, None), [3 / 6, 2 / 6, 1 / 6])
        np.testing.assert_allclose(InverseVolatility().weights(self.info, None), np.array([5, 2.5, 1.25]) / 8.75)

    def test_top_n_breaks_ties_by_symbol(self):
        info = pd.DataFrame({"score": [1.0, 1.0, 0.5]}, index=pd.Index(["B", "A", "C"], name="symbol"))
        self.assertEqual(list(TopN(2).select(info, None)), ["A", "B"])

    def test_cohort_budget_is_capped_by_cash(self):
        self.assertEqual(CohortBudget(13)(snap(cash=10, nav=1300)), 10)
        self.assertEqual(CohortBudget(13)(snap(cash=500, nav=1300)), 100)

    def test_name_cap_counts_existing_lots(self):
        out = NameCap(0.05).apply(pd.Series({"A": 30.0}), Ctx(snap(nav=1000, holdings={"A": 45.0})), None, 0.001)
        self.assertAlmostEqual(out["A"], 5.0 * 1.001)

    def test_sector_cap_is_cumulative_across_same_day_orders(self):
        sectors = {"A": "T", "B": "T", "C": "E"}
        orders = pd.Series({"A": 200.0, "B": 200.0, "C": 200.0})
        out = SectorCap(0.25, sectors).apply(orders, Ctx(snap(cash=1000, nav=1000)), None, 0.0)
        self.assertEqual(out.tolist(), [200.0, 50.0, 200.0])

    def test_weekly_schedule_triggers_after_last_session_of_week(self):
        class C:  # minimal context
            def __init__(self, s, e):
                self.signal_date, self.execution_date = pd.Timestamp(s), pd.Timestamp(e)
        self.assertTrue(Weekly().is_entry(C("2020-01-03", "2020-01-06")))   # Fri -> Mon
        self.assertFalse(Weekly().is_entry(C("2020-01-06", "2020-01-07")))  # Mon -> Tue

    def test_composite_strategy_runs_through_the_backtester(self):
        cal = pd.bdate_range("2020-01-01", periods=150)
        syms = pd.Index(["A", "B", "C"])
        px = np.tile([10.0, 20.0, 30.0], (150, 1))
        rows = [{"date": d, "symbol": s, "score": sc, "sector": "T", "vol_63": 0.2, "fit_date": d,
                 "train_max_label_end": pd.NaT} for d in cal for s, sc in zip(syms, [3, 2, 1])]
        strat = CompositeStrategy(ScoreBook(pd.DataFrame(rows)), Weekly(), TopN(2), EqualWeight(),
                                  CohortBudget(13), [NameCap(0.10)], min_order=1.0)
        res = Backtester(cal, syms, px, strat, OrderManagementSystem(1_000_000, 0)).run(cal[1], cal[-1])
        buys = res.fills.loc[res.fills.side.eq("BUY")]
        self.assertEqual(set(buys.symbol), {"A", "B"})
        first = buys.loc[buys.date.eq(buys.date.min())]
        self.assertAlmostEqual(first.notional.sum(), 1_000_000 / 13)
        # name cap: A's lots never exceed 10% of NAV at planning time (constant prices -> exact)
        a = res.fills.loc[res.fills.symbol.eq("A")]
        held_units = (np.where(a.side.eq("BUY"), a.units, -a.units)).cumsum()  # fills are in date order
        self.assertLessEqual(held_units.max() * 10.0, 0.10 * 1_000_000 + 1e-6)
        self.assertGreater(held_units.max() * 10.0, 0.09 * 1_000_000)  # the cap actually binds


class Metrics(unittest.TestCase):
    def test_hand_computed_path(self):
        nav = pd.Series([100.0, 110.0, 99.0], index=pd.to_datetime(["2020-01-01", "2020-01-02", "2021-01-01"]))
        p = M.performance(nav, 0.0)
        self.assertAlmostEqual(p.total_return, -0.01)
        self.assertAlmostEqual(p.max_drawdown, 99 / 110 - 1)
        self.assertAlmostEqual(p.CAGR, 0.99 ** (365.25 / 366) - 1)
        r = np.array([0.10, -0.10])
        self.assertAlmostEqual(p.Sharpe, np.sqrt(252) * r.mean() / r.std(ddof=1))

    def test_sharpe_uses_daily_equivalent_of_2pct(self):
        idx = pd.bdate_range("2020-01-01", periods=253)
        rng = np.random.default_rng(0)
        r = rng.normal(0.0005, 0.01, 252)
        nav = pd.Series(np.r_[1.0, np.cumprod(1 + r)], index=idx)
        rf = 1.02 ** (1 / 252) - 1
        self.assertAlmostEqual(M.performance(nav).Sharpe, np.sqrt(252) * (r - rf).mean() / r.std(ddof=1))

    def test_benchmark_invests_at_first_close(self):
        idx = pd.bdate_range("2020-01-01", periods=4)
        b = pd.Series([100.0, 101.0, 102.0, 99.0], index=idx)
        nav = M.benchmark_nav(b, idx, idx[1], 1e6)
        np.testing.assert_allclose(nav, [1e6, 1e6, 1e6 * 102 / 101, 1e6 * 99 / 101])


def _quarters(filing_dates, revenue, net_income=1.0):
    fiscal = pd.to_datetime(["2012-03-31", "2012-06-30", "2012-09-30", "2012-12-31", "2013-03-31"])
    return pd.DataFrame({
        "symbol": "X", "fiscal_date": fiscal, "period": "quarter", "filing_date": pd.to_datetime(filing_dates),
        "revenue": revenue, "gross_profit": 1.0, "operating_income": [1.0, 1.0, 1.0, 1.0, 3.0],
        "net_income": net_income, "total_assets": [10.0, 10.0, 10.0, 10.0, 12.0], "total_debt": 1.0,
        "total_equity": 5.0, "cash_and_equivalents": 1.0, "free_cash_flow": 1.0, "operating_cash_flow": 1.0,
        "shares_outstanding": [100.0, 100.0, 100.0, 100.0, 110.0]})


class TrendFeatures(unittest.TestCase):
    """Synthetic statements: every number below can be checked by hand."""

    ON_TIME = ["2012-05-01", "2012-08-01", "2012-11-01", "2013-02-01", "2013-05-01"]

    def test_year_over_year_and_trailing_values(self):
        ev = F.financial_events(_quarters(self.ON_TIME, [100.0, 1.0, 1.0, 1.0, 130.0], [1.0, 2.0, 3.0, 4.0, 5.0]))
        last = ev.loc[ev.fiscal_date.eq("2013-03-31")].iloc[0]
        self.assertAlmostEqual(last.revenue_growth_yoy, 0.30)
        self.assertAlmostEqual(last.asset_growth_yoy, 0.20)
        self.assertAlmostEqual(last.net_issuance_yoy, np.log(1.1))
        self.assertAlmostEqual(last.roa_yoy_change, 3 / 12 - 1 / 10)
        self.assertAlmostEqual(last.net_income_ttm, 2 + 3 + 4 + 5)  # the four latest quarters, not five
        first_year = ev.loc[ev.fiscal_date.lt("2013-03-31")]
        self.assertTrue(first_year.revenue_growth_yoy.isna().all())   # no year-ago quarter exists yet
        self.assertEqual(first_year.net_income_ttm.notna().sum(), 1)  # only once four quarters exist

    def test_year_ago_quarter_filed_late_is_not_used(self):
        # The 2012-03-31 quarter is filed only in June 2013, after the 2013-03-31 quarter was filed:
        # comparing against it on 2013-05-02 would use information nobody had.
        late = ["2013-06-15", "2012-08-01", "2012-11-01", "2013-02-01", "2013-05-01"]
        ev = F.financial_events(_quarters(late, [100.0, 1.0, 1.0, 1.0, 130.0]))
        last = ev.loc[ev.fiscal_date.eq("2013-03-31")].iloc[0]
        self.assertTrue(np.isnan(last.revenue_growth_yoy))
        self.assertTrue(np.isnan(last.net_issuance_yoy))
        self.assertAlmostEqual(last.net_income_ttm, 4.0)  # its own four quarters were all on time

    def test_trailing_sum_needs_all_four_quarters_known(self):
        late = ["2012-05-01", "2012-08-01", "2013-06-15", "2013-02-01", "2013-05-01"]  # Q3 2012 filed late
        ev = F.financial_events(_quarters(late, 1.0))
        self.assertTrue(ev.loc[ev.fiscal_date.ge("2012-12-31"), "net_income_ttm"].isna().all())

    def test_irregular_quarter_spacing_gives_no_trailing_sum(self):
        q = _quarters(self.ON_TIME, 1.0)
        q = q.drop(index=2)  # a missing quarter: four rows now span more than a year
        ev = F.financial_events(q)
        self.assertTrue(ev.net_income_ttm.isna().all())


class TunedRidge(unittest.TestCase):
    def test_penalty_is_chosen_from_training_rows_only_and_recorded(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(2000, 5))
        y = X[:, 0] + rng.normal(scale=3.0, size=2000)
        m = build_model("ridge_tuned", {"alphas": [1.0, 1000.0]}).fit(X, y)
        self.assertIsInstance(m, TimeTunedRidge)
        self.assertIn(m.alpha_, (1.0, 1000.0))
        self.assertEqual(set(m.inner_scores_), {1.0, 1000.0})
        self.assertEqual(m.alpha_, max(m.inner_scores_, key=m.inner_scores_.get))
        self.assertEqual(m.predict(X[:7]).shape, (7,))
        # the inner check is the last quarter of rows: scrambling its targets changes its scores,
        # scrambling nothing outside the training rows can (there is nothing else it sees)
        y2 = y.copy()
        y2[1500:] = rng.permutation(y2[1500:])
        m2 = build_model("ridge_tuned", {"alphas": [1.0, 1000.0]}).fit(X, y2)
        self.assertNotEqual(m.inner_scores_, m2.inner_scores_)


if __name__ == "__main__":
    unittest.main()
