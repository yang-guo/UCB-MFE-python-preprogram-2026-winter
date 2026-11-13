import unittest
import numpy as np
import pandas as pd
from feature_engineering import financial_events, join_financials, sector_financial_ranks, FINANCIAL_FEATURES

class FinancialTests(unittest.TestCase):
    def test_filing_day_is_not_available(self):
        reports = pd.DataFrame({"symbol": ["A", "A"], "period": "quarter", "fiscal_date": pd.to_datetime(["2012-09-30", "2012-12-31"]), "filing_date": pd.to_datetime(["2012-11-01", "2013-02-01"]), "revenue": [100., 200.], "net_income": [10., 80.], "operating_income": [20., 90.], "total_assets": [100., 200.], "total_debt": [10., 20.], "cash_and_equivalents": [5., 10.], "free_cash_flow": [3., 4.], "operating_cash_flow": [5., 6.]})
        panel = pd.DataFrame({"symbol": "A", "date": pd.date_range("2013-01-30", "2013-02-03")})
        universe = pd.DataFrame({"symbol": ["A"], "sector": ["Tech"]})
        old = join_financials(panel, financial_events(reports.iloc[:1]), universe)
        new = join_financials(panel, financial_events(reports), universe)
        before = panel.date.le("2013-02-01")
        pd.testing.assert_frame_equal(old.loc[before], new.loc[before])
        self.assertAlmostEqual(new.net_margin.iloc[-1], .4)
        self.assertEqual(new.available_date.iloc[-1], pd.Timestamp("2013-02-02"))

    def test_sector_rank_and_fallback(self):
        panel = pd.DataFrame({"symbol": list("ABCDEFGH"), "date": pd.Timestamp("2013-01-02"), "eligible": [True]*7+[False], "sector": ["Large"]*5+["Small"]*3})
        for name in FINANCIAL_FEATURES[:4]:
            panel[name] = [1.,2.,3.,4.,5.,6.,np.nan,100.]
        ranked = sector_financial_ranks(panel)
        np.testing.assert_allclose(ranked.net_margin.iloc[:5], [.2,.4,.6,.8,1.])
        self.assertEqual(ranked.net_margin.iloc[5], 1.)
        self.assertTrue(pd.isna(ranked.net_margin.iloc[6]))
        self.assertEqual(ranked.financial_rank_fallback.sum(), 2)
        self.assertTrue(ranked.net_margin.iloc[7:].isna().all())
