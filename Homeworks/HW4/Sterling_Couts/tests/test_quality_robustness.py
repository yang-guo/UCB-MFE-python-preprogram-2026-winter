"""Independent small examples for filing delays and lot attribution."""

import numpy as np
import pandas as pd

from quality_robustness import replace_financials, attribute_lots, sector_quality_ranks


def test_filing_delay_excludes_invalid_lags_and_cannot_see_future_filings():
    f = pd.DataFrame(
        {
            "symbol": ["A"] * 4,
            "period": ["quarter"] * 4,
            "fiscal_date": pd.to_datetime(
                ["2011-09-30", "2011-12-31", "2012-03-31", "2012-06-30"]
            ),
            "filing_date": pd.to_datetime(
                ["2011-09-29", "2012-02-01", "2012-05-01", "2012-08-01"]
            ),
            "operating_income": [999, 10, 20, 999],
        }
    )
    for col in [
        "total_assets",
        "revenue",
        "net_income",
        "total_debt",
        "cash_and_equivalents",
        "free_cash_flow",
        "operating_cash_flow",
    ]:
        f[col] = 100.0
    p = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2012-02-01", "2012-02-02", "2012-03-03", "2012-05-02"]
            ),
            "symbol": ["A"] * 4,
            "eligible": True,
            "risk_screen_pass": True,
        }
    )
    sectors = pd.Series({"A": "Test"})
    normal = replace_financials(p, f, sectors, 0)
    delayed = replace_financials(p, f, sectors, 30)
    np.testing.assert_allclose(
        normal.operating_roa, [np.nan, 0.1, 0.1, 0.2], equal_nan=True
    )
    np.testing.assert_allclose(
        delayed.operating_roa, [np.nan, np.nan, 0.1, 0.1], equal_nan=True
    )
    changed = f.copy()
    changed.loc[changed.filing_date.ge("2012-08-01"), "operating_income"] = -1e9
    pd.testing.assert_frame_equal(delayed, replace_financials(p, changed, sectors, 30))


def test_attribution_counts_closed_open_lots_and_actual_fees_once():
    fills = pd.DataFrame(
        [
            dict(
                lot_id=0,
                symbol="A",
                side="BUY",
                date=pd.Timestamp("2013-01-02"),
                units=2.0,
                cash_change=-202.0,
            ),
            dict(
                lot_id=1,
                symbol="B",
                side="BUY",
                date=pd.Timestamp("2013-02-01"),
                units=1.0,
                cash_change=-101.0,
            ),
            dict(
                lot_id=0,
                symbol="A",
                side="SELL",
                date=pd.Timestamp("2013-04-02"),
                units=2.0,
                cash_change=218.0,
            ),
        ]
    )
    opened = pd.DataFrame([dict(lot_id=1, symbol="B", units=1.0)])
    quotes = pd.DataFrame(
        {"A": [110.0], "B": [120.0]}, index=pd.to_datetime(["2013-12-31"])
    )
    out = attribute_lots(fills, opened, quotes, 1035.0, 1000.0)
    np.testing.assert_allclose(out.net_pnl, [16.0, 19.0])


def test_sector_quality_percentiles_ignore_ineligible_peers():
    p = pd.DataFrame(
        {
            "date": pd.Timestamp("2013-01-31"),
            "sector": ["A"] * 11 + ["B"] * 2,
            "eligible": [True] * 10 + [False] + [True] * 2,
            "risk_screen_pass": True,
            "operating_roa": list(range(13)),
            "fcf_to_assets": list(range(13)),
        }
    )
    out = sector_quality_ranks(p)
    np.testing.assert_allclose(
        out.sector_rank_operating_roa.iloc[:10], np.arange(1, 11) / 10
    )
    assert out.sector_rank_operating_roa.iloc[10:].isna().all()
