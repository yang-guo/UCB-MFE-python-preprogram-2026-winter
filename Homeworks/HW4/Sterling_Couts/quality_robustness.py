"""Additional availability checks and accounting attribution for rank research.

No original data or model is modified. Delays use the supplied course helpers;
they test sensitivity to availability assumptions, not original filing vintages.
"""

import numpy as np
import pandas as pd

from hw4.data import financial_events, join_financials

FINANCIALS = [
    "net_margin",
    "operating_roa",
    "debt_to_assets",
    "cash_to_assets",
    "fcf_to_assets",
    "accruals_to_assets",
]


def replace_financials(panel, fundamentals, sectors, lag_days):
    """Rebuild backward-as-of financial features, retaining all candidate rows."""
    if lag_days < 0:
        raise ValueError("Extra delay must be nonnegative")
    drop = FINANCIALS + [
        "fiscal_date",
        "filing_date",
        "available_date",
        "filing_age_days",
        "sector",
        "rank_operating_roa",
        "rank_fcf_to_assets",
        "sector_relative_operating_roa",
    ]
    base = panel.drop(columns=drop, errors="ignore").copy()
    base["_original_order"] = np.arange(len(base))
    universe = sectors.rename("sector").rename_axis("symbol").reset_index()
    joined = (
        join_financials(
            base, financial_events(fundamentals, extra_lag_days=lag_days), universe
        )
        .sort_values("_original_order")
        .drop(columns="_original_order")
    )
    joined.index = panel.index
    pd.testing.assert_frame_equal(joined[["symbol", "date"]], panel[["symbol", "date"]])
    eligible = joined.eligible & joined.risk_screen_pass
    for feature in ["operating_roa", "fcf_to_assets"]:
        joined["rank_" + feature] = np.nan
        joined.loc[eligible, "rank_" + feature] = (
            joined.loc[eligible].groupby("date")[feature].rank(pct=True)
        )
    known = joined.available_date.notna()
    assert (joined.loc[known, "available_date"] <= joined.loc[known, "date"]).all()
    assert (joined.loc[known, "filing_date"] > joined.loc[known, "fiscal_date"]).all()
    assert (
        (joined.loc[known, "available_date"] - joined.loc[known, "filing_date"])
        .eq(pd.Timedelta(days=1 + lag_days))
        .all()
    )
    return joined


def sector_quality_ranks(panel):
    """Within-sector percentiles; small groups (<10 valid peers) stay missing."""
    p = panel.copy()
    mask = p.eligible & p.risk_screen_pass
    for feature in ["operating_roa", "fcf_to_assets"]:
        p["sector_rank_" + feature] = np.nan
        groups = p.loc[mask].groupby(["date", "sector"])[feature]
        p.loc[mask, "sector_rank_" + feature] = groups.rank(pct=True).where(
            groups.transform("count").ge(10)
        )
    return p


def attribute_lots(fills, open_lots, quotes, final_nav, initial_cash):
    """Reconcile closed and still-held lots, including both sides' actual fees.

    This is dollar attribution, not additive time-weighted sector returns.
    Terminal open lots are marked, not assigned fictitious liquidation costs.
    """
    buys = fills.loc[fills.side.eq("BUY")].set_index("lot_id")
    sells = fills.loc[fills.side.eq("SELL")].set_index("lot_id")
    opened = open_lots.set_index("lot_id")
    assert buys.index.is_unique and sells.index.is_unique and opened.index.is_unique
    assert not set(sells.index) & set(opened.index)
    assert set(buys.index) == set(sells.index) | set(opened.index)
    rows = []
    for lot_id, buy in buys.iterrows():
        if lot_id in sells.index:
            exit_value = sells.loc[lot_id, "cash_change"]
            terminal_date = sells.loc[lot_id, "date"]
            np.testing.assert_allclose(sells.loc[lot_id, "units"], buy.units)
            status = "sold"
        else:
            terminal_date = quotes.index[-1]
            price = quotes.loc[terminal_date, buy.symbol]
            assert np.isfinite(price) and price > 0
            np.testing.assert_allclose(opened.loc[lot_id, "units"], buy.units)
            exit_value = buy.units * price
            status = "open_mark"
        rows.append(
            dict(
                lot_id=lot_id,
                symbol=buy.symbol,
                entry_date=buy.date,
                end_date=terminal_date,
                status=status,
                entry_cost=-buy.cash_change,
                terminal_value=exit_value,
                net_pnl=exit_value + buy.cash_change,
            )
        )
    result = pd.DataFrame(rows)
    np.testing.assert_allclose(
        result.net_pnl.sum(), final_nav - initial_cash, atol=1e-6
    )
    return result
