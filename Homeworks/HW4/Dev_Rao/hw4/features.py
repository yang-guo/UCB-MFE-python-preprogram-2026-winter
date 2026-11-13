"""Causal features: daily price/liquidity features, filing-dated fundamentals,
dated membership + eligibility, then same-date cross-sectional normalisation.

Code adapted from `Lectures/modeling/course_data.py` is marked in docstrings:
`price_features` (reindex to the market calendar, no filling),
`financial_events` / `join_financials` (filing date + 1 day, drop nonpositive
filing lags, 550-day staleness, late old filings never replace newer ones) and
`apply_membership` (join by the quarter of the NEXT-session entry, selection
date <= signal date).
"""

import bisect

import numpy as np
import pandas as pd

from . import config
from .labels import labels_for_symbol

# --------------------------------------------------------------------------------
# Feature lists (one-line definitions are in README.md)
# --------------------------------------------------------------------------------
PRICE_FEATURES = ["mom_21", "mom_63", "mom_126", "mom_12_1", "rev_5",
                  "vol_21", "vol_63", "vol_252", "beta_252", "ivol_63",
                  "dd_252", "px_ma200", "ma50_ma200", "max_ret_21", "skew_63"]
LIQUIDITY_FEATURES = ["log_dv_63", "amihud_63", "log_mcap", "turnover_63"]
FUNDAMENTAL_FEATURES = ["ey", "bm", "sp", "fcfy", "roa", "roe", "gp_a", "op_margin",
                        "accruals_a", "debt_a", "cash_a", "rev_yoy", "ni_chg_a",
                        "asset_growth", "share_iss", "filing_age_days"]
RAW_FEATURES = PRICE_FEATURES + LIQUIDITY_FEATURES + FUNDAMENTAL_FEATURES
SECTOR_RELATIVE = ["ey", "bm", "sp", "mom_12_1", "roa"]
SECTOR_REL_FEATURES = [f"{c}_srel" for c in SECTOR_RELATIVE] + ["sector_mom_126"]
REGIME_FEATURES = ["mkt_mom_63", "mkt_vol_21"]
# Model inputs: percentile ranks of every raw + sector-relative feature, regime
# features left raw (constant within a date), sector as a categorical.
RANKED_FEATURES = [f"r_{c}" for c in RAW_FEATURES + SECTOR_REL_FEATURES]
MODEL_NUMERIC = RANKED_FEATURES + REGIME_FEATURES
MODEL_CATEGORICAL = ["sector"]


def weekly_signal_dates(calendar, start, end, extra=(config.TEST_FIRST_SIGNAL,)):
    """Last market session of each calendar week (Mon-Sun), plus explicit extras
    (2016-03-31 so the first test entry is the 2016-04-01 close)."""
    cal = pd.Series(calendar, index=calendar)
    weekly = cal.groupby(calendar.to_period("W-SUN")).max()
    dates = pd.DatetimeIndex(weekly.values).union(pd.DatetimeIndex(list(extra)))
    dates = dates[(dates >= start) & (dates <= end)]
    assert dates.isin(calendar).all()
    return dates


def benchmark_daily(bench, calendar):
    b = bench.reindex(calendar)
    r = b.pct_change(fill_method=None)
    regime = pd.DataFrame({"mkt_mom_63": b.pct_change(63, fill_method=None),
                           "mkt_vol_21": r.rolling(21, min_periods=21).std() * np.sqrt(252)})
    return r, regime


# --------------------------------------------------------------------------------
# Price / liquidity features (per stock, daily, then sampled on signal dates)
# --------------------------------------------------------------------------------
def price_features_for_symbol(g, calendar, mkt_ret, signal_dates):
    """Adapted from course_data.price_features: reindex one stock to the market
    calendar WITHOUT filling, compute rolling features on the daily series, and
    return only signal-date rows that have a quote. Also returns the labels."""
    s = g.set_index("date").reindex(calendar)
    px, close, volume = s.adj_close, s.close, s.volume
    r = px.pct_change(fill_method=None)
    f = pd.DataFrame(index=calendar)
    f["mom_21"] = px.pct_change(21, fill_method=None)
    f["mom_63"] = px.pct_change(63, fill_method=None)
    f["mom_126"] = px.pct_change(126, fill_method=None)
    f["mom_12_1"] = px.shift(21) / px.shift(252) - 1
    f["rev_5"] = px.pct_change(5, fill_method=None)
    for w in (21, 63, 252):
        f[f"vol_{w}"] = r.rolling(w, min_periods=int(0.9 * w)).std() * np.sqrt(252)
    m = mkt_ret
    cov252 = r.rolling(252, min_periods=200).cov(m)
    f["beta_252"] = cov252 / m.rolling(252, min_periods=200).var()
    beta63 = r.rolling(63, min_periods=57).cov(m) / m.rolling(63, min_periods=57).var()
    resid_var = r.rolling(63, min_periods=57).var() - beta63 ** 2 * m.rolling(63, min_periods=57).var()
    f["ivol_63"] = np.sqrt(resid_var.clip(lower=0)) * np.sqrt(252)
    f["dd_252"] = px / px.rolling(252, min_periods=200).max() - 1
    ma50 = px.rolling(50, min_periods=45).mean()
    ma200 = px.rolling(200, min_periods=180).mean()
    f["px_ma200"] = px / ma200 - 1
    f["ma50_ma200"] = ma50 / ma200 - 1
    f["max_ret_21"] = r.rolling(21, min_periods=19).max()
    f["skew_63"] = r.rolling(63, min_periods=57).skew()
    # Liquidity: close and volume are both split-adjusted, so close*volume is the
    # true contemporaneous dollar volume.
    dv = (close * volume).where(lambda x: x > 0)
    dv_med = dv.rolling(63, min_periods=50).median()
    f["dv_med_63"] = dv_med
    f["log_dv_63"] = np.log(dv_med)
    f["amihud_63"] = (r.abs() / dv).rolling(63, min_periods=50).mean() * 1e9
    f["vol_med_63"] = volume.rolling(63, min_periods=50).median()
    f["n_history"] = px.notna().cumsum()
    f["close"] = close
    f["adj_close"] = px
    out = f.reindex(signal_dates)
    out = out.loc[out.adj_close.notna()]           # a signal needs a quote on t
    labels = labels_for_symbol(px, calendar, out.index)
    out = out.join(labels)
    out["symbol"] = g.symbol.iloc[0]
    return out.reset_index(names="date")


# --------------------------------------------------------------------------------
# Fundamentals: filing-event level TTM / YoY quantities
# --------------------------------------------------------------------------------
def _consecutive(fiscals):
    """Four fiscal quarter-ends are consecutive if they span ~9 months."""
    span = (fiscals[-1] - fiscals[0]).days
    return 240 <= span <= 300


def fundamental_events(fund):
    """One row per (symbol, availability date) with what was KNOWN on that date.

    Adapted from course_data.financial_events: quarterly rows only, nonpositive
    filing lags dropped, available_date = filing_date + 1 day. Extended to
    build TTM sums and YoY comparisons at the filing-event level:
      * the known set on date a = all quarters with available_date <= a;
      * the "current" quarter = the newest fiscal period in the known set, so a
        late-filed OLDER quarter never replaces a newer one (it can only fill a
        gap in the TTM window);
      * TTM = sum of the 4 newest known quarters, only if they are consecutive
        and all filed by a (no backfilling from future reports);
      * YoY compares the current quarter to the known quarter ~1 year earlier.
    """
    f = fund.loc[fund.period.eq("quarter") & fund.filing_date.gt(fund.fiscal_date)].copy()
    n_dropped = int((fund.period.eq("quarter") & ~fund.filing_date.gt(fund.fiscal_date)).sum())
    f["available_date"] = f.filing_date + pd.Timedelta(days=config.FILING_LAG_DAYS)
    f = f.sort_values(["symbol", "available_date", "fiscal_date"])
    flow = ["revenue", "gross_profit", "operating_income", "net_income",
            "operating_cash_flow", "free_cash_flow"]
    rows = []
    for symbol, g in f.groupby("symbol", sort=False):
        known = {}           # fiscal_date -> record
        fiscals = []         # sorted known fiscal dates
        recs = g.to_dict("records")
        i = 0
        while i < len(recs):
            a = recs[i]["available_date"]
            while i < len(recs) and recs[i]["available_date"] == a:
                r = recs[i]
                if r["fiscal_date"] not in known:
                    bisect.insort(fiscals, r["fiscal_date"])
                known[r["fiscal_date"]] = r
                i += 1
            q = fiscals[-1]
            cur = known[q]
            ev = {"symbol": symbol, "available_date": a, "fiscal_date": q}
            for c in ["total_assets", "total_equity", "total_debt", "cash_and_equivalents",
                      "shares_outstanding", "revenue", "net_income"]:
                ev[c] = cur[c]
            last4 = fiscals[-4:]
            ttm_ok = len(last4) == 4 and _consecutive(last4)
            for c in flow:
                vals = [known[d][c] for d in last4] if ttm_ok else []
                ev[f"ttm_{c}"] = float(np.sum(vals)) if ttm_ok and np.all(np.isfinite(vals)) else np.nan
            # year-ago quarter among KNOWN quarters (same fiscal quarter +/- 45 days)
            target = q - pd.Timedelta(days=365)
            j = bisect.bisect_left(fiscals, target - pd.Timedelta(days=45))
            prev = None
            while j < len(fiscals) and fiscals[j] <= target + pd.Timedelta(days=45):
                if prev is None or abs((fiscals[j] - target).days) < abs((prev - target).days):
                    prev = fiscals[j]
                j += 1
            for c in ["revenue", "net_income", "total_assets", "shares_outstanding"]:
                ev[f"{c}_ya"] = known[prev][c] if prev is not None else np.nan
            rows.append(ev)
    ev = pd.DataFrame(rows)
    assets = ev.total_assets.where(ev.total_assets > 0)
    equity = ev.total_equity.where(ev.total_equity > 0)
    ttm_rev = ev.ttm_revenue.where(ev.ttm_revenue > 0)
    out = ev[["symbol", "available_date", "fiscal_date", "shares_outstanding",
              "total_equity", "ttm_net_income", "ttm_revenue", "ttm_free_cash_flow"]].copy()
    out["roa"] = ev.ttm_net_income / assets
    out["roe"] = ev.ttm_net_income / equity
    out["gp_a"] = ev.ttm_gross_profit / assets
    out["op_margin"] = ev.ttm_operating_income / ttm_rev
    out["accruals_a"] = (ev.ttm_net_income - ev.ttm_operating_cash_flow) / assets
    out["debt_a"] = ev.total_debt / assets
    out["cash_a"] = ev.cash_and_equivalents / assets
    rev_ya = ev.revenue_ya.where(ev.revenue_ya > 0)
    out["rev_yoy"] = ev.revenue / rev_ya - 1
    out["ni_chg_a"] = (ev.net_income - ev.net_income_ya) / assets
    out["asset_growth"] = ev.total_assets / ev.total_assets_ya.where(ev.total_assets_ya > 0) - 1
    out["share_iss"] = ev.shares_outstanding / ev.shares_outstanding_ya.where(ev.shares_outstanding_ya > 0) - 1
    num = out.select_dtypes("number").columns
    out[num] = out[num].replace([np.inf, -np.inf], np.nan)
    assert (out.available_date > out.fiscal_date).all()
    out.attrs["dropped_nonpositive_lag"] = n_dropped
    return out.sort_values(["available_date", "symbol"]).reset_index(drop=True)


def join_fundamentals(panel, events):
    """Backward as-of join (course_data.join_financials) + price-based ratios.

    Market-cap proxy = split-adjusted close x latest AVAILABLE split-adjusted
    shares. Features whose fiscal period is > 550 days old are blanked.
    """
    left = panel.sort_values("date")
    right = events.sort_values("available_date")
    j = pd.merge_asof(left, right, left_on="date", right_on="available_date", by="symbol",
                      direction="backward")
    assert (j.available_date.isna() | (j.available_date <= j.date)).all(), "future filing joined"
    j["filing_age_days"] = (j.date - j.available_date).dt.days.astype("float64")
    shares = j.shares_outstanding.where(j.shares_outstanding > 0)
    mcap = (j.close * shares).where(lambda x: x > 0)
    j["log_mcap"] = np.log(mcap)
    j["turnover_63"] = j.vol_med_63 / shares
    j["ey"] = j.ttm_net_income / mcap
    j["bm"] = j.total_equity / mcap
    j["sp"] = j.ttm_revenue / mcap
    j["fcfy"] = j.ttm_free_cash_flow / mcap
    stale = (j.date - j.fiscal_date).dt.days.gt(config.MAX_FISCAL_AGE_DAYS) | j.fiscal_date.isna()
    fund_cols = [c for c in FUNDAMENTAL_FEATURES] + ["log_mcap", "turnover_63"]
    j.loc[stale, fund_cols] = np.nan
    num = j.select_dtypes("number").columns
    j[num] = j[num].replace([np.inf, -np.inf], np.nan)
    return j


# --------------------------------------------------------------------------------
# Membership and eligibility
# --------------------------------------------------------------------------------
def apply_membership(panel, membership, calendar):
    """Adapted from course_data.apply_membership: the list used is the one for the
    quarter of the NEXT-session entry, and it must be known (selection_date <=
    signal date). Rows are kept; `is_member` / `eligible` are flags."""
    calendar = pd.DatetimeIndex(calendar)
    pos = calendar.get_indexer(panel.date)
    assert (pos >= 0).all()
    nxt = np.where(pos + 1 < len(calendar), pos + 1, pos)
    out = panel.copy()
    out["membership_quarter"] = calendar[nxt].to_period("Q").start_time
    out.loc[pos + 1 >= len(calendar), "membership_quarter"] = pd.NaT
    right = membership.rename(columns={"quarter_start": "membership_quarter"})
    out = out.merge(right, on=["symbol", "membership_quarter"], how="left", validate="many_to_one")
    out["is_member"] = out.universe_rank.notna()
    mem = out.is_member
    assert (out.loc[mem, "selection_date"] <= out.loc[mem, "date"]).all(), \
        "universe selection unavailable at the signal close"
    out["eligible"] = (mem & out.n_history.ge(config.MIN_HISTORY_SESSIONS)
                       & out.dv_med_63.ge(config.MIN_MEDIAN_DOLLAR_VOLUME))
    return out


# --------------------------------------------------------------------------------
# Same-date cross-sectional transforms (run on ELIGIBLE rows only)
# --------------------------------------------------------------------------------
def cross_sectional(panel):
    """Percentile ranks within each signal date's eligible stocks (NaN stays NaN),
    sector-relative versions of key features, sector momentum, and a liquidity
    rank used for the large-cap/liquid universe option."""
    assert panel.eligible.all(), "cross-sectional stats must use eligible rows only"
    p = panel.copy()
    by_ds = p.groupby(["date", "sector"])
    for c in SECTOR_RELATIVE:
        p[f"{c}_srel"] = p[c] - by_ds[c].transform("median")
    p["sector_mom_126"] = by_ds["mom_126"].transform("median")
    by_d = p.groupby("date")
    for c in RAW_FEATURES + SECTOR_REL_FEATURES:
        p[f"r_{c}"] = by_d[c].rank(pct=True).astype("float32")
    p["liq_rank"] = by_d["dv_med_63"].rank(ascending=False, method="first")
    return p
