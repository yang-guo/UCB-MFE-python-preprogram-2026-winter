"""Point-in-time features and 90-day labels for HW4.

The panel has one row per (signal date, stock) for stocks that are eligible on
that signal date. Signal dates are every 5th market session ("weekly"), on a
grid that contains 2016-03-31 so the first test entry is the first test close.

Eligible = dated universe member (entry quarter) with a usable quote at t,
63 trailing returns, >= $1M trailing dollar volume and no extreme daily print
in the trailing year. All of these are known at t's close.

Timing for a row with signal date t (all values known after t's close):
- price features use adjusted closes up to and including t;
- financial features use quarterly reports from filing_date + 1 day <= t;
- membership is the quarter of the *entry* session t+1 (as in the data guide);
- the label is the adjusted-close return from the close of t+1 to the first
  quote on/after t+1 + 90 calendar days (at most 5 sessions late, else missing).

Adapted from Lectures/modeling/course_data.py: the quarantine rules, the
filing-date availability rule, the "late old filing does not replace a newer
period" rule, the 550-day stale blanking, and the entry/exit label convention.
The wide-matrix implementation and the feature set are my own.
"""

from __future__ import annotations

from pathlib import Path
import warnings

import duckdb
import numpy as np
import pandas as pd

HOLDING_DAYS = 90
MAX_EXIT_DELAY = 5
SIGNAL_STEP = 5
MIN_DOLLAR_VOL = 1e6        # eligibility: trailing 63-day mean dollar volume
MAX_DAILY_MOVE = np.log(4)  # eligibility: no |log return| > log 4 (+300% / -75%) in 252 days
ANCHOR = pd.Timestamp("2016-03-31")  # last session before the test period

PRICE_FEATURES = ["mom_1m", "mom_3m", "mom_6m_1m", "mom_12_1", "vol_63", "vol_252",
                  "max_ret_21", "dist_high_252", "beta_252", "idio_vol_252",
                  "log_dollar_vol", "log_amihud"]
FUND_FEATURES = ["log_mcap", "earnings_yield", "book_to_market", "sales_to_price",
                 "cashflow_to_price", "gross_profit_to_assets", "roe", "asset_growth",
                 "sales_growth", "earnings_change_to_price", "accruals_to_assets",
                 "debt_to_assets", "share_issuance", "filing_age_days"]
FEATURES = PRICE_FEATURES + FUND_FEATURES  # every one ranked within date; plus "sector"


# ----------------------------------------------------------------- loading

def _ns(s):
    """DuckDB returns dates at varying precision; pandas joins need one."""
    return pd.to_datetime(s).astype("datetime64[ns]")


def load_market(database: str | Path, start="2008-01-01", end="2026-09-30"):
    """Wide adjusted close, split-adjusted close, volume; calendar; benchmark."""
    with duckdb.connect(str(database), read_only=True) as con:
        long = con.execute("""SELECT symbol, date, close, adj_close, volume FROM prices
                              WHERE date BETWEEN ? AND ?""", [start, end]).fetchdf()
        bench = con.execute("""SELECT date, adj_close FROM benchmark WHERE date BETWEEN ? AND ?
                               ORDER BY date""", [start, end]).fetchdf()
        universe = con.execute("SELECT symbol, sector FROM universe").fetchdf()
        membership = con.execute("""SELECT quarter_start, symbol, universe_rank, selection_date
                                    FROM universe_membership""").fetchdf()
    long["date"] = _ns(long.date)
    bench["date"] = _ns(bench.date)
    calendar = pd.DatetimeIndex(bench.date)
    long = quarantine_quotes(long, calendar)
    wide = {c: long.pivot(index="date", columns="symbol", values=c).reindex(calendar)
            for c in ["adj_close", "close", "volume"]}
    membership["quarter_start"] = _ns(membership.quarter_start)
    membership["selection_date"] = _ns(membership.selection_date)
    return dict(A=wide["adj_close"], C=wide["close"], V=wide["volume"], calendar=calendar,
                bench=bench.set_index("date").adj_close, universe=universe,
                membership=membership)


def quarantine_quotes(long: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """Drop rows that are not usable quotes, with a warning (never filled).

    - non-positive / missing adjusted close (data guide: BHRB);
    - dates off the market calendar (data guide: 28 rows);
    - zero or missing volume: no trade happened, and in this source such rows
      are often a second, unrelated price series (e.g. AAM prints of $0.19
      between real $24,000 closes). Knowable at that day's close.
    """
    bad = ~np.isfinite(long.adj_close) | (long.adj_close <= 0)
    off = ~long.date.isin(calendar)
    no_trade = ~(long.volume > 0)
    warnings.warn(f"Quarantined {int(bad.sum())} non-positive, {int(off.sum())} off-calendar "
                  f"and {int((no_trade & ~bad & ~off).sum())} zero-volume price rows", stacklevel=3)
    return long[~bad & ~off & ~no_trade]


def signal_dates(calendar: pd.DatetimeIndex, start, end, anchor=ANCHOR) -> pd.DatetimeIndex:
    """Every SIGNAL_STEP-th session, on a grid that includes ``anchor``."""
    anchor = pd.Timestamp(anchor)
    if anchor not in calendar:
        raise ValueError(f"Anchor {anchor.date()} is not a market session in the data")
    pos = np.arange(len(calendar))
    anchor = calendar.get_loc(anchor)
    grid = calendar[(pos - anchor) % SIGNAL_STEP == 0]
    grid = grid[(grid >= pd.Timestamp(start)) & (grid <= pd.Timestamp(end))]
    return grid[grid < calendar[-1]]  # need a next session to enter


# ------------------------------------------------------------ price features

def price_feature_matrices(m: dict) -> dict[str, pd.DataFrame]:
    """Trailing price/volume features, one wide matrix each. No future rows used."""
    A, C, V = m["A"], m["C"], m["V"]
    r = A.pct_change(fill_method=None)
    mkt = m["bench"].reindex(A.index).pct_change()
    mm = r.notna().mul(mkt, axis=0).where(r.notna())  # market return on the stock's quote days
    out = {
        "mom_1m": A / A.shift(21) - 1,
        "mom_3m": A / A.shift(63) - 1,
        "mom_6m_1m": A.shift(21) / A.shift(126) - 1,
        "mom_12_1": A.shift(21) / A.shift(252) - 1,
        "vol_63": r.rolling(63, min_periods=60).std(),
        "vol_252": r.rolling(252, min_periods=200).std(),
        "max_ret_21": r.rolling(21, min_periods=15).max(),
        "dist_high_252": A / A.rolling(252, min_periods=200).max() - 1,
    }
    roll = lambda x: x.rolling(252, min_periods=200).mean()
    mean_r, mean_m = roll(r), roll(mm)
    cov = roll(r * mm) - mean_r * mean_m
    var_m = roll(mm * mm) - mean_m ** 2
    beta = cov / var_m
    out["beta_252"] = beta
    resid_var = (roll(r * r) - mean_r ** 2) - beta * cov
    out["idio_vol_252"] = np.sqrt(resid_var.clip(lower=0))
    dollar = C * V  # split-adjusted close x split-adjusted volume = nominal dollars traded
    out["log_dollar_vol"] = np.log(dollar.rolling(63, min_periods=40).mean().where(lambda x: x > 0))
    illiq = (r.abs() / dollar.where(dollar > 0)).rolling(63, min_periods=40).mean()
    out["log_amihud"] = np.log(illiq.where(illiq > 0))
    # Data-quality flag (not a model feature): largest absolute daily log move in a year.
    out["max_abs_logret_252"] = np.abs(np.log1p(r)).rolling(252, min_periods=1).max()
    return {k: v.astype("float32") for k, v in out.items()}


# ------------------------------------------------------------ fundamentals

def load_fundamental_events(database, end="2026-09-30") -> pd.DataFrame:
    """Quarterly records -> availability-dated events with TTM and YoY values."""
    with duckdb.connect(str(database), read_only=True) as con:
        f = con.execute("""SELECT symbol, fiscal_date, filing_date, revenue, gross_profit,
                net_income, operating_cash_flow, total_assets, total_equity, total_debt,
                shares_outstanding FROM fundamentals
                WHERE period = 'quarter' AND fiscal_date >= '2004-01-01' AND fiscal_date <= ?""",
                            [end]).fetchdf()
    for c in ["fiscal_date", "filing_date"]:
        f[c] = _ns(f[c])
    # Invalid or missing filing lag -> not usable (data guide baseline policy).
    f = f[f.filing_date > f.fiscal_date].sort_values(["symbol", "fiscal_date"]).copy()
    f["available_date"] = f.filing_date + pd.Timedelta(days=1)
    g = f.groupby("symbol", sort=False)

    def lag(col, k):
        return g[col].shift(k)

    # TTM sums need four consecutive quarters (span ~ 9 months between 1st and 4th).
    span3 = (f.fiscal_date - lag("fiscal_date", 3)).dt.days
    ttm_ok = span3.between(250, 300)
    # YoY comparisons need the same quarter last year.
    span4 = (f.fiscal_date - lag("fiscal_date", 4)).dt.days
    yoy_ok = span4.between(340, 390)
    # Prior quarters used must themselves have been public by this filing.
    prior_known = lag("available_date", 4) <= f.available_date

    for col in ["revenue", "gross_profit", "net_income", "operating_cash_flow"]:
        s = g[col].rolling(4, min_periods=4).sum().reset_index(level=0, drop=True)
        f[col + "_ttm"] = s.where(ttm_ok)
    assets = f.total_assets.where(f.total_assets > 0)
    equity = f.total_equity.where(f.total_equity > 0)
    f["gross_profit_to_assets"] = f.gross_profit_ttm / assets
    f["roe"] = f.net_income_ttm / equity
    f["accruals_to_assets"] = (f.net_income_ttm - f.operating_cash_flow_ttm) / assets
    f["debt_to_assets"] = f.total_debt / assets
    ok = yoy_ok & prior_known
    f["asset_growth"] = (f.total_assets / lag("total_assets", 4).where(lambda x: x > 0) - 1).where(ok)
    f["sales_growth"] = (f.revenue / lag("revenue", 4).where(lambda x: x > 0) - 1).where(ok)
    f["share_issuance"] = (f.shares_outstanding / lag("shares_outstanding", 4)
                           .where(lambda x: x > 0) - 1).where(ok)
    f["ni_change"] = (f.net_income - lag("net_income", 4)).where(ok)

    # A late filing of an old fiscal period must not replace a newer known one.
    f = f.sort_values(["symbol", "available_date", "fiscal_date"])
    f = f[f.fiscal_date.eq(f.groupby("symbol").fiscal_date.cummax())]
    f = f.drop_duplicates(["symbol", "available_date"], keep="last")
    keep = ["symbol", "fiscal_date", "available_date", "revenue_ttm", "net_income_ttm",
            "operating_cash_flow_ttm", "total_equity", "shares_outstanding", "ni_change",
            "gross_profit_to_assets", "roe", "accruals_to_assets", "debt_to_assets",
            "asset_growth", "sales_growth", "share_issuance"]
    return f[keep].replace([np.inf, -np.inf], np.nan).reset_index(drop=True)


# ------------------------------------------------------------ panel

def build_panel(m: dict, events: pd.DataFrame, start, end, pf=None, anchor=ANCHOR) -> pd.DataFrame:
    """Stack eligible (signal date, stock) rows with features and labels."""
    A, cal = m["A"], m["calendar"]
    pf = price_feature_matrices(m) if pf is None else pf
    dates = signal_dates(cal, start, end, anchor)
    rows = cal.get_indexer(dates)
    symbols = A.columns

    # Membership for the quarter of the next-session entry.
    entry_dates = cal[rows + 1]
    entry_q = entry_dates.to_period("Q").start_time
    mem = m["membership"]
    member = np.zeros((len(dates), len(symbols)), dtype=bool)
    rank = np.full((len(dates), len(symbols)), np.nan, dtype="float32")
    col = pd.Index(symbols)
    for q, grp in mem.groupby("quarter_start"):
        di = np.flatnonzero(entry_q == q)
        if not len(di):
            continue
        assert (grp.selection_date.max() <= dates[di].min()), "membership not yet known"
        cj = col.get_indexer(grp.symbol)
        ok = cj >= 0
        member[np.ix_(di, cj[ok])] = True
        rank[np.ix_(di, cj[ok])] = grp.universe_rank.to_numpy()[ok]

    a_now = A.to_numpy()[rows]
    with np.errstate(invalid="ignore"):
        eligible = (member & np.isfinite(a_now) & np.isfinite(pf["vol_63"].to_numpy()[rows])
                    & (np.exp(pf["log_dollar_vol"].to_numpy()[rows]) >= MIN_DOLLAR_VOL)
                    & (pf["max_abs_logret_252"].to_numpy()[rows] <= MAX_DAILY_MOVE))
    di, sj = np.nonzero(eligible)
    panel = pd.DataFrame({"date": dates[di], "symbol": symbols[sj],
                          "universe_rank": rank[di, sj]})
    for name in PRICE_FEATURES:
        panel[name] = pf[name].to_numpy()[rows][di, sj]
    panel["close"] = m["C"].to_numpy()[rows][di, sj]

    # Labels: entry at t+1 close, exit at first quote on/after entry + 90 days.
    A_np = A.to_numpy()
    entry_px = A_np[rows + 1][di, sj]
    due = entry_dates + pd.Timedelta(days=HOLDING_DAYS)
    due_row = cal.searchsorted(due)                    # first session on/after due date
    idx = pd.DataFrame(np.where(np.isfinite(A_np), np.arange(len(cal))[:, None], np.nan))
    next_quote_row = idx.bfill(limit=MAX_EXIT_DELAY).to_numpy()
    n = len(cal)
    in_range = due_row < n
    exit_row = np.full(len(di), np.nan)
    exit_row[in_range[di]] = next_quote_row[due_row[di][in_range[di]], sj[in_range[di]]]
    has_exit = np.isfinite(exit_row) & np.isfinite(entry_px)
    exit_px = np.full(len(di), np.nan)
    er = exit_row[has_exit].astype(int)
    exit_px[has_exit] = A_np[er, sj[has_exit]]
    panel["entry_date"] = entry_dates[di]
    panel["label_end"] = pd.NaT
    panel.loc[has_exit, "label_end"] = cal[er]
    panel["ret_90"] = exit_px / entry_px - 1
    # A label becomes usable for training only after the label_end close; if the
    # exit is still unknown it is "censored" and never used in a fit.
    del idx, next_quote_row

    panel = panel.merge(m["universe"], on="symbol", how="left")
    panel["sector"] = panel.sector.fillna("Unknown")
    panel = add_fundamentals(panel, events)
    panel["target"] = panel.groupby("date").ret_90.rank(pct=True)
    return rank_features(panel)


def add_fundamentals(panel: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    panel = pd.merge_asof(panel.sort_values("date"), events.sort_values("available_date"),
                          left_on="date", right_on="available_date", by="symbol",
                          direction="backward")
    assert (panel.available_date.isna() | (panel.available_date <= panel.date)).all()
    stale = (panel.date - panel.fiscal_date).dt.days > 550
    fund_cols = [c for c in events.columns if c not in ("symbol", "fiscal_date", "available_date")]
    panel.loc[stale, fund_cols] = np.nan
    mcap = panel.close * panel.shares_outstanding  # same split basis (data guide caveat)
    mcap = mcap.where(mcap > 0)
    panel["log_mcap"] = np.log(mcap)
    panel["earnings_yield"] = panel.net_income_ttm / mcap
    panel["book_to_market"] = panel.total_equity / mcap
    panel["sales_to_price"] = panel.revenue_ttm / mcap
    panel["cashflow_to_price"] = panel.operating_cash_flow_ttm / mcap
    panel["earnings_change_to_price"] = panel.ni_change / mcap
    panel["filing_age_days"] = (panel.date - panel.available_date).dt.days.astype("float32")
    panel = panel.replace([np.inf, -np.inf], np.nan)
    drop = [c for c in fund_cols if c not in FEATURES] + ["available_date", "close"]
    return panel.drop(columns=drop).sort_values(["date", "symbol"]).reset_index(drop=True)


def rank_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Replace each feature by its percentile rank among the same date's eligible
    stocks. Uses only same-date information, so nothing is learned from the future.
    Raw dollar volume is kept for the liquidity screen."""
    panel["dollar_vol_raw"] = np.exp(panel.log_dollar_vol)
    ranks = panel.groupby("date")[FEATURES].rank(pct=True).astype("float32")
    panel[FEATURES] = ranks
    return panel
