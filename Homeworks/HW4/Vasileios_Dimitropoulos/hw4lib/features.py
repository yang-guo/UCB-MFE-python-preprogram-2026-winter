"""Causal features, dated eligibility and 90-calendar-day labels.

Timing convention (same as the lecture examples):

* features and scores are computed **after the close of the signal date t**;
* the purchase fills at the **next market session's close** (the entry date);
* the label is the adjusted-price return from that entry close to the first
  observed close on/after entry + 90 calendar days (at most 5 sessions late).

Only ``attach_labels`` looks at future prices.  Its columns (``target``,
``label_end``, ``exit_price``...) are never model inputs and never used to
choose which stocks are eligible.

Adapted ideas (cited): trailing-price features, filing-date availability
(filing date + 1 day), the "late old fiscal period does not replace a newer
one" rule, the 550-day staleness rule, next-session membership and the label
definition all follow ``Lectures/modeling/course_data.py``.  The vectorised wide
implementation, TTM/YoY fundamentals, valuation ratios, liquidity screen and
cross-sectional rank transform are my own additions.
"""

import numpy as np
import pandas as pd

PRICE_FEATURES = ["mom_5", "mom_21", "mom_63", "mom_126", "mom_252_21", "vol_63", "vol_252",
                  "max_ret_21", "drawdown_252", "beta_252", "log_dollar_volume"]
FUND_FEATURES = ["gp_to_assets", "roa", "roe", "op_margin", "accruals", "fcf_to_assets",
                 "debt_to_assets", "cash_to_assets", "asset_growth", "sales_growth", "net_buyback"]
VALUE_FEATURES = ["log_mcap", "earnings_yield", "book_to_price", "sales_to_price", "fcf_yield"]
NUMERIC_FEATURES = PRICE_FEATURES + FUND_FEATURES + VALUE_FEATURES + ["filing_age_days"]
FEATURES = NUMERIC_FEATURES + ["sector"]
SUSPECT_JUMP = 3.0   # a quote more than x3 or /3 from the previous one is treated as suspect


# --------------------------------------------------------------------------- schedule
def weekly_signal_dates(calendar, start, end, extra=()):
    """First market session of each calendar week, plus any explicitly added dates.

    The purchase happens at the next session's close, so a Monday signal buys
    on Tuesday.  Why the *first* session: 90 calendar days = 12 weeks + 6 days,
    so a Tuesday purchase matures on Monday; its sale proceeds are then in the
    Monday-close snapshot used to plan Tuesday's purchases (with a Friday
    signal / Monday purchase the proceeds would sit idle for a whole week).
    ``extra`` lets the final test add 2016-03-31 so that the first purchase
    happens at the first test close (2016-04-01), like the benchmark.
    """
    cal = pd.Series(calendar, index=calendar)
    window = cal[(cal >= pd.Timestamp(start)) & (cal <= pd.Timestamp(end))]
    weekly = window.groupby(window.index.to_period("W-SUN")).min()
    dates = pd.DatetimeIndex(sorted(set(weekly) | {pd.Timestamp(d) for d in extra}))
    # A signal on the very last session has no next session to trade on.
    return dates[dates < calendar[-1]]


# --------------------------------------------------------------------------- prices
def price_feature_frames(md):
    """Trailing price features as wide frames (sessions x symbols). No filling."""
    adj, close, volume = md.adj, md.close, md.volume
    ret = adj.pct_change(fill_method=None)
    f = {}
    f["mom_5"] = adj / adj.shift(5) - 1           # short-term reversal
    f["mom_21"] = adj / adj.shift(21) - 1
    f["mom_63"] = adj / adj.shift(63) - 1
    f["mom_126"] = adj / adj.shift(126) - 1
    f["mom_252_21"] = adj.shift(21) / adj.shift(252) - 1   # 12-1 month momentum
    f["vol_63"] = ret.rolling(63, min_periods=63).std() * np.sqrt(252)
    # Screen only (not a model input): a pending cash acquisition pins the price,
    # so its 21-session volatility collapses.  Used by the post-test revision.
    f["vol_21"] = ret.rolling(21, min_periods=21).std() * np.sqrt(252)
    f["vol_252"] = ret.rolling(252, min_periods=200).std() * np.sqrt(252)
    f["max_ret_21"] = ret.rolling(21, min_periods=21).max()  # "lottery" effect
    f["drawdown_252"] = adj / adj.rolling(252, min_periods=200).max() - 1
    # Beta to the S&P 500 TR on days where both returns exist.
    m = md.benchmark.pct_change()
    y = ret.notna().mul(m, axis=0).where(ret.notna())
    mean_x = ret.rolling(252, min_periods=200).mean()
    mean_y = y.rolling(252, min_periods=200).mean()
    cov = (ret * y).rolling(252, min_periods=200).mean() - mean_x * mean_y
    var = (y * y).rolling(252, min_periods=200).mean() - mean_y ** 2
    f["beta_252"] = cov / var.where(var > 1e-12)
    # Liquidity uses *raw* close x raw volume (not future-adjusted prices).
    dollar_volume = (close * volume).where(volume > 0)
    f["dollar_volume_63"] = dollar_volume.rolling(63, min_periods=40).mean()
    f["log_dollar_volume"] = np.log(f["dollar_volume_63"])
    # Lecture rule: 63 complete trailing daily returns are required to predict.
    f["complete_63"] = ret.notna().rolling(63, min_periods=63).sum().eq(63)
    # Data-quality screen: some tickers interleave two different price series
    # (e.g. AAM jumps 25,700 -> 0.19 -> 26,000).  Flag any move larger than x3
    # (or /3) versus the previous observed quote in the trailing 252 sessions.
    jump = np.log(adj.ffill()).diff().abs().where(adj.notna()) > np.log(SUSPECT_JUMP)
    f["clean_252"] = ~jump.rolling(252, min_periods=1).max().astype(bool)
    return f


# --------------------------------------------------------------------------- fundamentals
def fundamental_events(fundamentals, extra_lag_days=0):
    """Quarterly records -> availability-dated feature events.

    TTM sums use the four most recent fiscal quarters (spanning <= 300 days);
    YoY comparisons use the record four quarters earlier (330-400 days back).
    A record becomes usable from ``filing_date + 1 day`` (filing dates carry no
    time of day).  Records with filing_date <= fiscal_date are untrusted and are
    not used as events.
    """
    f = fundamentals.loc[fundamentals.period.eq("quarter")].copy()
    f = f.sort_values(["symbol", "fiscal_date", "filing_date"]).drop_duplicates(
        ["symbol", "fiscal_date"], keep="last")
    g = f.groupby("symbol", sort=False)

    span3 = (f.fiscal_date - g.fiscal_date.shift(3)).dt.days
    ttm_ok = span3.le(300)
    for col in ["revenue", "gross_profit", "operating_income", "net_income",
                "operating_cash_flow", "free_cash_flow"]:
        roll = g[col].rolling(4, min_periods=4).sum().reset_index(level=0, drop=True)
        f[col + "_ttm"] = roll.where(ttm_ok)
    span4 = (f.fiscal_date - g.fiscal_date.shift(4)).dt.days
    yoy_ok = span4.between(330, 400)
    prev_assets = g.total_assets.shift(4).where(yoy_ok)
    prev_revenue = g.revenue.shift(4).where(yoy_ok)
    buyback = f.common_stock_repurchased.abs() - f.common_stock_issued.abs()
    buyback_ttm = buyback.groupby(f.symbol).rolling(4, min_periods=4).sum().reset_index(level=0, drop=True)

    assets = f.total_assets.where(f.total_assets.gt(0))
    equity = f.total_equity.where(f.total_equity.gt(0))
    rev_ttm = f.revenue_ttm.where(f.revenue_ttm.gt(0))
    f["gp_to_assets"] = f.gross_profit_ttm / assets
    f["roa"] = f.net_income_ttm / assets
    f["roe"] = f.net_income_ttm / equity
    f["op_margin"] = f.operating_income_ttm / rev_ttm
    f["accruals"] = (f.net_income_ttm - f.operating_cash_flow_ttm) / assets
    f["fcf_to_assets"] = f.free_cash_flow_ttm / assets
    f["debt_to_assets"] = f.total_debt / assets
    f["cash_to_assets"] = f.cash_and_equivalents / assets
    f["asset_growth"] = assets / prev_assets.where(prev_assets.gt(0)) - 1
    f["sales_growth"] = f.revenue / prev_revenue.where(prev_revenue.gt(0)) - 1
    f["net_buyback"] = buyback_ttm.where(ttm_ok) / assets
    # Kept for valuation ratios, which need the price at the signal date.
    f["equity_book"] = f.total_equity

    f = f.loc[f.filing_date.gt(f.fiscal_date)].copy()
    f["available_date"] = f.filing_date + pd.Timedelta(days=1 + extra_lag_days)
    keep = FUND_FEATURES + ["net_income_ttm", "revenue_ttm", "free_cash_flow_ttm", "equity_book"]
    f[keep] = f[keep].replace([np.inf, -np.inf], np.nan)
    f = f.sort_values(["symbol", "available_date", "fiscal_date"])
    # A late old fiscal period must not replace a newer known period (lecture rule).
    latest = f.groupby("symbol").fiscal_date.cummax()
    f = f.loc[f.fiscal_date.eq(latest)].drop_duplicates(["symbol", "available_date"], keep="last")
    return f[["symbol", "fiscal_date", "filing_date", "available_date"] + keep].reset_index(drop=True)


# --------------------------------------------------------------------------- panel
def build_panel(md, signal_dates, min_price=5.0, min_dollar_volume=2e6):
    """One row per (signal date, eligible stock), with features only.

    Eligibility at signal date t (all information known at t's close):
      * member of the quarterly universe for the quarter of the *entry* session
        (t + 1 session), whose selection date is <= t;
      * a quote at t and 63 complete trailing daily returns;
      * screen: raw close >= $5 and 63-session average raw dollar volume >= $2M;
      * data-quality screen: no x3 (or /3) quote jump in the trailing 252 sessions.
    """
    cal = md.calendar
    t_idx = cal.get_indexer(signal_dates)
    if (t_idx < 0).any() or (t_idx + 1 >= len(cal)).any():
        raise ValueError("Signal dates must be sessions with a following session")
    entry_dates = cal[t_idx + 1]
    sched = pd.DataFrame({"date": signal_dates, "t": t_idx, "entry_date": entry_dates})
    sched["quarter_start"] = sched.entry_date.dt.to_period("Q").dt.start_time.astype("datetime64[ns]")

    mem = md.membership.merge(sched, on="quarter_start", how="inner")
    mem = mem.loc[mem.selection_date.le(mem.date)]
    sym_pos = md.symbols.get_indexer(mem.symbol)
    mem = mem.loc[sym_pos >= 0].assign(s=sym_pos[sym_pos >= 0])

    frames = price_feature_frames(md)
    t, s = mem.t.to_numpy(), mem.s.to_numpy()
    out = mem[["date", "entry_date", "symbol", "universe_rank", "selection_shares"]].reset_index(drop=True)
    out["t"], out["s"] = t, s
    out["adj_close"] = md.adj.to_numpy()[t, s]
    out["close"] = md.close.to_numpy()[t, s]
    for name, frame in frames.items():
        out[name] = frame.to_numpy()[t, s]
    del frames

    out["eligible"] = (out.adj_close.notna() & out.complete_63.astype(bool) & out.clean_252.astype(bool)
                       & out.close.ge(min_price) & out.dollar_volume_63.ge(min_dollar_volume))
    out = out.loc[out.eligible].drop(columns=["complete_63", "clean_252"]).reset_index(drop=True)

    # As-of join of the most recent *available* filing within each stock.
    events = fundamental_events(md.fundamentals)
    out = out.sort_values("date")
    out = pd.merge_asof(out, events.sort_values("available_date"), left_on="date",
                        right_on="available_date", by="symbol", direction="backward")
    assert (out.available_date.isna() | out.available_date.le(out.date)).all()
    out["filing_age_days"] = (out.date - out.available_date).dt.days
    stale = (out.date - out.fiscal_date).dt.days.gt(550)
    fund_cols = FUND_FEATURES + ["net_income_ttm", "revenue_ttm", "free_cash_flow_ttm", "equity_book"]
    out.loc[stale, fund_cols] = np.nan

    # Valuation: point-in-time shares from the universe selection x signal-date raw close.
    mcap = (out.selection_shares * out.close).where(lambda x: x > 0)
    out["log_mcap"] = np.log(mcap)
    out["earnings_yield"] = out.net_income_ttm / mcap
    out["book_to_price"] = out.equity_book / mcap
    out["sales_to_price"] = out.revenue_ttm / mcap
    out["fcf_yield"] = out.free_cash_flow_ttm / mcap
    out[NUMERIC_FEATURES] = out[NUMERIC_FEATURES].replace([np.inf, -np.inf], np.nan)

    sectors = md.universe.set_index("symbol").sector
    out["sector"] = out.symbol.map(sectors).fillna("Unknown")
    return out.sort_values(["date", "symbol"]).reset_index(drop=True)


def attach_labels(panel, md, holding_days=90, max_exit_delay=5):
    """Add the 90-calendar-day forward return.  FUTURE DATA -- never a feature.

    Entry: adjusted close on the next session (must be quoted, else no label).
    Exit: first quoted session on/after entry + 90 calendar days, at most
    ``max_exit_delay`` sessions late, and never after the last loaded session.
    Missing outcomes stay NaN (they are censored, not zero).
    """
    cal = md.calendar
    adj = md.adj.to_numpy()
    p = panel.copy()
    t_entry = p.t.to_numpy() + 1
    s = p.s.to_numpy()
    p["entry_price"] = adj[t_entry, s]
    due = p.entry_date + pd.Timedelta(days=holding_days)
    first = cal.searchsorted(due)                      # first session >= due date
    exit_t = np.full(len(p), -1)
    for k in range(max_exit_delay + 1):
        cand = first + k
        ok = (cand < len(cal)) & (exit_t < 0)
        has = np.zeros(len(p), dtype=bool)
        has[ok] = np.isfinite(adj[cand[ok], s[ok]])
        exit_t[has] = cand[has]
    valid = (exit_t >= 0) & np.isfinite(p.entry_price.to_numpy())
    p["exit_price"] = np.where(valid, adj[np.clip(exit_t, 0, None), s], np.nan)
    p["label_end"] = pd.NaT
    p.loc[valid, "label_end"] = cal[exit_t[valid]]
    p["label_end"] = p.label_end.astype("datetime64[ns]")
    p["target"] = p.exit_price / p.entry_price - 1
    return p
