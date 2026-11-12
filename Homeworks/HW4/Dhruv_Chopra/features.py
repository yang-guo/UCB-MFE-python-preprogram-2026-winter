"""Causal features and 90-calendar-day labels on the wide (date x symbol) arrays.

Every feature at row t uses only prices through the close of t and financial
statements whose filing date + 1 day <= t. Labels (the only place future
prices appear) are computed separately and never enter a feature matrix.

Adapted ideas (cited): the momentum / volatility / drawdown definitions,
"filing date + 1 day" availability rule, "latest fiscal period only" rule, and
the 550-day staleness cut follow Lectures/modeling/course_data.py
(price_features, financial_events, join_financials, make_labels). The
implementation here is vectorised over wide arrays and is my own.
"""

import numpy as np
import pandas as pd

PRICE_FEATURES = [
    "mom_5",            # 1-week return (short-term reversal)
    "mom_21",           # 1-month return (reversal)
    "mom_63",           # 3-month momentum
    "mom_126",          # 6-month momentum
    "mom_252_21",       # 12-1 month momentum (classic Jegadeesh-Titman)
    "vol_63",           # 3-month annualised volatility (low-vol anomaly)
    "max_ret_21",       # largest daily return in a month (lottery / MAX effect)
    "drawdown_252",     # distance from the 52-week high
    "log_dollar_vol_63" # liquidity & size proxy; split-invariant
]
FUNDAMENTAL_FEATURES = [
    "gross_profitability",  # gross profit / assets (Novy-Marx)
    "operating_roa",
    "roe",
    "net_margin",
    "debt_to_assets",
    "cash_to_assets",
    "fcf_to_assets",
    "accruals_to_assets",   # (NI - CFO) / assets (Sloan accruals)
    "revenue_growth_yoy",
    "asset_growth_yoy",     # investment factor
    "filing_age_days",
]
FEATURES = PRICE_FEATURES + FUNDAMENTAL_FEATURES      # v1 model (frozen before the first final test)

# Ideas carried over from my own intraday skew research (skewness_signal repo,
# research/model_selection_report.md), rebuilt on daily data. Tested in
# run_ideas.py on 2010-2015 only; see README "Where the v2 ideas came from".
IDEA_FEATURES = [
    "skew_63",          # skew of daily returns, 3 months  (my long-high-skew prior)
    "skew_252",         # skew of daily returns, 1 year
    "kurt_63",          # excess kurtosis of daily returns, 3 months
    "dist_high_20",     # distance from the 20-day high (my k=20 signal)
    "clv_21",           # 21-day mean close location (C-L)/(H-L)
]

# Further hypotheses of mine (tested in run_ideas.py --group ..., 2010-2015 only).
SKEW_180 = ["skew_180"]                 # skew of the previous 180 daily returns
DIVIDEND_FEATURES = [
    "div_payer",            # paid any dividend over the last four quarters
    "div_to_assets",        # TTM dividends / assets
    "div_payout_cfo",       # TTM dividends / TTM operating cash flow, clipped to [0, 2]
    "div_change_yoy",       # TTM vs prior-year TTM; initiation +1, omission -1
]
ISSUANCE_FEATURES = [
    "net_issuance_yoy",     # log change in shares outstanding over a year
    "issued_to_assets",     # TTM common stock issued / assets
    "buyback_to_assets",    # TTM common stock repurchased / assets
    "net_payout_to_assets", # (dividends + buybacks - issuance), TTM / assets
]
DECOMP_FEATURES = [
    "beta_mkt_252",         # market beta from the market + industry regression
    "resid_mom_252_21",     # stock-specific return, 12 months to 1 month ago
    "ind_mom_126",          # industry return in excess of the market, 6 months
    "idio_vol_252",         # volatility of the stock-specific residual
    "r2_252",               # share of variance explained by market + industry
]
HEALTH_FEATURES = [
    "operating_margin",     # operating income / revenue
    "cash_to_ltd",          # cash / (cash + long-term debt); 1 = no long-term debt
    "net_cash_to_assets",   # (cash - total debt) / assets
    "current_ratio",        # current assets / current liabilities, clipped at 10
    "interest_coverage",    # operating income / interest expense, clipped to [-20, 50]
    "st_debt_share",        # short-term debt / (short + long-term debt); 0 = no debt
]
LONG_MOM_FEATURES = [
    "avg_ret_252",          # mean daily return over the last year (incl. last month)
    "mom_504_252",          # return from 2 years ago to 1 year ago (month-ends)
    "mom_756_252",          # return from 3 years ago to 1 year ago (month-ends)
]
MOMENTS_FEATURES = [       # the four moments of the past year's daily returns
    "avg_ret_252", "vol_252", "skew_252", "kurt_252",
]
RESID_SQ_FEATURES = [     # squared residual returns after removing market + sector effects
    "sec_resid_sq_21", "sec_resid_sq_63", "sec_resid_sq_252", "sec_resid_sq_ratio_21_252",
]
FEATURE_GROUPS = {"skew": IDEA_FEATURES, "skew_180": SKEW_180, "dividends": DIVIDEND_FEATURES,
                  "issuance": ISSUANCE_FEATURES, "decomposition": DECOMP_FEATURES,
                  "health": HEALTH_FEATURES, "long_mom": LONG_MOM_FEATURES, "moments": MOMENTS_FEATURES,
                  "resid_sq": RESID_SQ_FEATURES}
FINANCIAL_EXTRA = DIVIDEND_FEATURES + ISSUANCE_FEATURES + HEALTH_FEATURES
# Every panel column, once (some features belong to two groups).
PANEL_FEATURES = list(dict.fromkeys(FEATURES + [f for g in FEATURE_GROUPS.values() for f in g]))

HOLDING_DAYS = 90
MAX_EXIT_DELAY = 5     # sessions an exit may slip if the stock has no quote at maturity
STALE_FISCAL_DAYS = 550


def _frame(a, data):
    return pd.DataFrame(a, index=data.calendar, columns=data.symbols)


def price_features(data):
    """Return {name: float32 array [T, N]}; gaps stay NaN (no price filling)."""
    px = _frame(data.adj_close, data).astype("float64")
    ret = px.pct_change(fill_method=None)
    out = {}
    for k in (5, 21, 63, 126):
        out[f"mom_{k}"] = px.pct_change(k, fill_method=None)
    out["mom_252_21"] = px.shift(21) / px.shift(252) - 1
    out["vol_63"] = ret.rolling(63, min_periods=63).std() * np.sqrt(252)
    out["max_ret_21"] = ret.rolling(21, min_periods=21).max()
    out["drawdown_252"] = px / px.rolling(252, min_periods=200).max() - 1
    dv = _frame(data.dollar_volume, data).astype("float64")
    out["log_dollar_vol_63"] = np.log1p(dv.rolling(63, min_periods=40).mean())
    out["adv_63"] = dv.rolling(63, min_periods=40).mean()   # used for the liquidity screen, not as a feature
    out.update(idea_features(px, ret, data))
    out.update(idea_features(px, ret, data, skew_windows=(180,), kurt_windows=(252,), high_windows=(),
                             clv_windows=()))
    out["vol_252"] = ret.rolling(252, min_periods=_min_obs(252)).std() * np.sqrt(252)
    return {k: v.to_numpy(np.float32) for k, v in out.items()}


def idea_features(px, ret, data, skew_windows=(63, 252), kurt_windows=(63,),
                  high_windows=(20,), clv_windows=(21,)):
    """Daily versions of my skew-research signals. All windows end at the close
    of t. pandas' rolling skew/kurt are the bias-adjusted estimators, i.e.
    scipy.stats.skew(bias=False) and kurtosis(fisher=True, bias=False)."""
    out = {}
    for k in skew_windows:
        out[f"skew_{k}"] = ret.rolling(k, min_periods=_min_obs(k)).skew()
    for k in kurt_windows:
        out[f"kurt_{k}"] = ret.rolling(k, min_periods=_min_obs(k)).kurt()
    for k in high_windows:
        out[f"dist_high_{k}"] = px / px.rolling(k, min_periods=_min_obs(k)).max() - 1
    clv = getattr(data, "close_location", None)
    clv = _frame(clv, data).astype("float64") if clv is not None else px * np.nan
    for k in clv_windows:
        out[f"clv_{k}"] = clv.rolling(k, min_periods=_min_obs(k)).mean()
    return out


def _min_obs(k):
    """Full window for short lookbacks (as vol_63); ~80% for long ones (as drawdown_252)."""
    return k if k <= 63 else int(0.8 * k)


def financial_events(fundamentals):
    """One row per (symbol, availability date) with ratio features.

    Rules (from course_data.financial_events): quarterly rows only; filing
    must be after fiscal end (invalid lags excluded); usable from filing+1
    day; a late, older fiscal period never replaces a newer known one.
    """
    f = fundamentals.loc[fundamentals.filing_date.gt(fundamentals.fiscal_date)].copy()
    f = f.sort_values(["symbol", "fiscal_date", "filing_date"]).drop_duplicates(["symbol", "fiscal_date"], keep="last")
    f["available_date"] = f.filing_date + pd.Timedelta(days=1)
    assets = f.total_assets.where(f.total_assets.gt(0))
    revenue = f.revenue.where(f.revenue.gt(0))
    equity = f.total_equity.where(f.total_equity.gt(0))
    f["gross_profitability"] = f.gross_profit / assets
    f["operating_roa"] = f.operating_income / assets
    f["roe"] = f.net_income / equity
    f["net_margin"] = f.net_income / revenue
    f["debt_to_assets"] = f.total_debt / assets
    f["cash_to_assets"] = f.cash_and_equivalents / assets
    f["fcf_to_assets"] = f.free_cash_flow / assets
    f["accruals_to_assets"] = (f.net_income - f.operating_cash_flow) / assets

    # Year-over-year growth versus the same fiscal quarter one year earlier.
    # The earlier quarter was necessarily filed before the current one is used
    # (we only compare to rows with an earlier fiscal date in the same table
    # and require that it was available by the current availability date).
    g = f.groupby("symbol", sort=False)
    prev_fiscal = g.fiscal_date.shift(4)
    prev_avail = g.available_date.shift(4)
    gap = (f.fiscal_date - prev_fiscal).dt.days
    same_quarter = gap.between(350, 380) & prev_avail.le(f.available_date)
    prev_rev = g.revenue.shift(4).where(lambda s: s > 0)
    prev_assets = g.total_assets.shift(4).where(lambda s: s > 0)
    f["revenue_growth_yoy"] = (revenue / prev_rev - 1).where(same_quarter)
    f["asset_growth_yoy"] = (assets / prev_assets - 1).where(same_quarter)

    f = _payout_features(f, g, assets, same_quarter)
    f = _health_features(f, assets, revenue)

    ratio_cols = [c for c in FUNDAMENTAL_FEATURES + FINANCIAL_EXTRA if c != "filing_age_days"]
    f[ratio_cols] = f[ratio_cols].replace([np.inf, -np.inf], np.nan)
    f = f.sort_values(["symbol", "available_date", "fiscal_date"])
    latest = f.groupby("symbol").fiscal_date.cummax()
    f = f.loc[f.fiscal_date.eq(latest)].drop_duplicates(["symbol", "available_date"], keep="last")
    return f[["symbol", "fiscal_date", "available_date"] + ratio_cols]


def _ttm(g, col_frame, f, first_shift):
    """Sum of four consecutive quarters ending `first_shift` filings back.

    Valid only when the four rows span one year of fiscal quarters (gap of
    250-300 days between the first and the fourth) and every one of them was
    available by the current row's availability date (point in time).
    """
    parts = [col_frame.groupby(f.symbol, sort=False).shift(first_shift + k) if first_shift + k else col_frame
             for k in range(4)]
    total = sum(parts)
    fiscal = [g.fiscal_date.shift(first_shift + k) if first_shift + k else f.fiscal_date for k in range(4)]
    avail = [g.available_date.shift(first_shift + k) if first_shift + k else f.available_date for k in range(4)]
    span = (fiscal[0] - fiscal[3]).dt.days
    ok = span.between(250, 300)
    for a in avail:
        ok &= a.le(f.available_date)
    return total.where(ok)


def dividend_quarters(f, g):
    """Dividends paid per quarter (positive), with the data-hole rule.

    The vendor's dividends_paid is often blank in a filing that does have a
    cash-flow statement. Checked against the dividends table, a blank after a
    paying quarter is usually a data hole (345 cases), while a blank after a
    non-paying quarter is a true zero (3,025 cases). So, using only filings
    already known:
      * reported -> that amount;
      * blank, no cash-flow statement -> unknown (NaN);
      * blank after a paying quarter -> previous amount (hole), *unless* the
        next quarter is also blank, which confirms an omission -> 0;
      * blank after a non-paying quarter -> 0.
    Returns a function q(k) giving quarter (current - k) as known at each row.
    """
    raw = (-f.dividends_paid).clip(lower=0)                # cash-flow sign: paid < 0
    blank = f.dividends_paid.isna() & f.operating_cash_flow.notna()
    no_cf = f.dividends_paid.isna() & f.operating_cash_flow.isna()
    prev_raw = g.dividends_paid.shift(1).mul(-1).clip(lower=0)
    prev_paid = prev_raw.gt(0)

    def q(k):
        sh = (lambda x: x.groupby(f.symbol, sort=False).shift(k)) if k else (lambda x: x)
        v_raw, v_blank, v_nocf = sh(raw), sh(blank).astype("boolean"), sh(no_cf).astype("boolean")
        v_prev, v_prev_paid = sh(prev_raw), sh(prev_paid).astype("boolean")
        if k:   # the following quarter is known at this row: did it confirm an omission?
            nxt_blank = (blank.groupby(f.symbol, sort=False).shift(k - 1) if k > 1 else blank).astype("boolean")
        else:
            nxt_blank = pd.Series(False, index=f.index, dtype="boolean")
        out = v_raw.copy()
        hole = v_blank & v_prev_paid & ~nxt_blank
        out = out.mask(hole.fillna(False).astype(bool), v_prev)
        out = out.mask((v_blank & ~hole).fillna(False).astype(bool), 0.0)
        out = out.mask(v_nocf.fillna(True).astype(bool) & v_raw.isna(), np.nan)
        return out
    return q


def _payout_features(f, g, assets, same_quarter):
    """Dividend and share-issuance features (my hypotheses; see README)."""
    q = dividend_quarters(f, g)
    quarters = [q(k) for k in range(8)]
    fiscal = [f.fiscal_date] + [g.fiscal_date.shift(k) for k in range(1, 8)]
    avail = [f.available_date] + [g.available_date.shift(k) for k in range(1, 8)]

    def window(first):
        ok = (fiscal[first] - fiscal[first + 3]).dt.days.between(250, 300)
        for k in range(first, first + 4):
            ok &= avail[k].le(f.available_date)
        return sum(quarters[first:first + 4]).where(ok)

    div = window(0)
    div_prev = window(4).where((fiscal[0] - fiscal[4]).dt.days.between(350, 380))
    f["div_payer"] = div.gt(0).astype(float).where(div.notna())
    f["div_to_assets"] = div / assets
    cfo = _ttm(g, f.operating_cash_flow, f, 0)
    payout = (div / cfo).where(cfo > 0, 2.0).clip(0, 2)
    f["div_payout_cfo"] = payout.where(div.gt(0), 0.0).where(div.notna() & cfo.notna())
    change = (div / div_prev - 1).clip(-1, 1)
    change = change.where(div_prev.gt(0), np.where(div.gt(0), 1.0, 0.0))   # initiation / never paid
    f["div_change_yoy"] = change.where(div.notna() & div_prev.notna())

    shares = f.shares_outstanding.where(f.shares_outstanding.gt(0))
    prev_shares = g.shares_outstanding.shift(4).where(lambda s: s > 0)
    f["net_issuance_yoy"] = np.log(shares / prev_shares).where(same_quarter)
    issued = _ttm(g, f.common_stock_issued.clip(lower=0), f, 0)
    bought = _ttm(g, (-f.common_stock_repurchased).clip(lower=0), f, 0)
    f["issued_to_assets"] = issued / assets
    f["buyback_to_assets"] = bought / assets
    f["net_payout_to_assets"] = (div + bought - issued) / assets
    return f


def _health_features(f, assets, revenue):
    """Balance-sheet health ratios from the latest filing (my hypotheses)."""
    cash = f.cash_and_equivalents.where(f.cash_and_equivalents.ge(0))
    ltd = f.long_term_debt.where(f.long_term_debt.ge(0))
    std = f.short_term_debt.where(f.short_term_debt.ge(0))
    f["operating_margin"] = f.operating_income / revenue
    f["cash_to_ltd"] = (cash / (cash + ltd)).where((cash + ltd) > 0)
    f["net_cash_to_assets"] = (cash - f.total_debt) / assets
    tcl = f.total_current_liabilities.where(f.total_current_liabilities.gt(0))
    f["current_ratio"] = (f.total_current_assets / tcl).clip(upper=10)
    ie = f.interest_expense.where(f.interest_expense.ge(0))
    cover = (f.operating_income / ie.where(ie > 0)).clip(-20, 50)
    f["interest_coverage"] = cover.where(ie.ne(0), 50.0).where(ie.notna() & f.operating_income.notna())
    debt = std + ltd
    f["st_debt_share"] = (std / debt).where(debt > 0, 0.0).where(debt.notna())
    return f


def long_momentum(data, lags=((24, 12), (36, 12)), avg_windows=(252,)):
    """Long-horizon momentum.

    avg_ret_k: mean daily return over the last k sessions.
    mom_{a}_{b}: return between month-ends a and b months before the latest
    *completed* month (a month counts once its calendar end is <= t), using
    month-end prices: cleaned daily prices inside the window, earlier
    month-ends from data.early_monthly.
    """
    px = _frame(data.adj_close, data).astype("float64")
    ret = px.pct_change(fill_method=None)
    out = {}
    for k in avg_windows:
        out[f"avg_ret_{k}"] = ret.rolling(k, min_periods=_min_obs(k)).mean().to_numpy(np.float32)
    monthly = px.resample("ME").last()
    early = getattr(data, "early_monthly", None)
    if early is not None and len(early):
        monthly = pd.concat([early.reindex(columns=data.symbols).loc[lambda e: e.index < monthly.index[0]],
                             monthly])
    m_px = monthly.to_numpy()
    # Latest completed month for every session (label <= t).
    m_idx = monthly.index.searchsorted(data.calendar, side="right") - 1
    for a, b in lags:
        ia, ib = m_idx - a, m_idx - b
        ok = ia >= 0
        num = np.where(ok[:, None], m_px[np.clip(ib, 0, None)], np.nan)
        den = np.where(ok[:, None], m_px[np.clip(ia, 0, None)], np.nan)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[f"mom_{a * 21}_{b * 21}"] = (num / den - 1).astype(np.float32)
    return out


def fundamental_features(data):
    """As-of (backward) join of financial events onto every session."""
    events = financial_events(data.fundamentals)
    T, N = len(data.calendar), len(data.symbols)
    # First session on/after the availability date: the feature is usable at that close.
    t = data.calendar.searchsorted(events.available_date.to_numpy(), side="left")
    n = data.symbols.get_indexer(events.symbol)
    keep = (t < T) & (n >= 0)
    # Events available before the window start land on row 0 (they are known by then).
    events, t, n = events.loc[keep], t[keep], n[keep]
    # Several events may map to the same session; keep the latest available one.
    order = np.lexsort((events.available_date.to_numpy(), t, n))
    events, t, n = events.iloc[order], t[order], n[order]
    last_in_cell = np.r_[(t[1:] != t[:-1]) | (n[1:] != n[:-1]), True]
    events, t, n = events.iloc[last_in_cell], t[last_in_cell], n[last_in_cell]

    # Carry forward *which event* is current, then read every field from that
    # same event. (Forward-filling each column separately would let an older
    # filing's value survive a newer filing in which that field is missing.)
    current = np.full((T, N), np.nan)
    current[t, n] = np.arange(len(events))
    current = pd.DataFrame(current).ffill().to_numpy()
    known = np.isfinite(current)
    row = np.where(known, current, 0).astype(np.int64)

    def as_of(values):
        values = np.asarray(values, dtype=float)
        return np.where(known, values[row], np.nan)

    fiscal = as_of(events.fiscal_date.to_numpy().astype("datetime64[D]").astype(float))
    avail = as_of(events.available_date.to_numpy().astype("datetime64[D]").astype(float))
    today = data.calendar.to_numpy().astype("datetime64[D]").astype(float)[:, None]
    stale = (today - fiscal) > STALE_FISCAL_DAYS
    out = {}
    for column in [c for c in FUNDAMENTAL_FEATURES + FINANCIAL_EXTRA if c != "filing_age_days"]:
        a = as_of(events[column].to_numpy(dtype=float))
        a[stale | ~known] = np.nan
        out[column] = a.astype(np.float32)
    age = today - avail
    age[stale] = np.nan
    out["filing_age_days"] = age.astype(np.float32)
    assert np.nanmin(age) >= 0, "A financial feature was used before it was available"
    return out


def labels(data):
    """Gross 90-calendar-day return for a signal at t: buy at close t+1,
    sell at the first quoted close on/after entry + 90 days (<= 5 sessions late).

    Returns (target, label_end_index). label_end_index is the session index of
    the exit quote; a training row is usable at fit date F only if
    calendar[label_end] < F. Censored rows are NaN (never filled).
    """
    px = data.adj_close
    T, N = px.shape
    cal = data.calendar
    entry = np.arange(T) + 1
    entry_ok = entry < T
    due = np.full(T, T, dtype=np.int64)
    due[entry_ok] = cal.searchsorted(cal[entry[entry_ok]] + pd.Timedelta(days=HOLDING_DAYS), side="left")

    # Index of the first quote at or after each session, if within MAX_EXIT_DELAY sessions.
    idx = np.where(np.isfinite(px), np.arange(T, dtype=np.float32)[:, None], np.nan)
    next_quote = pd.DataFrame(idx).bfill(limit=MAX_EXIT_DELAY).to_numpy()

    target = np.full((T, N), np.nan, dtype=np.float32)
    label_end = np.full((T, N), np.nan, dtype=np.float32)
    rows = np.flatnonzero(entry_ok & (due < T))
    exit_idx = next_quote[due[rows]]                       # [rows, N]
    entry_px = px[entry[rows]]
    have = np.isfinite(exit_idx) & np.isfinite(entry_px)
    exit_px = np.where(have, px[np.nan_to_num(exit_idx, nan=0).astype(int), np.arange(N)], np.nan)
    target[rows] = np.where(have, exit_px / entry_px - 1, np.nan)
    label_end[rows] = np.where(have, exit_idx, np.nan)
    return target, label_end


def peer_returns(ret, member, groups, min_peers=5, fallback=None):
    """Equal-weight leave-one-out return of each stock's peer group.

    ret [T, N] daily returns (NaN = none), member [T, N] universe members,
    groups / fallback: group label per stock. Peers are members with a return
    that day, excluding the stock itself. Where fewer than `min_peers` peers
    exist, the fallback grouping (e.g. sector) is used.
    """
    def loo(labels):
        codes, uniq = pd.factorize(pd.Series(labels))
        G = np.zeros((len(codes), len(uniq)))
        G[np.arange(len(codes)), codes] = 1.0
        use = member & np.isfinite(ret)
        r = np.where(use, ret, 0.0)
        S, C = r @ G, use.astype(float) @ G                  # [T, groups]
        own_s, own_c = S[:, codes], C[:, codes]               # back to [T, N]
        peers = own_c - use
        with np.errstate(invalid="ignore", divide="ignore"):
            out = (own_s - r) / peers
        return np.where(peers >= min_peers, out, np.nan)

    out = loo(groups)
    if fallback is not None:
        out = np.where(np.isfinite(out), out, loo(fallback))
    return out


def decomposition_features(data, window=252, resid_skip=21, ind_window=126, peers="industry",
                           resid_sq_windows=()):
    """Split each stock's daily return into market, industry and own parts.

    For every stock and session t, regress its daily returns over the
    trailing `window` sessions (ending at t) on
        x1 = ^SP500TR return (market), and
        x2 = industry peer return - market return (industry, net of market).
    Using x2 net of the market keeps the two regressors nearly uncorrelated.
    The fit uses rolling sums, so every number at t uses data through t only.

    peers="sector" uses the stock's sector instead of its industry (names get
    a "sec_" prefix). resid_sq_windows=(m, ...) adds the mean *squared*
    residual over the last m sessions, using the coefficients fitted at t,
    plus the full-window mean and the ratio of the shortest window to it.
    """
    px = _frame(data.adj_close, data).astype("float64")
    y_all = px.pct_change(fill_method=None).to_numpy()
    del px
    rm = data.benchmark.pct_change().to_numpy()
    sectors = data.sector.reindex(data.symbols).fillna("Unknown").to_numpy()
    if peers == "sector":
        x2_all = peer_returns(y_all, data.member, sectors)
    else:
        groups = data.industry.reindex(data.symbols).fillna("Unknown").to_numpy()
        x2_all = peer_returns(y_all, data.member, groups, fallback=sectors)
    pre = "sec_" if peers == "sector" else ""
    x2_all -= rm[:, None]
    T, N = y_all.shape
    names = [pre + n_ for n_ in (f"beta_mkt_{window}", f"r2_{window}", f"idio_vol_{window}",
                                 f"resid_mom_{window}_{resid_skip}", f"ind_mom_{ind_window}")]
    sq_names = {m: f"{pre}resid_sq_{m}" for m in resid_sq_windows}
    if resid_sq_windows:
        sq_names[window] = f"{pre}resid_sq_{window}"
        names += list(dict.fromkeys(sq_names.values()))
        ratio_name = f"{pre}resid_sq_ratio_{min(resid_sq_windows)}_{window}"
        names.append(ratio_name)
    out = {k_: np.full((T, N), np.nan, dtype=np.float32) for k_ in names}
    k = window - resid_skip

    def lagged(a_):
        sh = np.full_like(a_, np.nan)
        sh[resid_skip:] = a_[:-resid_skip]
        return sh

    for lo in range(0, N, 400):           # chunks of stocks keep memory bounded
        cols = slice(lo, min(lo + 400, N))
        y, x2 = y_all[:, cols], x2_all[:, cols]
        x1 = np.broadcast_to(rm[:, None], y.shape)
        ok = np.isfinite(y) & np.isfinite(x2) & np.isfinite(x1)

        def rsum(a, w):
            c = np.cumsum(np.where(ok, a, 0.0), axis=0)
            r = c.copy()
            r[w:] = c[w:] - c[:-w]
            return r

        one = np.ones_like(y)
        n = rsum(one, window)
        s1, s2, sy = rsum(x1, window), rsum(x2, window), rsum(y, window)
        with np.errstate(invalid="ignore", divide="ignore"):
            c11 = rsum(x1 * x1, window) - s1 * s1 / n
            c22 = rsum(x2 * x2, window) - s2 * s2 / n
            c12 = rsum(x1 * x2, window) - s1 * s2 / n
            c1y = rsum(x1 * y, window) - s1 * sy / n
            c2y = rsum(x2 * y, window) - s2 * sy / n
            cyy = rsum(y * y, window) - sy * sy / n
            det = c11 * c22 - c12 * c12
            b1 = (c22 * c1y - c12 * c2y) / det
            b2 = (c11 * c2y - c12 * c1y) / det
            a = (sy - b1 * s1 - b2 * s2) / n
            ssr = np.maximum(cyy - b1 * c1y - b2 * c2y, 0.0)
            enough = n >= int(0.8 * window)
            out[names[0]][:, cols] = np.where(enough, b1, np.nan)
            out[names[1]][:, cols] = np.where(enough & (cyy > 0), 1 - ssr / cyy, np.nan)
            out[names[2]][:, cols] = np.where(enough, np.sqrt(ssr / (n - 3) * 252), np.nan)
            # Residual return from t-window+1 to t-resid_skip, using the betas fitted at t.
            nn, ss1, ss2, ssy = (lagged(rsum(v, k)) for v in (one, x1, x2, y))
            out[names[3]][:, cols] = np.where(enough & (nn >= 0.8 * k), ssy - a * nn - b1 * ss1 - b2 * ss2, np.nan)
            ni = rsum(one, ind_window)
            out[names[4]][:, cols] = np.where(ni >= 0.8 * ind_window, rsum(x2, ind_window), np.nan)
            # Mean squared residual over the last m sessions, with a, b1, b2 fitted at t:
            # sum (y - a - b1 x1 - b2 x2)^2 expanded into raw rolling sums.
            for m in resid_sq_windows:
                nm = rsum(one, m)
                S = {key: rsum(v, m) for key, v in [("1", x1), ("2", x2), ("y", y), ("11", x1 * x1),
                                                     ("22", x2 * x2), ("12", x1 * x2), ("1y", x1 * y),
                                                     ("2y", x2 * y), ("yy", y * y)]}
                sse = (S["yy"] + nm * a * a + b1 * b1 * S["11"] + b2 * b2 * S["22"]
                       - 2 * a * S["y"] - 2 * b1 * S["1y"] - 2 * b2 * S["2y"]
                       + 2 * a * b1 * S["1"] + 2 * a * b2 * S["2"] + 2 * b1 * b2 * S["12"])
                out[sq_names[m]][:, cols] = np.where(enough & (nm >= 0.8 * m), np.maximum(sse, 0) / nm, np.nan)
            if resid_sq_windows:
                full = np.where(enough, ssr / n, np.nan)
                out[sq_names[window]][:, cols] = full
                short = out[sq_names[min(resid_sq_windows)]][:, cols]
                out[ratio_name][:, cols] = np.where(full > 0, short / full, np.nan)
    return out


def build_all(data):
    feats = price_features(data)
    feats.update(fundamental_features(data))
    feats.update(decomposition_features(data))
    feats.update(long_momentum(data))
    sq = decomposition_features(data, peers="sector", resid_sq_windows=(21, 63))
    feats.update({k: sq[k] for k in RESID_SQ_FEATURES})
    del sq
    target, label_end = labels(data)
    return feats, target, label_end
