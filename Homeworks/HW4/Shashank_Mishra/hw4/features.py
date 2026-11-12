"""Candidate features, labels, and the low-correlation feature screen.

Timing convention (same as the lectures): every feature uses data through the
close of signal date ``t``. The trade enters at the close of the *next*
market session ``t+1`` and exits at the first quoted close on/after
``t+1 + 90 calendar days`` (at most 5 sessions late, otherwise the label is
missing/censored). Future prices appear only in :func:`make_labels`.
"""

import numpy as np
import pandas as pd

# --- Candidate features ---------------------------------------------------------------
# Each one is a classic, economically motivated equity signal. The screen in
# select_low_correlation_features keeps a subset whose cross-sectional rank
# correlations are low, so each surviving feature carries different information.
PRICE_CANDIDATES = {
    "mom_12_1": "12-month return skipping the last month (momentum)",
    "rev_1m": "last 21-session return (short-term reversal)",
    "vol_63": "63-session daily-return volatility",
    "beta_252": "252-session beta to the S&P 500 TR",
    "idio_vol_252": "252-session residual volatility after beta",
    "dist_52w_high": "price / 252-session high - 1",
    "max_ret_21": "largest daily return in 21 sessions (lottery demand)",
    "volume_trend": "log(21-session avg volume / 252-session avg volume)",
}
FUND_CANDIDATES = {
    "log_mcap": "log(close x latest known shares)",
    "earnings_yield": "TTM net income / market cap",
    "book_to_market": "book equity / market cap",
    "sales_to_price": "TTM revenue / market cap",
    "gross_profitability": "TTM gross profit / total assets",
    "asset_growth": "total assets / total assets 4 quarters earlier - 1",
    "accruals": "(TTM net income - TTM operating cash flow) / total assets",
    "debt_to_assets": "total debt / total assets",
    "earnings_change": "(NI_q - NI_q-4) / total assets (earnings-surprise proxy)",
}
CANDIDATES = list(PRICE_CANDIDATES) + list(FUND_CANDIDATES)
STALE_FISCAL_DAYS = 550


def feature_dates(calendar, start, end, anchor="2016-03-31", step=5):
    """Every ``step``-th session, anchored so that ``anchor`` is a feature date.

    Anchoring on 2016-03-31 makes the first test entry the April 1, 2016 close.
    The last session is excluded because it has no next-session entry.
    """
    calendar = pd.DatetimeIndex(calendar)
    a = calendar.get_loc(pd.Timestamp(anchor))
    idx = np.arange(a % step, len(calendar) - 1, step)
    dates = calendar[idx]
    keep = (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))
    return dates[keep]


def _rolling_mean(a, window, min_periods):
    return pd.DataFrame(a).rolling(window, min_periods=min_periods).mean().to_numpy()


def price_feature_matrices(md):
    """Full-calendar price features (rows = sessions). No price filling."""
    P = md.adj_close
    with np.errstate(invalid="ignore", divide="ignore"):
        R = P[1:] / P[:-1] - 1
    R = np.vstack([np.full((1, P.shape[1]), np.nan), R])      # NaN if either day is missing
    m = md.benchmark.pct_change().to_numpy()[:, None]

    def shift(a, k):
        out = np.full_like(a, np.nan)
        out[k:] = a[:-k]
        return out

    out = {}
    with np.errstate(invalid="ignore", divide="ignore"):
        out["mom_12_1"] = shift(P, 21) / shift(P, 252) - 1
        out["rev_1m"] = P / shift(P, 21) - 1
        out["vol_63"] = pd.DataFrame(R).rolling(63, min_periods=63).std().to_numpy() * np.sqrt(252)

        # Beta / residual vol from rolling moments, using only days the stock traded.
        mask = np.isfinite(R)
        Mm = np.where(mask, m, np.nan)
        w, mp = 252, 200
        er, em = _rolling_mean(R, w, mp), _rolling_mean(Mm, w, mp)
        erm, emm, err = _rolling_mean(R * Mm, w, mp), _rolling_mean(Mm * Mm, w, mp), _rolling_mean(R * R, w, mp)
        var_m, var_r, cov = emm - em ** 2, err - er ** 2, erm - er * em
        beta = cov / var_m
        out["beta_252"] = beta
        out["idio_vol_252"] = np.sqrt(np.clip(var_r - beta * cov, 0, None)) * np.sqrt(252)

        hi = pd.DataFrame(P).rolling(252, min_periods=200).max().to_numpy()
        out["dist_52w_high"] = P / hi - 1
        out["max_ret_21"] = pd.DataFrame(R).rolling(21, min_periods=15).max().to_numpy()
        V = np.where(md.volume > 0, md.volume, np.nan)
        out["volume_trend"] = np.log(_rolling_mean(V, 21, 15) / _rolling_mean(V, 252, 200))
    # Data-quality screen (not a model input): a >=100% one-day move in the past
    # year is far more often a vendor price error than an investable signal.
    out["max_abs_ret_252"] = pd.DataFrame(np.abs(R)).rolling(252, min_periods=1).max().to_numpy()
    out["n_returns_63"] = pd.DataFrame(np.isfinite(R).astype(float)).rolling(63, min_periods=1).sum().to_numpy()
    return out


def build_panel(md, events, membership, dates):
    """Long (date, symbol) panel for the requested signal dates."""
    pf = price_feature_matrices(md)
    rows = md.calendar.get_indexer(dates)
    assert (rows >= 0).all()
    n_d, n_s = len(rows), len(md.symbols)
    panel = pd.DataFrame({
        "date": np.repeat(dates.to_numpy(), n_s),
        "symbol": np.tile(md.symbols.to_numpy(), n_d),
        "adj_close": md.adj_close[rows].ravel(),
        "close": md.close[rows].ravel(),
    })
    for k in list(PRICE_CANDIDATES) + ["n_returns_63", "max_abs_ret_252"]:
        panel[k] = pf[k][rows].ravel()
    panel = panel.loc[panel.adj_close.notna()].reset_index(drop=True)   # must quote at the signal close

    # Membership is decided by the quarter of the next-session entry (lecture convention).
    nxt = md.calendar[rows + 1]
    entry_q = pd.Series(pd.PeriodIndex(nxt, freq="Q").start_time.astype("datetime64[ns]"), index=dates)
    panel["entry_date"] = panel.date.map(pd.Series(nxt, index=dates))
    panel["membership_quarter"] = panel.date.map(entry_q)
    mem = membership.rename(columns={"quarter_start": "membership_quarter"})
    panel = panel.merge(mem[["symbol", "membership_quarter", "selection_date", "universe_rank"]],
                        on=["symbol", "membership_quarter"], how="left", validate="many_to_one")
    member = panel.universe_rank.notna()
    assert panel.loc[member, "selection_date"].le(panel.loc[member, "date"]).all(), "membership not yet known"
    # Same eligibility rule as class: dated membership + 63 complete trailing returns.
    panel["eligible"] = member & panel.n_returns_63.ge(63) & panel.vol_63.notna()

    # Backward as-of join of availability-dated financial events.
    ev = events.sort_values("available_date")
    panel = pd.merge_asof(panel.sort_values("date"), ev, left_on="date", right_on="available_date",
                          by="symbol", direction="backward")
    assert (panel.available_date.isna() | panel.available_date.le(panel.date)).all()
    stale = (panel.date - panel.fiscal_date).dt.days.gt(STALE_FISCAL_DAYS) | panel.fiscal_date.isna()
    with np.errstate(invalid="ignore", divide="ignore"):
        assets = panel.total_assets.where(panel.total_assets > 0)
        mcap = (panel.close * panel.shares_outstanding).where(lambda x: x > 0)
        panel["log_mcap"] = np.log(mcap)
        panel["earnings_yield"] = panel.net_income_ttm / mcap
        panel["book_to_market"] = panel.total_equity / mcap
        panel["sales_to_price"] = panel.revenue_ttm / mcap
        panel["gross_profitability"] = panel.gross_profit_ttm / assets
        panel["asset_growth"] = panel.total_assets / panel.assets_lag4.where(panel.assets_lag4 > 0) - 1
        panel["accruals"] = (panel.net_income_ttm - panel.operating_cash_flow_ttm) / assets
        panel["debt_to_assets"] = panel.total_debt / assets
        panel["earnings_change"] = (panel.net_income - panel.net_income_lag4) / assets
    panel[list(FUND_CANDIDATES)] = panel[list(FUND_CANDIDATES)].replace([np.inf, -np.inf], np.nan)
    panel.loc[stale, list(FUND_CANDIDATES)] = np.nan
    panel["filing_age_days"] = (panel.date - panel.available_date).dt.days
    panel["sector"] = panel.symbol.map(md.sector).fillna("Unknown")
    keep = ["date", "symbol", "entry_date", "adj_close", "close", "universe_rank", "eligible", "sector",
            "fiscal_date", "available_date", "filing_age_days", "max_abs_ret_252"] + CANDIDATES
    # Only eligible rows are ever trained on or traded; held positions are
    # marked from the price matrix, not from this panel.
    panel = panel.loc[panel.eligible, keep]
    return panel.sort_values(["date", "symbol"]).reset_index(drop=True)


def make_labels(panel, md, holding_days=90, max_exit_delay=5):
    """Attach entry/exit dates, prices, and the 90-calendar-day gross return."""
    cal = md.calendar
    sym_ix = md.symbols.get_indexer(panel.symbol)
    entry_ix = cal.get_indexer(panel.entry_date)
    entry_px = md.adj_close[entry_ix, sym_ix]
    due = panel.entry_date + pd.Timedelta(days=holding_days)
    due_ix = cal.searchsorted(due)                         # first session on/after maturity
    exit_ix = np.full(len(panel), -1)
    for k in range(max_exit_delay + 1):                    # first quote within 5 sessions of maturity
        cand = due_ix + k
        ok = (exit_ix < 0) & (cand < len(cal))
        px = np.full(len(panel), np.nan)
        px[ok] = md.adj_close[cand[ok], sym_ix[ok]]
        exit_ix = np.where(ok & np.isfinite(px), cand, exit_ix)
    has_exit = (exit_ix >= 0) & np.isfinite(entry_px)
    out = panel.copy()
    out["entry_price"] = entry_px
    out["label_end"] = pd.NaT
    out.loc[has_exit, "label_end"] = cal[exit_ix[has_exit]]
    out["exit_price"] = np.where(has_exit, md.adj_close[np.clip(exit_ix, 0, None), sym_ix], np.nan)
    out["target"] = out.exit_price / out.entry_price - 1
    return out


def add_cross_sectional_ranks(panel, columns):
    """Percentile-rank each feature within its date among *eligible* stocks.

    This is not learned from training data (it uses only same-date peers), so
    it cannot leak across time. It removes level drift (e.g. vol in 2008 vs
    2017) and caps the influence of the bad vendor values the data guide warns
    about. Missing values stay missing; LightGBM routes them natively.
    """
    out = panel.copy()
    elig = out.eligible
    ranks = out.loc[elig].groupby("date")[columns].rank(pct=True)
    for c in columns:
        out[f"r_{c}"] = np.nan
        out.loc[elig, f"r_{c}"] = ranks[c]
    out["target_rank"] = np.nan
    lab = elig & out.target.notna()
    out.loc[lab, "target_rank"] = out.loc[lab].groupby("date").target.rank(pct=True)
    # Alternative target keeping magnitude: same-date winsorised (1%/99%) return
    # minus the same-date mean. Clipping per date stops bad vendor prices
    # (e.g. 100,000x "returns") from dominating squared-error fits.
    g = out.loc[lab].groupby("date").target
    clipped = out.loc[lab, "target"].clip(g.transform(lambda s: s.quantile(0.01)), g.transform(lambda s: s.quantile(0.99)))
    out["target_excess"] = np.nan
    out.loc[lab, "target_excess"] = clipped - clipped.groupby(out.loc[lab, "date"]).transform("mean")
    return out


def mean_rank_correlations(frame, columns):
    """Average same-date Spearman correlation matrix (ranked inputs => Pearson)."""
    mats = []
    for _, g in frame.groupby("date"):
        c = g[columns].corr(min_periods=100)
        mats.append(c.to_numpy())
    return pd.DataFrame(np.nanmean(np.stack(mats), axis=0), index=columns, columns=columns)


def rank_ic(frame, feature, target="target_rank"):
    ic = frame.groupby("date").apply(lambda g: g[feature].corr(g[target]) if g[feature].notna().sum() > 100 else np.nan,
                                     include_groups=False)
    return ic


def select_low_correlation_features(frame, columns, max_abs_corr=0.5):
    """Greedy screen: strongest |IC| first, keep a feature only if its average
    cross-sectional |correlation| with every kept feature is <= max_abs_corr."""
    corr = mean_rank_correlations(frame, columns)
    ic = pd.DataFrame({c: rank_ic(frame, c) for c in columns})
    stats = pd.DataFrame({"mean_ic": ic.mean(), "ic_tstat": ic.mean() / ic.std() * np.sqrt(ic.notna().sum()),
                          "coverage": frame[columns].notna().mean()})
    order = stats.mean_ic.abs().sort_values(ascending=False).index
    kept, reasons = [], {}
    for c in order:
        clash = [k for k in kept if abs(corr.loc[c, k]) > max_abs_corr]
        if clash:
            reasons[c] = f"dropped: |corr| {abs(corr.loc[c, clash[0]]):.2f} with {clash[0]}"
        else:
            kept.append(c)
            reasons[c] = "kept"
    stats["decision"] = pd.Series(reasons)
    return kept, corr, stats.loc[order]
