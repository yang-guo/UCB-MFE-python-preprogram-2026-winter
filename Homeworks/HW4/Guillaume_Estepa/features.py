"""Build the (signal date, stock) panel: features, eligibility and labels.

Timing convention (same as the lectures):
* features and scores are computed after the close of signal date ``t``;
* the purchase happens at the close of the next session ``t+1``;
* the label is the adjusted-price return from that entry to the first quoted
  close on/after entry + 90 calendar days (at most 5 sessions late).

Every rolling window looks backwards only. Financial statements are used
from filing date + 1 calendar day. All features are finally converted to
same-date cross-sectional percentile ranks, which needs no fitted parameters.

``fundamental_events`` adapts ``financial_events`` / ``join_financials`` from
``Lectures/modeling/course_data.py`` (filing-date availability, "late old
period must not replace a newer one", 550-day staleness rule).
"""

import numpy as np
import pandas as pd

PRICE_FEATURES = [
    "mom_21", "mom_63", "mom_126", "mom_252_21",  # reversal / momentum
    "vol_63", "beta_252", "max_ret_21",  # risk and lottery-like behaviour
    "drawdown_252", "ma_gap_200",  # distance from 52-week high / trend
    "dollar_vol_63", "volume_trend",  # liquidity and attention
]
FUNDAMENTAL_FEATURES = [
    "net_margin", "operating_roa", "gross_profitability", "debt_to_assets",
    "cash_to_assets", "fcf_to_assets", "accruals_to_assets",
    "revenue_growth", "asset_growth",
    "earnings_yield", "fcf_yield", "book_to_market",
    "filing_age_days",
]
FEATURES = PRICE_FEATURES + FUNDAMENTAL_FEATURES

HOLDING_DAYS = 90
MAX_EXIT_DELAY = 5  # sessions


def weekly_grid(sessions, start="2010-01-01", step=5):
    """Every 5th session from ``start``: the dates used to train the model."""
    sessions = pd.DatetimeIndex(sessions)
    return sessions[sessions.searchsorted(pd.Timestamp(start))::step]


# --------------------------------------------------------------- price side
def price_feature_tables(md, rows):
    """Return {feature: array[len(rows), n_symbols]} using data up to each row."""
    px = md.adj_close
    ret = px.pct_change(fill_method=None)
    out = {}

    def keep(name, table):
        out[name] = table.to_numpy(dtype=np.float32)[rows]

    keep("mom_21", px / px.shift(21) - 1)
    keep("mom_63", px / px.shift(63) - 1)
    keep("mom_126", px / px.shift(126) - 1)
    keep("mom_252_21", px.shift(21) / px.shift(252) - 1)
    keep("vol_63", ret.rolling(63, min_periods=50).std())
    keep("max_ret_21", ret.rolling(21, min_periods=15).max())
    keep("drawdown_252", px / px.rolling(252, min_periods=200).max() - 1)
    keep("ma_gap_200", px / px.rolling(200, min_periods=150).mean() - 1)

    # Rolling one-year beta to the benchmark (only on days the stock traded).
    market = ret.notna().mul(md.benchmark.pct_change(), axis=0).where(ret.notna())
    mean = lambda table: table.rolling(252, min_periods=200).mean()
    covariance = mean(ret * market) - mean(ret) * mean(market)
    variance = mean(market * market) - mean(market) ** 2
    keep("beta_252", covariance / variance)
    del market, covariance, variance

    # Data-sanity screen inputs: largest one-day moves over the past year.
    keep("_max_ret_252", ret.rolling(252, min_periods=1).max())
    keep("_min_ret_252", ret.rolling(252, min_periods=1).min())
    del ret

    dollar_volume = md.dollar_volume.astype(np.float64)
    median_63 = dollar_volume.rolling(63, min_periods=40).median()
    keep("dollar_vol_63", median_63)
    keep("volume_trend", dollar_volume.rolling(21, min_periods=15).mean()
         / dollar_volume.rolling(252, min_periods=200).mean())
    out["_close"] = md.close.to_numpy()[rows]
    out["_has_price"] = px.notna().to_numpy()[rows]
    return out


# ---------------------------------------------------------- fundamental side
def fundamental_events(fundamentals):
    """Quarterly statements -> one row per (symbol, date the data became usable)."""
    f = fundamentals.loc[fundamentals.filing_date.gt(fundamentals.fiscal_date)].copy()
    f = (f.sort_values(["symbol", "fiscal_date"])
         .drop_duplicates(["symbol", "fiscal_date"], keep="last").reset_index(drop=True))
    by_symbol = f.groupby("symbol", sort=False)

    assets = f.total_assets.where(f.total_assets > 0)
    revenue = f.revenue.where(f.revenue > 0)
    f["net_margin"] = f.net_income / revenue
    f["operating_roa"] = f.operating_income / assets
    f["gross_profitability"] = f.gross_profit / assets
    f["debt_to_assets"] = f.total_debt / assets
    f["cash_to_assets"] = f.cash_and_equivalents / assets
    f["fcf_to_assets"] = f.free_cash_flow / assets
    f["accruals_to_assets"] = (f.net_income - f.operating_cash_flow) / assets

    # Year-over-year growth: the row four quarters back must be ~1 year older.
    gap_4q = (f.fiscal_date - by_symbol.fiscal_date.shift(4)).dt.days
    one_year = gap_4q.between(330, 400)
    f["revenue_growth"] = (revenue / by_symbol.revenue.shift(4).where(lambda s: s > 0) - 1).where(one_year)
    f["asset_growth"] = (assets / by_symbol.total_assets.shift(4).where(lambda s: s > 0) - 1).where(one_year)

    # Trailing-twelve-month sums need four consecutive quarters (~9 months apart).
    gap_3q = (f.fiscal_date - by_symbol.fiscal_date.shift(3)).dt.days
    full_year = gap_3q.between(240, 310)
    for column in ["net_income", "free_cash_flow"]:
        ttm = by_symbol[column].rolling(4, min_periods=4).sum().reset_index(level=0, drop=True)
        f[column + "_ttm"] = ttm.where(full_year)

    # Usable from the day after filing. Taking the running maximum also makes
    # sure the older quarters inside a TTM/growth number were already public,
    # and that a late filing of an old period never replaces a newer one.
    f["available_date"] = by_symbol.filing_date.cummax() + pd.Timedelta(days=1)
    f = f.drop_duplicates(["symbol", "available_date"], keep="last").replace([np.inf, -np.inf], np.nan)
    columns = ["symbol", "fiscal_date", "available_date", "net_margin", "operating_roa",
               "gross_profitability", "debt_to_assets", "cash_to_assets", "fcf_to_assets",
               "accruals_to_assets", "revenue_growth", "asset_growth", "net_income_ttm",
               "free_cash_flow_ttm", "total_equity", "shares_outstanding"]
    return f[columns].sort_values("available_date").reset_index(drop=True)


def join_fundamentals(panel, events):
    """Backward as-of join: the latest statement available on each signal date."""
    panel = panel.sort_values("date")
    joined = pd.merge_asof(panel, events, left_on="date", right_on="available_date",
                           by="symbol", direction="backward")
    assert (joined.available_date.isna() | (joined.available_date <= joined.date)).all()
    joined["filing_age_days"] = (joined.date - joined.available_date).dt.days.astype("float32")
    # Market value on the signal date. Vendor price and share bases can
    # disagree (see the data guide), so these ratios are only used as ranks.
    market_value = (joined.shares_outstanding * joined["_close"]).where(lambda s: s > 0)
    joined["earnings_yield"] = joined.net_income_ttm / market_value
    joined["fcf_yield"] = joined.free_cash_flow_ttm / market_value
    joined["book_to_market"] = joined.total_equity / market_value
    stale = (joined.date - joined.fiscal_date).dt.days > 550
    joined.loc[stale, FUNDAMENTAL_FEATURES] = np.nan
    return joined.drop(columns=["fiscal_date", "available_date", "net_income_ttm",
                                "free_cash_flow_ttm", "total_equity", "shares_outstanding"])


# -------------------------------------------------------------------- labels
def forward_returns(md, signal_rows, symbol_columns):
    """90-calendar-day return of a purchase made the session after each signal.

    Returns (forward_return, entry_date, label_end). ``label_end`` is the
    session whose close completes the outcome; training may only use rows
    whose ``label_end`` is before the fit date.
    """
    sessions = md.sessions
    prices = md.adj_close.to_numpy()
    n = len(sessions)
    entry_rows = signal_rows + 1
    has_entry = entry_rows < n
    safe_entry = np.where(has_entry, entry_rows, 0)
    entry_dates = sessions[safe_entry]
    due_rows = sessions.searchsorted(entry_dates + pd.Timedelta(days=HOLDING_DAYS))

    entry_price = np.where(has_entry, prices[safe_entry, symbol_columns], np.nan)
    exit_price = np.full(len(signal_rows), np.nan)
    exit_rows = np.full(len(signal_rows), -1)
    for delay in range(MAX_EXIT_DELAY + 1):
        rows = due_rows + delay
        candidate = (exit_rows < 0) & has_entry & (rows < n)
        quote = np.full(len(signal_rows), np.nan)
        quote[candidate] = prices[rows[candidate], symbol_columns[candidate]]
        found = candidate & np.isfinite(quote)
        exit_price[found], exit_rows[found] = quote[found], rows[found]

    forward = exit_price / entry_price - 1
    done = np.isfinite(forward)
    label_end = pd.Series(pd.NaT, index=range(len(signal_rows)), dtype="datetime64[ns]")
    label_end[done] = sessions[exit_rows[done]]
    entry = pd.Series(entry_dates).where(has_entry)
    return forward, entry.to_numpy(), label_end.to_numpy()


# --------------------------------------------------------------------- panel
def build_panel(md, signal_dates):
    """One row per eligible (signal date, stock) with ranked features and label.

    Eligible = in the quarterly universe of the *entry* session, quoted on the
    signal date, at least one year of price history, a positive trailing
    dollar volume, and no one-day move above +100% / below -50% in the past
    year (penny-stock and data-error guard). All of these are known at ``t``.
    """
    sessions = md.sessions
    signal_dates = pd.DatetimeIndex(sorted(set(signal_dates)))
    rows = sessions.get_indexer(signal_dates)
    assert (rows >= 0).all(), "signal dates must be market sessions"
    usable = rows + 1 < len(sessions)  # an entry session must exist
    signal_dates, rows = signal_dates[usable], rows[usable]
    tables = price_feature_tables(md, rows)

    # Membership of the quarter in which the purchase would happen.
    column_of = {symbol: j for j, symbol in enumerate(md.adj_close.columns)}
    members = {quarter: (group.symbol.map(column_of).to_numpy(), group.universe_rank.to_numpy())
               for quarter, group in md.membership.groupby("quarter_start")}
    entry_quarters = sessions[rows + 1].to_period("Q").start_time
    date_index, columns, ranks = [], [], []
    for k, quarter in enumerate(entry_quarters):
        if quarter in members:
            cols, rank = members[quarter]
            date_index.append(np.full(len(cols), k))
            columns.append(cols)
            ranks.append(rank)
    k, j = np.concatenate(date_index), np.concatenate(columns)

    panel = pd.DataFrame({"date": signal_dates[k], "symbol": md.adj_close.columns[j],
                          "universe_rank": np.concatenate(ranks)})
    for name, table in tables.items():
        panel[name] = table[k, j]
    panel["_row"], panel["_column"] = rows[k], j

    eligible = (panel["_has_price"] & panel.vol_63.notna() & panel.mom_252_21.notna()
                & panel.dollar_vol_63.gt(0)
                & panel["_max_ret_252"].le(1.0) & panel["_min_ret_252"].ge(-0.5))
    panel = panel.loc[eligible].reset_index(drop=True)

    forward, entry_date, label_end = forward_returns(md, panel["_row"].to_numpy(), panel["_column"].to_numpy())
    panel["entry_date"], panel["label_end"], panel["forward_return"] = entry_date, label_end, forward

    panel = join_fundamentals(panel, fundamental_events(md.fundamentals))
    panel["sector"] = panel.symbol.map(md.sectors).fillna("Unknown")

    by_date = panel.groupby("date")
    panel["liquidity_rank"] = by_date.dollar_vol_63.rank(ascending=False, method="first")
    # Same-date percentile ranks in [-0.5, 0.5]; missing values stay missing.
    panel[FEATURES] = (by_date[FEATURES].rank(pct=True) - 0.5).astype("float32")
    # Target: return relative to the same-date median stock (what ranking needs).
    panel["target"] = panel.forward_return - by_date.forward_return.transform("median")
    # Robust alternative: the same outcome expressed as a same-date percentile.
    panel["target_rank"] = by_date.forward_return.rank(pct=True) - 0.5
    panel = panel.drop(columns=[c for c in panel.columns if c.startswith("_")])
    return panel.sort_values(["date", "symbol"]).reset_index(drop=True)
