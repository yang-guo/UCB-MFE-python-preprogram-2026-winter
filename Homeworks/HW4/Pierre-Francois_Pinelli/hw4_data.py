"""HW4 data: one row per stock and month-end, with 18 characteristics and the 90-day return.

Adapted from the course file Lectures/modeling/course_data.py (Lectures 7-8): same rules for
prices, filing dates, membership and 90-day labels. What I changed: we read the tickers in batches,
we only keep the month-end rows, and I added a few characteristics.
"""

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

DB = Path(__file__).resolve().parents[3] / "Lectures" / "modeling" / "data" / "equities_top2000.duckdb"
START = "2008-01-01"  # we need one year of history before the first signal (Dec 2009)
END = "2026-09-30"

# the 18 characteristics
PRICE_FEATURES = ["momentum_21", "momentum_63", "momentum_252_21", "volatility_63",
                  "drawdown_252", "beta_252", "log_dollar_volume_63"]
FINANCIAL_FEATURES = ["net_margin", "operating_roa", "debt_to_assets", "cash_to_assets",
                      "fcf_to_assets", "accruals_to_assets", "asset_growth", "gross_profitability"]
VALUE_FEATURES = ["book_to_market", "earnings_yield"]
FEATURES = PRICE_FEATURES + FINANCIAL_FEATURES + VALUE_FEATURES + ["log_size_rank"]


def sql(query, params=(), db=DB):
    """Run a query on the database (read only); the date columns come back as datetimes."""
    if not Path(db).is_file():
        raise FileNotFoundError(f"Download the database from bCourses to {db}")
    with duckdb.connect(str(db), read_only=True) as con:
        df = con.execute(query, list(params)).fetchdf()
    for col in df.columns:
        if "date" in col or col == "quarter_start":
            df[col] = pd.to_datetime(df[col]).astype("datetime64[ns]")
    return df


def load_benchmark(db=DB):
    """^SP500TR adjusted close. We use its dates as the market calendar."""
    b = sql("SELECT date, adj_close FROM benchmark WHERE date <= ? ORDER BY date", [END], db)
    return b.set_index("date").adj_close


def load_prices(symbols, calendar, db=DB, start=START):
    """Daily prices of some tickers. Bad rows are dropped, we never fix them (course rule)."""
    p = sql("""SELECT symbol, date, close, adj_close, volume FROM prices
               WHERE date >= ? AND date <= ? AND symbol IN (SELECT unnest(?))
               ORDER BY symbol, date""", [start, END, list(symbols)], db)
    bad = ~np.isfinite(p.adj_close) | (p.adj_close <= 0) | ~p.date.isin(calendar)
    return p[~bad], int(bad.sum())


def month_ends(calendar, start):
    """Last market day of each month = our signal dates."""
    days = calendar[calendar >= start].to_series()
    return pd.DatetimeIndex(days.groupby(days.index.to_period("M")).max())


def exit_of(entry, quotes, calendar):
    """When we sell a 90-day trade: first quote on or after entry + 90 days, max 5 sessions late."""
    if pd.isna(entry) or entry not in quotes.index:
        return pd.NaT, np.nan
    due = entry + pd.Timedelta(days=90)
    if due > calendar[-1]:
        return pd.NaT, np.nan
    later = quotes.loc[due:]
    if later.empty or calendar.get_loc(later.index[0]) - calendar.searchsorted(due) > 5:
        return pd.NaT, np.nan
    return later.index[0], later.iloc[0]


def stock_rows(prices, calendar, market_returns, signals):
    """For one stock: the characteristics we know at the signal close, and the 90-day return after."""
    p = prices.set_index("date").reindex(calendar)  # one row per market day, missing days stay empty
    px = p.adj_close
    r = px.pct_change(fill_method=None)
    f = pd.DataFrame({"adj_close": px, "close": p.close}, index=calendar)
    f["momentum_21"] = px.pct_change(21, fill_method=None)
    f["momentum_63"] = px.pct_change(63, fill_method=None)
    f["momentum_252_21"] = px.shift(21) / px.shift(252) - 1  # 12-month momentum, skipping the last month
    f["volatility_63"] = r.rolling(63, min_periods=63).std() * np.sqrt(252)
    f["drawdown_252"] = px / px.rolling(252, min_periods=252).max() - 1
    f["beta_252"] = (r.rolling(252, min_periods=200).cov(market_returns)
                     / market_returns.rolling(252, min_periods=200).var())
    dollar_volume = (p.close * p.volume).where((p.close > 0) & (p.volume > 0))
    f["log_dollar_volume_63"] = np.log(dollar_volume.rolling(63, min_periods=40).median())
    quotes = px.dropna()
    # data check: biggest one-day move over the last year (a x5 jump = broken price series)
    f["max_jump_252"] = np.log(quotes / quotes.shift(1)).abs().reindex(calendar).rolling(252, min_periods=1).max()

    rows = f.loc[signals].copy()
    rows["symbol"] = prices.symbol.iloc[0]
    rows["has_price"] = rows.adj_close.notna()
    # 90-day return: we buy at the next day's close and sell at the 90-day exit (course labels)
    rows["entry_date"] = calendar.to_series().shift(-1).loc[signals].to_numpy()
    rows["entry_price"] = quotes.reindex(rows.entry_date).to_numpy()
    exits = [exit_of(e, quotes, calendar) for e in rows.entry_date]
    rows["label_end"] = pd.to_datetime([e[0] for e in exits]).astype("datetime64[ns]")
    rows["exit_price"] = [e[1] for e in exits]
    rows["target"] = rows.exit_price / rows.entry_price - 1
    return rows.rename_axis("date").reset_index()


def load_financials(db=DB):
    """Quarterly filings and the ratios. We can only use a filing from filing date + 1 day (course rule)."""
    f = sql("""SELECT symbol, fiscal_date, filing_date, revenue, gross_profit, net_income,
                      operating_income, total_assets, total_debt, total_equity,
                      cash_and_equivalents, free_cash_flow, operating_cash_flow, shares_outstanding
               FROM fundamentals
               WHERE period = 'quarter' AND fiscal_date >= '2005-01-01' AND fiscal_date <= ?
                 AND (filing_date <= ? OR filing_date IS NULL)
               ORDER BY symbol, fiscal_date""", [END, END], db)
    # asset growth vs one year ago, and net income of the last 4 quarters (only if the dates make sense)
    by_stock = f.groupby("symbol")
    year_ago = by_stock[["fiscal_date", "total_assets"]].shift(4)
    is_year_ago = (f.fiscal_date - year_ago.fiscal_date).dt.days.between(340, 390)
    f["asset_growth"] = (f.total_assets / year_ago.total_assets.where(year_ago.total_assets > 0) - 1).where(is_year_ago)
    is_four_quarters = (f.fiscal_date - by_stock.fiscal_date.shift(3)).dt.days.between(250, 300)
    f["ttm_net_income"] = by_stock.net_income.rolling(4).sum().reset_index(level=0, drop=True).where(is_four_quarters)

    f = f[f.filing_date > f.fiscal_date].copy()  # a filing dated before the quarter end: we don't trust it
    f["available_date"] = f.filing_date + pd.Timedelta(days=1)
    assets = f.total_assets.where(f.total_assets > 0)
    f["net_margin"] = f.net_income / f.revenue.where(f.revenue > 0)
    f["operating_roa"] = f.operating_income / assets
    f["debt_to_assets"] = f.total_debt / assets
    f["cash_to_assets"] = f.cash_and_equivalents / assets
    f["fcf_to_assets"] = f.free_cash_flow / assets
    f["accruals_to_assets"] = (f.net_income - f.operating_cash_flow) / assets
    f["gross_profitability"] = f.gross_profit / assets
    f[FINANCIAL_FEATURES] = f[FINANCIAL_FEATURES].replace([np.inf, -np.inf], np.nan)

    f = f.sort_values(["symbol", "available_date", "fiscal_date"])
    f = f[f.fiscal_date == f.groupby("symbol").fiscal_date.cummax()]  # an old quarter filed late never replaces a newer one
    f = f.drop_duplicates(["symbol", "available_date"], keep="last")
    return f[["symbol", "fiscal_date", "available_date", "total_equity", "shares_outstanding", "ttm_net_income"]
             + FINANCIAL_FEATURES]


def add_financials(panel, financials):
    """Take the last filing published before each signal date, and compute the value ratios."""
    out = pd.merge_asof(panel.sort_values("date"), financials.sort_values("available_date"),
                        left_on="date", right_on="available_date", by="symbol", direction="backward")
    market_value = (out.close * out.shares_outstanding).where((out.close > 0) & (out.shares_outstanding > 0))
    out["book_to_market"] = out.total_equity / market_value
    out["earnings_yield"] = out.ttm_net_income / market_value
    too_old = (out.date - out.fiscal_date).dt.days > 550  # course rule: after 550 days the filing is too old
    out.loc[too_old, FINANCIAL_FEATURES + VALUE_FEATURES] = np.nan
    assert (out.available_date.isna() | (out.available_date <= out.date)).all()  # no filing from the future
    return out


def add_membership(panel, db=DB):
    """Top-2000 list of the quarter of the entry date (course rule)."""
    m = sql("""SELECT symbol, quarter_start, selection_date, universe_rank
               FROM universe_membership WHERE quarter_start <= ?""", [END], db)
    panel["quarter_start"] = panel.entry_date.dt.to_period("Q").dt.start_time
    out = panel.merge(m, on=["symbol", "quarter_start"], how="left", validate="many_to_one")
    out["is_member"] = out.universe_rank.notna()
    assert (out.selection_date[out.is_member] <= out.date[out.is_member]).all()  # the list is known at the signal
    out["log_size_rank"] = np.log(out.universe_rank)
    return out


def build_panel(db=DB):
    """One row per stock and month-end since Dec 2009: characteristics, membership and 90-day return."""
    bench = load_benchmark(db)
    calendar = bench.index[bench.index >= START]
    market_returns = bench[calendar].pct_change()
    signals = month_ends(calendar, "2009-12-01")
    symbols = sql("SELECT symbol FROM universe ORDER BY symbol", db=db).symbol.tolist()
    pieces, removed = [], 0
    for i in range(0, len(symbols), 300):  # 300 tickers at a time, otherwise too much memory
        prices, bad = load_prices(symbols[i:i + 300], calendar, db)
        removed += bad
        pieces += [stock_rows(g, calendar, market_returns, signals) for _, g in prices.groupby("symbol")]
    panel = pd.concat(pieces, ignore_index=True)
    panel = panel[panel.has_price | panel.target.notna()]
    panel = add_financials(panel, load_financials(db))
    panel = add_membership(panel, db)
    panel = panel.merge(sql("SELECT symbol, sector FROM universe", db=db), on="symbol", how="left")
    panel["sector"] = panel.sector.fillna("Unknown")
    # a stock can be scored if: in the list, has a price today, 63 days of returns, no broken price this year
    panel["clean_history"] = ~(panel.max_jump_252 > np.log(5))
    panel["eligible"] = panel.is_member & panel.has_price & panel.volatility_63.notna() & panel.clean_history
    print(f"panel: {len(panel):,} rows, {panel.symbol.nunique():,} stocks, {removed} bad price rows removed")
    return panel.sort_values(["date", "symbol"]).reset_index(drop=True)


def load_panel(db=DB, cache=Path(__file__).resolve().parent / "outputs" / "panel.pkl"):
    """We build the panel once (about 2 minutes) and save it in outputs/ (not on git)."""
    cache = Path(cache)
    if not cache.is_file():
        cache.parent.mkdir(exist_ok=True)
        build_panel(db).to_pickle(cache)
    return pd.read_pickle(cache)


def load_price_matrix(symbols, start, db=DB):
    """Daily adjusted closes for the backtester: one column per stock, empty when no quote."""
    calendar = load_benchmark(db).index
    prices, _ = load_prices(sorted(symbols), calendar, db, start=start)
    return prices.pivot(index="date", columns="symbol", values="adj_close").reindex(calendar[calendar >= start])
