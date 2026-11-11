from pathlib import Path
import sys
import time
import duckdb
import pandas as pd
import numpy as np

root = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
model_dir = root / "Lectures/modeling"
if str(model_dir) not in sys.path:
    sys.path.insert(0, str(model_dir))

from course_data import (load_data, load_membership, price_features, financial_events, join_financials, apply_membership, make_labels)

TECH_FEATURES = ['rsi_21', "dma_20", "dma_50", "dma_100"]
COURSE_FEATURES = ["momentum_252_21", "volatility_63"]
FUNDAMENTAL_FEATURES = ["roe", "debt_to_equity", "book_to_px", "earnings_yield", "fcf_yield", "ebitda_ev"]
RAW_FUNDAMENTALS = ["ttm_net_income", "ttm_free_cash_flow", "ttm_ebitda", "total_equity",
                    "total_debt", "cash_and_equivalents", "shares_outstanding"]
FEATURES = TECH_FEATURES + COURSE_FEATURES

TEST_DB = model_dir / "data/equities_class.duckdb"
FULL_DB = model_dir / "data/equities_top2000.duckdb"

def technical_features(px: pd.Series) -> pd.DataFrame:
    """px: Individual ticker's adj_close, reindexed to mkt calendar
    
    Returns df with same index and one column per feature
    """
    log_ret = np.log(px/px.shift(1))

    ### RSI 21
    avg_gain = log_ret.clip(lower=0).rolling(window=21, min_periods=21).sum()
    avg_loss = (-log_ret).clip(lower=0).rolling(window=21, min_periods=21).sum()
    rs = avg_gain / avg_loss
    rsi = 100-(100/(1+rs))

    ### 20/50/100 DMA distance
    def dma(n):
        return px / px.rolling(n, min_periods=n).mean()-1
    dma_20 = dma(20)
    dma_50 = dma(50)
    dma_100 = dma(100)

    out = pd.DataFrame(index=px.index)
    out["rsi_21"] = rsi
    out["dma_20"] = dma_20
    out["dma_50"] = dma_50
    out["dma_100"] = dma_100
    
    return out

def load_fundamentals(database, end, symbols):
    """Get fields needed to calc new ratios"""
    with duckdb.connect(str(database), read_only=True) as con:
        f = con.execute("""SELECT symbol, fiscal_date, period, filing_date, revenue, net_income, operating_income, total_assets, total_debt,
                        cash_and_equivalents, free_cash_flow, operating_cash_flow, ebitda, total_equity, shares_outstanding
                        FROM fundamentals
                        WHERE fiscal_date <= ? AND (filing_date <= ? OR filing_date IS NULL)
                            AND symbol IN (SELECT unnest(?))
                        ORDER BY symbol, fiscal_date""", [end, end, list(symbols)]).fetchdf()

    for c in ["fiscal_date", "filing_date"]:
        f[c] = pd.to_datetime(f[c]).astype("datetime64[ns]")
    return f

def fundamental_features(fundamentals: pd.DataFrame) -> pd.DataFrame:
    events = financial_events(fundamentals)
    q = (fundamentals.loc[fundamentals.period.eq("quarter")]
         .sort_values(["symbol", "fiscal_date", "filing_date"])
         .drop_duplicates(["symbol", "fiscal_date"], keep="last").copy())
    g = q.groupby("symbol")
    for col in ["net_income", "free_cash_flow", "ebitda"]:
        q["ttm_" + col] = g[col].transform(lambda s: s.rolling(4, min_periods=4).sum())

    span = (q.fiscal_date - g.fiscal_date.shift(3)).dt.days
    q.loc[~span.between(250, 300), ["ttm_net_income", "ttm_free_cash_flow", "ttm_ebitda"]]= np.nan

    return events.merge(q[["symbol", "fiscal_date"] + RAW_FUNDAMENTALS],
                        on=["symbol", "fiscal_date"], how="left", validate="many_to_one")


def add_fundamental_ratios(panel):
    stale = (panel.date - panel.fiscal_date).dt.days.gt(550)
    raw = panel[RAW_FUNDAMENTALS].mask(stale)
    mkt_cap = (panel.close * raw.shares_outstanding)
    equity = raw.total_equity.where(raw.total_equity > 0)
    ev = (mkt_cap + raw.total_debt.fillna(0) - raw.cash_and_equivalents.fillna(0)).where(lambda e: e > 0)

    ratios = pd.DataFrame({
        "roe": raw.ttm_net_income / equity,
        "debt_to_equity": raw.total_debt / equity,
        "book_to_px": raw.total_equity / mkt_cap,
        "earnings_yield": raw.ttm_net_income / mkt_cap,
        "fcf_yield": raw.ttm_free_cash_flow / mkt_cap,
        "ebitda_ev": raw.ttm_ebitda / ev,
    }, index=panel.index).replace([np.inf, -np.inf], np.nan)
    return panel.drop(columns=RAW_FUNDAMENTALS).join(ratios)

def month_end(calendar, start, end):
    """Last good day of each month, buy at next"""
    sessions = calendar[(calendar >= pd.Timestamp(start)) & (calendar <= pd.Timestamp(end))]
    return pd.DatetimeIndex(sessions.to_series().groupby(sessions.to_period("M")).max())

def load_calendar(db):
    """All good dates from benchmark"""
    with duckdb.connect(str(db), read_only=True) as con:
        dates = con.execute("SELECT date FROM benchmark ORDER BY date").fetchdf().date
    return pd.DatetimeIndex(pd.to_datetime(dates))

def get_symbols(db):
    with duckdb.connect(str(db), read_only=True) as con:
        return con.execute("SELECT symbol FROM universe ORDER BY symbol").fetchdf().symbol.tolist()

def add_technial(prices, calendar):
    """Run technical_features per stock on px"""
    frames = []
    for symbol, group in prices.groupby("symbol", sort=True):
        px = group.set_index("date").adj_close.reindex(calendar)
        tech = technical_features(px)
        tech["symbol"] = symbol
        tech.index.name = "date"
        frames.append(tech.loc[px.notna()].reset_index())
    return pd.concat(frames, ignore_index=True)

def build_input(database, end, start="2009-01-01", signal_dates=None, size=250):
    """For each batch: Load date -> lecture px features -> add technical indicators -> join -> add fundamental features -> add membership -> sample -> labels"""
    calendar = load_calendar(database)
    calendar = calendar[calendar <= pd.Timestamp(end)]

    if signal_dates is None:
        signal_dates = month_end(calendar, start, end)

    membership = load_membership(database, end=end)
    symbols = get_symbols(database)

    pieces, t0 = [], time.perf_counter()
    for i in range(0, len(symbols), size):
        batch = symbols[i:i + size]
        prices, _, universe, benchmark = load_data(database, end=end, symbols=batch)
        fundamentals = load_fundamentals(database, end, batch)

        if prices.empty:
            continue

        cal = pd.DatetimeIndex(benchmark.date)

        daily = price_features(prices, cal) # From lecture
        daily = daily.merge(add_technial(prices, cal), on=["date", "symbol"],
                            how="left", validate="one_to_one")
        daily = join_financials(daily, fundamental_features(fundamentals), universe)
        daily = apply_membership(daily, membership, cal)
        daily = add_fundamental_ratios(daily.loc[daily.date.isin(signal_dates)])
        daily = daily.loc[daily.date.isin(signal_dates)]

        pieces.append(make_labels(daily, prices, cal)) # From lecture

        print(f"  {min(i + size, len(symbols)):>5}/{len(symbols)} symbols "
                  f"({time.perf_counter() - t0:.0f}s)", flush=True)

    panel = pd.concat(pieces, ignore_index=True)
    return panel.sort_values(["date", "symbol"]).reset_index(drop=True)


