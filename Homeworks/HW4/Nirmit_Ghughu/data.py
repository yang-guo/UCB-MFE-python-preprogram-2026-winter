"""Load the course DuckDB file and build a weekly, look-ahead-safe research panel.

Design
------
* Prices are pivoted into wide ``date x symbol`` matrices on the supplied market
  calendar.  Missing quotes stay ``NaN`` (never forward-filled), so a 21-session
  momentum is ``NaN`` unless both endpoints were actually quoted.  This mirrors
  ``Lectures/modeling/course_data.price_features`` (adapted: vectorised across all
  stocks at once instead of a per-ticker loop, which matters for 3,849 tickers).
* Signals are made after the close of the **last market session of each week**.
  The trade (and the label's entry) is the **next session's close**.
* Quarterly membership is joined on the quarter of that next-session entry, as in
  ``course_data.apply_membership``.
* Financial statements become usable from ``filing_date + 1 calendar day``; old
  fiscal periods filed late never replace newer ones (adapted from
  ``course_data.financial_events`` / ``join_financials``).
* Labels: entry at next session close, exit at the first quoted close on/after
  entry + 90 calendar days (at most 5 sessions late), as in ``course_data.make_labels``.
  ``label_end`` is stored so that every fit can use only completed outcomes.

Nothing in this module removes a stock because of what happens to it later.
"""

from dataclasses import dataclass
from pathlib import Path
import warnings

import duckdb
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "Lectures" / "modeling" / "data"
FULL_DB = DATA_DIR / "equities_top2000.duckdb"
CLASS_DB = DATA_DIR / "equities_class.duckdb"

HOLDING_DAYS = 90
MAX_EXIT_DELAY = 5          # sessions a label/exit may wait for a missing quote

PRICE_FEATURES = [
    "mom_5", "mom_21", "mom_63", "mom_126", "mom_252_21",
    "vol_63", "max_ret_21", "drawdown_252", "beta_252", "log_dollar_vol_63",
]
FUNDAMENTAL_FEATURES = [
    "gross_profitability", "operating_roa", "net_margin", "debt_to_assets",
    "cash_to_assets", "fcf_to_assets", "accruals_to_assets", "asset_growth",
    "earnings_yield", "book_to_market", "filing_age_days",
]
SIZE_FEATURES = ["log_mcap"]
NUMERIC_FEATURES = PRICE_FEATURES + FUNDAMENTAL_FEATURES + SIZE_FEATURES


@dataclass
class MarketData:
    calendar: pd.DatetimeIndex      # every market session in the loaded window
    symbols: pd.Index               # all 3,849 union symbols (sorted)
    adj_close: pd.DataFrame         # date x symbol, NaN when not quoted
    close: pd.DataFrame             # raw (as-traded) close, float32
    dollar_volume: pd.DataFrame     # raw close * volume, float32
    benchmark: pd.Series            # ^SP500TR adjusted close
    sectors: pd.Series              # snapshot sector (audit caveat: undated)
    quarantine: dict                # counts of rows removed before use


def _ns(x):
    """DuckDB may return microsecond datetimes; pandas joins need one unit."""
    return pd.to_datetime(x).astype("datetime64[ns]")


def load_market(database=FULL_DB, start="2008-01-01", end="2015-12-31"):
    """Read prices into wide matrices. ``end`` hard-limits what the code can see."""
    database = Path(database)
    if not database.is_file():
        raise FileNotFoundError(f"Download {database.name} from bCourses into {database.parent}")
    with duckdb.connect(str(database), read_only=True) as con:
        cal = pd.DatetimeIndex(_ns(con.execute(
            "SELECT date FROM calendar WHERE date BETWEEN ? AND ? ORDER BY date", [start, end]
        ).fetchdf().date))
        bench = con.execute("SELECT date, adj_close FROM benchmark WHERE date BETWEEN ? AND ? ORDER BY date",
                            [start, end]).fetchdf()
        uni = con.execute("SELECT symbol, sector FROM universe ORDER BY symbol").fetchdf()
        # Audit what we quarantine, then never use it (see Lectures/modeling/README.md).
        bad_price = con.execute("""SELECT count(*) FROM prices WHERE date BETWEEN ? AND ?
            AND (adj_close IS NULL OR NOT isfinite(adj_close) OR adj_close <= 0)""", [start, end]).fetchone()[0]
        off_cal = con.execute("""SELECT count(*) FROM prices p WHERE date BETWEEN ? AND ?
            AND NOT EXISTS (SELECT 1 FROM calendar c WHERE c.date = p.date)""", [start, end]).fetchone()[0]
        # Let DuckDB emit integer row/column positions so pandas never holds 9M strings.
        raw = con.execute("""
            WITH s AS (SELECT symbol, row_number() OVER (ORDER BY symbol) - 1 AS sid FROM universe),
                 c AS (SELECT date, row_number() OVER (ORDER BY date) - 1 AS tid
                       FROM calendar WHERE date BETWEEN ? AND ?)
            SELECT c.tid, s.sid, p.adj_close, p.close, p.close * p.volume AS dv
            FROM prices p JOIN c USING (date) JOIN s USING (symbol)
            WHERE p.adj_close > 0 AND isfinite(p.adj_close)""", [start, end]).fetchnumpy()

    symbols = pd.Index(uni.symbol)
    shape = (len(cal), len(symbols))
    adj = np.full(shape, np.nan)
    close = np.full(shape, np.nan, dtype=np.float32)
    dv = np.full(shape, np.nan, dtype=np.float32)
    t, s = raw["tid"].astype(np.int64), raw["sid"].astype(np.int64)
    adj[t, s] = raw["adj_close"]
    close[t, s] = raw["close"]
    dv[t, s] = np.asarray(raw["dv"], dtype=np.float64)
    if bad_price or off_cal:
        warnings.warn(f"Quarantined {bad_price} nonpositive/invalid adjusted prices and "
                      f"{off_cal} off-calendar price rows", stacklevel=2)
    bench = bench.set_index(pd.DatetimeIndex(_ns(bench.date))).adj_close
    assert bench.index.equals(cal), "benchmark must define the market calendar"
    return MarketData(
        calendar=cal, symbols=symbols,
        adj_close=pd.DataFrame(adj, index=cal, columns=symbols),
        close=pd.DataFrame(close, index=cal, columns=symbols),
        dollar_volume=pd.DataFrame(dv, index=cal, columns=symbols),
        benchmark=bench,
        sectors=uni.set_index("symbol").sector.fillna("Unknown"),
        quarantine={"invalid_adj_close": int(bad_price), "off_calendar": int(off_cal)},
    )


def weekly_signal_dates(calendar, first="2010-01-01"):
    """Last market session of each calendar week, keeping only those with a next session."""
    cal = pd.Series(calendar, index=calendar)
    last = cal.groupby(calendar.to_period("W-SUN")).max()
    last = pd.DatetimeIndex(last.values)
    return last[(last >= pd.Timestamp(first)) & (last < calendar[-1])]


# --------------------------------------------------------------------------- price features
def _price_feature_matrices(m: MarketData):
    """Yield (name, wide DataFrame) one at a time to keep peak memory low."""
    px = m.adj_close
    ret = px.pct_change(fill_method=None)
    yield "mom_5", px.pct_change(5, fill_method=None)
    yield "mom_21", px.pct_change(21, fill_method=None)
    yield "mom_63", px.pct_change(63, fill_method=None)
    yield "mom_126", px.pct_change(126, fill_method=None)
    yield "mom_252_21", px.shift(21) / px.shift(252) - 1      # classic 12-1 momentum
    yield "vol_63", ret.rolling(63, min_periods=63).std() * np.sqrt(252)
    yield "max_ret_21", ret.rolling(21, min_periods=21).max()  # "lottery" effect
    yield "max_abs_ret_63", ret.abs().rolling(63, min_periods=1).max()  # data-error screen
    yield "drawdown_252", px / px.rolling(252, min_periods=252).max() - 1
    # Rolling beta to ^SP500TR, using only days where the stock was quoted.
    mkt = m.benchmark.pct_change(fill_method=None).to_numpy()[:, None]
    mask = ret.notna().to_numpy()
    x = ret.to_numpy()
    mk = np.where(mask, mkt, np.nan)
    roll = lambda a: pd.DataFrame(a).rolling(252, min_periods=126).sum().to_numpy()
    n = pd.DataFrame(mask.astype(float)).rolling(252, min_periods=126).sum().to_numpy()
    sx, sm = roll(x), roll(mk)
    cov = roll(x * mk) - sx * sm / n
    var = roll(mk * mk) - sm * sm / n
    with np.errstate(invalid="ignore", divide="ignore"):
        beta = np.where(var > 0, cov / var, np.nan)
    yield "beta_252", pd.DataFrame(beta, index=px.index, columns=px.columns)
    del x, mk, sx, sm, cov, var, beta
    yield "log_dollar_vol_63", np.log(m.dollar_volume.astype(float).rolling(63, min_periods=42).mean().clip(lower=1))
    yield "raw_close", m.close.astype(float)
    yield "quoted", px.notna().astype(float)


# --------------------------------------------------------------------------- labels
def _labels(px: pd.DataFrame, signal_pos: np.ndarray):
    """Entry next session, exit first quote >= entry+90d (<=5 sessions late)."""
    cal = px.index
    a = px.to_numpy()
    n_sig, n_sym = len(signal_pos), a.shape[1]
    entry_px = np.full((n_sig, n_sym), np.nan)
    exit_px = np.full((n_sig, n_sym), np.nan)
    exit_pos = np.full((n_sig, n_sym), -1, dtype=np.int64)
    entry_dates = cal[signal_pos + 1]
    for i, p in enumerate(signal_pos):
        entry_px[i] = a[p + 1]
        k = cal.searchsorted(entry_dates[i] + pd.Timedelta(days=HOLDING_DAYS))
        if k >= len(cal):
            continue  # outcome not finished inside the loaded window
        window = a[k:k + MAX_EXIT_DELAY + 1]
        has = ~np.isnan(window)
        first = np.where(has.any(axis=0), has.argmax(axis=0), -1)
        ok = first >= 0
        exit_pos[i, ok] = k + first[ok]
        exit_px[i, ok] = window[first[ok], np.flatnonzero(ok)]
    target = exit_px / entry_px - 1
    return entry_dates, target, exit_pos


# --------------------------------------------------------------------------- fundamentals
def load_financial_events(database=FULL_DB, end="2015-12-31"):
    """Availability-dated quarterly ratios (one row per symbol x available_date)."""
    with duckdb.connect(str(database), read_only=True) as con:
        f = con.execute("""SELECT symbol, fiscal_date, filing_date, revenue, gross_profit,
                operating_income, net_income, operating_cash_flow, free_cash_flow,
                total_assets, total_debt, total_equity, cash_and_equivalents
            FROM fundamentals
            WHERE period = 'quarter' AND filing_date > fiscal_date AND filing_date < ?""",
                        [end]).fetchdf()
    for c in ["fiscal_date", "filing_date"]:
        f[c] = _ns(f[c])
    f["available_date"] = f.filing_date + pd.Timedelta(days=1)
    f = f.loc[f.available_date <= pd.Timestamp(end)]
    assets = f.total_assets.where(f.total_assets > 0)
    revenue = f.revenue.where(f.revenue > 0)
    f["gross_profitability"] = f.gross_profit / assets
    f["operating_roa"] = f.operating_income / assets
    f["net_margin"] = f.net_income / revenue
    f["debt_to_assets"] = f.total_debt / assets
    f["cash_to_assets"] = f.cash_and_equivalents / assets
    f["fcf_to_assets"] = f.free_cash_flow / assets
    f["accruals_to_assets"] = (f.net_income - f.operating_cash_flow) / assets
    # Year-over-year asset growth: match the fiscal quarter ~one year earlier.
    f = f.sort_values(["fiscal_date"])
    prior = f[["symbol", "fiscal_date", "total_assets"]].rename(
        columns={"fiscal_date": "prior_fiscal", "total_assets": "prior_assets"})
    f["lookup"] = f.fiscal_date - pd.Timedelta(days=365)
    f = pd.merge_asof(f.sort_values("lookup"), prior.sort_values("prior_fiscal"), left_on="lookup",
                      right_on="prior_fiscal", by="symbol", direction="nearest",
                      tolerance=pd.Timedelta(days=35))
    # The prior filing must itself have been public; it was filed ~1y earlier, but check anyway.
    f["asset_growth"] = (f.total_assets / f.prior_assets.where(f.prior_assets > 0) - 1)
    ratio_cols = ["gross_profitability", "operating_roa", "net_margin", "debt_to_assets", "cash_to_assets",
                  "fcf_to_assets", "accruals_to_assets", "asset_growth"]
    f[ratio_cols] = f[ratio_cols].replace([np.inf, -np.inf], np.nan)
    f = f.sort_values(["symbol", "available_date", "fiscal_date"])
    latest = f.groupby("symbol").fiscal_date.cummax()            # a late, old period never wins
    f = f.loc[f.fiscal_date.eq(latest)].drop_duplicates(["symbol", "available_date"], keep="last")
    return f[["symbol", "fiscal_date", "available_date", "net_income", "total_equity"] + ratio_cols]


def load_membership(database=FULL_DB, end="2015-12-31"):
    with duckdb.connect(str(database), read_only=True) as con:
        mem = con.execute("""SELECT quarter_start, selection_date, symbol, universe_rank, market_cap_proxy
            FROM universe_membership WHERE quarter_start <= ?""", [end]).fetchdf()
    mem["quarter_start"] = _ns(mem.quarter_start)
    mem["selection_date"] = _ns(mem.selection_date)
    assert not mem.duplicated(["quarter_start", "symbol"]).any()
    return mem


# --------------------------------------------------------------------------- panel
def build_panel(database=FULL_DB, end="2015-12-31", start="2008-01-01", first_signal="2010-01-01",
                min_price=5.0, min_dollar_volume=5e6, max_abs_daily_move=0.75):
    """Return (panel, market). One panel row per weekly signal date x universe member.

    ``tradable`` marks rows passing the entry screens, all computed with data
    through the signal close:
      * quoted on the signal date and 63 complete trailing daily returns,
      * raw close >= $5 and 63-day average dollar volume >= $5M,
      * no |daily return| > 75% in the last 63 sessions (likely bad data/corporate action).
    """
    m = load_market(database, start=start, end=end)
    signals = weekly_signal_dates(m.calendar, first_signal)
    sig_pos = m.calendar.get_indexer(signals)
    entry_dates, target, exit_pos = _labels(m.adj_close, sig_pos)

    # Universe membership for the quarter of the next-session entry.
    mem = load_membership(database, end=end)
    entry_q = pd.DatetimeIndex(entry_dates).to_period("Q").start_time
    sig_q = pd.DataFrame({"quarter_start": entry_q, "sig": np.arange(len(signals))})
    mem_rows = mem.merge(sig_q, on="quarter_start")      # each member x each signal of its quarter
    mem_rows["sid"] = m.symbols.get_indexer(mem_rows.symbol)
    mem_rows = mem_rows.loc[mem_rows.sid >= 0]
    sel_ok = mem_rows.selection_date.to_numpy() <= signals[mem_rows.sig.to_numpy()]
    assert sel_ok.all(), "membership list must be known at the signal close"
    ri, ci = mem_rows.sig.to_numpy(), mem_rows.sid.to_numpy()

    panel = pd.DataFrame({
        "date": signals[ri], "entry_date": entry_dates[ri], "symbol": m.symbols[ci],
        "sector": m.sectors.reindex(m.symbols[ci]).to_numpy(),
        "universe_rank": mem_rows.universe_rank.to_numpy(),
        "log_mcap": np.log(mem_rows.market_cap_proxy.where(mem_rows.market_cap_proxy > 0)).to_numpy(),
        "market_cap_proxy": mem_rows.market_cap_proxy.to_numpy(),
        "target": target[ri, ci],
    })
    ep = exit_pos[ri, ci]
    panel["label_end"] = pd.DatetimeIndex(np.where(ep >= 0, m.calendar.values[np.maximum(ep, 0)],
                                                   np.datetime64("NaT")))
    for name, frame in _price_feature_matrices(m):
        panel[name] = frame.to_numpy()[sig_pos][ri, ci].astype(np.float32)
        del frame

    # Fundamentals: backward as-of join on availability, blank if fiscal period is stale.
    ev = load_financial_events(database, end=end)
    panel = panel.sort_values("date")
    panel = pd.merge_asof(panel, ev.sort_values("available_date"), left_on="date",
                          right_on="available_date", by="symbol", direction="backward")
    assert (panel.available_date.isna() | (panel.available_date <= panel.date)).all()
    stale = (panel.date - panel.fiscal_date).dt.days > 550
    fcols = [c for c in FUNDAMENTAL_FEATURES if c in panel.columns] + ["net_income", "total_equity"]
    panel.loc[stale, fcols] = np.nan
    panel["filing_age_days"] = (panel.date - panel.available_date).dt.days.where(~stale)
    # Valuation ratios use the capitalization proxy known at the quarter's selection date.
    cap = panel.market_cap_proxy.where(panel.market_cap_proxy > 0)
    panel["earnings_yield"] = 4 * panel.net_income / cap
    panel["book_to_market"] = panel.total_equity / cap

    panel["tradable"] = ((panel.quoted == 1) & panel.vol_63.notna() & (panel.raw_close >= min_price)
                         & (np.exp(panel.log_dollar_vol_63) >= min_dollar_volume)
                         & (panel.max_abs_ret_63 <= max_abs_daily_move))
    keep = ["date", "entry_date", "symbol", "sector", "universe_rank", "tradable", "target", "label_end"] + NUMERIC_FEATURES
    panel = panel[keep].sort_values(["date", "symbol"]).reset_index(drop=True)
    panel[NUMERIC_FEATURES] = panel[NUMERIC_FEATURES].replace([np.inf, -np.inf], np.nan)
    return panel, m
