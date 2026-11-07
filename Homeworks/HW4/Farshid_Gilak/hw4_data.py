"""Data loading, causal features, and 90-day labels for HW4.

Adapted from Lectures/modeling/course_data.py: the invalid/off-calendar price
quarantine (load_data), calendar reindexing without price filling
(price_features), filing-date availability and the "late old fiscal period
never replaces a newer one" rule (financial_events), next-session membership
(apply_membership), and next-session-entry labels (make_labels). The
implementation here is vectorized on (date x symbol) matrices so the full
3,849-stock database fits in memory, and adds new features.

Only make_labels-style columns (entry_date, label_end, target) look into the
future. They are never model inputs.
"""

from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

HOLDING_DAYS = 90
MAX_EXIT_DELAY = 5
STALE_FINANCIAL_DAYS = 550

PRICE_FEATURES = ["mom_21", "mom_63", "mom_126", "mom_252_21", "vol_63",
                  "drawdown_252", "beta_252", "max_ret_21"]
FINANCIAL_FEATURES = ["gp_to_assets", "roa", "net_margin", "accruals", "debt_to_assets",
                      "cash_to_assets", "asset_growth", "sales_growth", "share_growth"]
VALUE_FEATURES = ["earnings_yield", "book_to_market", "sales_to_price", "fcf_yield", "log_mcap"]
NUMERIC_FEATURES = PRICE_FEATURES + FINANCIAL_FEATURES + VALUE_FEATURES + ["filing_age_days"]
FEATURES = NUMERIC_FEATURES + ["sector"]

DEFAULT_DATABASE = Path(__file__).resolve().parents[3] / "Lectures/modeling/data/equities_top2000.duckdb"


@dataclass
class MarketData:
    """Market-session prices as dense (date x symbol) matrices with NaN gaps."""

    calendar: pd.DatetimeIndex
    symbols: pd.Index
    adj_close: pd.DataFrame
    close: pd.DataFrame
    benchmark: pd.Series
    sectors: pd.Series
    membership: pd.DataFrame
    fundamentals: pd.DataFrame
    audit: dict


def load_market_data(database=DEFAULT_DATABASE, end="2026-09-30"):
    """Read prices, benchmark, membership, and fundamentals in read-only mode."""
    database = Path(database)
    if not database.is_file():
        raise FileNotFoundError(f"Download the database from bCourses to {database}")
    with duckdb.connect(str(database), read_only=True) as con:
        benchmark = con.execute("SELECT date, adj_close FROM benchmark WHERE date <= ? ORDER BY date",
                                [end]).fetchdf()
        universe = con.execute("SELECT symbol, sector FROM universe ORDER BY symbol").fetchdf()
        audit = {
            "price_rows": con.execute("SELECT count(*) FROM prices WHERE date <= ?", [end]).fetchone()[0],
            "invalid_adj_close_rows": con.execute(
                "SELECT count(*) FROM prices WHERE date <= ? AND NOT (adj_close > 0 AND isfinite(adj_close))",
                [end]).fetchone()[0],
            "off_calendar_rows": con.execute(
                "SELECT count(*) FROM prices p ANTI JOIN benchmark b USING (date) WHERE p.date <= ?",
                [end]).fetchone()[0],
        }
        # Integer positions keep the 15.8M-row price table out of pandas object columns.
        quotes = con.execute("""
            WITH cal AS (SELECT date, row_number() OVER (ORDER BY date) - 1 AS di
                         FROM benchmark WHERE date <= ?),
                 sym AS (SELECT symbol, row_number() OVER (ORDER BY symbol) - 1 AS si FROM universe)
            SELECT cal.di, sym.si, p.close, p.adj_close
            FROM prices p JOIN cal USING (date) JOIN sym USING (symbol)
            WHERE p.adj_close > 0 AND isfinite(p.adj_close)""", [end]).fetchnumpy()
        membership = con.execute("""SELECT quarter_start, symbol, selection_date, universe_rank
            FROM universe_membership WHERE quarter_start <= ? ORDER BY quarter_start, universe_rank""",
                                 [end]).fetchdf()
        fundamentals = con.execute("""SELECT symbol, fiscal_date, filing_date, revenue, gross_profit,
            net_income, operating_cash_flow, free_cash_flow, total_assets, total_debt, total_equity,
            cash_and_equivalents, shares_outstanding
            FROM fundamentals WHERE period = 'quarter' AND fiscal_date <= ? AND filing_date <= ?
            ORDER BY symbol, fiscal_date""", [end, end]).fetchdf()

    calendar = pd.DatetimeIndex(pd.to_datetime(benchmark.date), name="date")
    symbols = pd.Index(universe.symbol, name="symbol")
    adj = np.full((len(calendar), len(symbols)), np.nan)
    close = np.full_like(adj, np.nan)
    adj[quotes["di"], quotes["si"]] = quotes["adj_close"]
    close[quotes["di"], quotes["si"]] = quotes["close"]
    audit["loaded_price_rows"] = int(np.isfinite(adj).sum())

    for frame, columns in [(membership, ["quarter_start", "selection_date"]),
                           (fundamentals, ["fiscal_date", "filing_date"])]:
        for column in columns:
            frame[column] = pd.to_datetime(frame[column]).astype("datetime64[ns]")
    if membership.duplicated(["quarter_start", "symbol"]).any():
        raise ValueError("Duplicate quarterly membership key")
    return MarketData(
        calendar=calendar, symbols=symbols,
        adj_close=pd.DataFrame(adj, index=calendar, columns=symbols),
        close=pd.DataFrame(close, index=calendar, columns=symbols),
        benchmark=pd.Series(benchmark.adj_close.to_numpy(), index=calendar, name="^SP500TR"),
        sectors=universe.set_index("symbol").sector.fillna("Unknown"),
        membership=membership, fundamentals=fundamentals, audit=audit)


def weekly_signal_dates(calendar, start, end, extra=()):
    """First market session of every ISO week, plus any explicitly requested sessions."""
    sessions = calendar[(calendar >= pd.Timestamp(start)) & (calendar <= pd.Timestamp(end))]
    iso = sessions.isocalendar()
    firsts = pd.Series(sessions, index=sessions).groupby([iso.year.to_numpy(), iso.week.to_numpy()]).min()
    dates = pd.DatetimeIndex(firsts.to_numpy()).union(pd.DatetimeIndex(extra))
    if not dates.isin(calendar).all():
        raise ValueError("Signal dates must be market sessions")
    return dates


def price_feature_matrices(market):
    """Trailing price features on the market calendar; gaps stay missing."""
    px = market.adj_close
    ret = px / px.shift(1) - 1
    market_ret = market.benchmark.pct_change()
    features = {
        "mom_21": px / px.shift(21) - 1,
        "mom_63": px / px.shift(63) - 1,
        "mom_126": px / px.shift(126) - 1,
        "mom_252_21": px.shift(21) / px.shift(252) - 1,
        "vol_63": ret.rolling(63, min_periods=63).std() * np.sqrt(252),
        "drawdown_252": px / px.rolling(252, min_periods=200).max() - 1,
        "max_ret_21": ret.rolling(21, min_periods=15).max(),
    }
    # Beta uses the market return only on sessions where the stock also has a return.
    m = pd.DataFrame(np.where(ret.notna(), market_ret.to_numpy()[:, None], np.nan),
                     index=px.index, columns=px.columns)
    roll = lambda x: x.rolling(252, min_periods=200).mean()
    mean_r, mean_m = roll(ret), roll(m)
    variance = roll(m * m) - mean_m ** 2
    features["beta_252"] = (roll(ret * m) - mean_r * mean_m) / variance.where(variance > 0)
    # Screening input only (not a model feature): the largest one-session price move in a year.
    features["max_abs_logret_252"] = np.log(px / px.shift(1)).abs().rolling(252, min_periods=1).max()
    return features


def financial_events(fundamentals):
    """Availability-dated quarterly ratios (filing date + 1 calendar day)."""
    f = fundamentals.loc[fundamentals.filing_date.gt(fundamentals.fiscal_date)].copy()
    f = f.sort_values(["symbol", "fiscal_date", "filing_date"]).drop_duplicates(["symbol", "fiscal_date"], keep="last")
    f["available_date"] = f.filing_date + pd.Timedelta(days=1)
    g = f.groupby("symbol", sort=False)

    def lagged(column, quarters):
        return g[column].shift(quarters)

    # Trailing twelve months require four consecutive fiscal quarters (about 273 days apart end to end).
    span_3 = (f.fiscal_date - lagged("fiscal_date", 3)).dt.days
    consecutive = span_3.between(250, 300)
    for column in ["revenue", "gross_profit", "net_income", "operating_cash_flow", "free_cash_flow"]:
        ttm = g[column].rolling(4, min_periods=4).sum().reset_index(level=0, drop=True)
        f[f"ttm_{column}"] = ttm.where(consecutive)
    available_ns = f.available_date.astype("int64").groupby(f.symbol, sort=False)
    ttm_available = pd.to_datetime(available_ns.rolling(4, min_periods=4).max()
                                   .reset_index(level=0, drop=True)).astype("datetime64[ns]")
    f["available_date"] = ttm_available.where(consecutive, f.available_date)

    year_ago = (f.fiscal_date - lagged("fiscal_date", 4)).dt.days.between(340, 390)
    assets = f.total_assets.where(f.total_assets.gt(0))
    revenue_ttm = f.ttm_revenue.where(f.ttm_revenue.gt(0))
    f["gp_to_assets"] = f.ttm_gross_profit / assets
    f["roa"] = f.ttm_net_income / assets
    f["net_margin"] = f.ttm_net_income / revenue_ttm
    f["accruals"] = (f.ttm_net_income - f.ttm_operating_cash_flow) / assets
    f["debt_to_assets"] = f.total_debt / assets
    f["cash_to_assets"] = f.cash_and_equivalents / assets
    f["asset_growth"] = (assets / lagged("total_assets", 4).where(lambda x: x > 0) - 1).where(year_ago)
    f["sales_growth"] = (f.revenue / lagged("revenue", 4).where(lambda x: x > 0) - 1).where(year_ago)
    f["share_growth"] = (f.shares_outstanding / lagged("shares_outstanding", 4).where(lambda x: x > 0) - 1).where(year_ago)
    f["shares"] = f.shares_outstanding.where(f.shares_outstanding.gt(0))
    keep = ["symbol", "fiscal_date", "available_date", "shares", "total_equity", "ttm_net_income",
            "ttm_revenue", "ttm_free_cash_flow"] + FINANCIAL_FEATURES
    f = f[keep].replace([np.inf, -np.inf], np.nan)
    f = f.sort_values(["symbol", "available_date", "fiscal_date"])
    # An old fiscal period that becomes available late must not replace a newer known period.
    newest = f.groupby("symbol").fiscal_date.cummax()
    f = f.loc[f.fiscal_date.eq(newest)].drop_duplicates(["symbol", "available_date"], keep="last")
    return f.reset_index(drop=True)


def build_panel(market, signal_dates, holding_days=HOLDING_DAYS, max_exit_delay=MAX_EXIT_DELAY):
    """One row per (signal date, eligible member) with features and the 90-day label.

    Eligible: quote on the signal date, 63 complete trailing daily returns, and
    membership in the quarter of the next-session entry, whose selection date
    is on or before the signal close.
    """
    calendar = market.calendar
    signal_dates = pd.DatetimeIndex(signal_dates)
    sig_pos = calendar.get_indexer(signal_dates)
    if (sig_pos < 0).any() or (sig_pos + 1 >= len(calendar)).any():
        raise ValueError("Each signal needs a following market session")
    price_feats = price_feature_matrices(market)
    px = market.adj_close.to_numpy()
    eligible_price = np.isfinite(px[sig_pos]) & np.isfinite(price_feats["vol_63"].to_numpy()[sig_pos])
    rows, cols = np.nonzero(eligible_price)
    panel = pd.DataFrame({"date": signal_dates[rows], "symbol": market.symbols[cols],
                          "_t": sig_pos[rows], "_j": cols})
    for name, matrix in price_feats.items():
        panel[name] = matrix.to_numpy()[panel._t, panel._j]
    panel["close"] = market.close.to_numpy()[panel._t, panel._j]

    # Membership is joined by the entry quarter, as in course_data.apply_membership.
    panel["entry_date"] = calendar[panel._t + 1]
    panel["membership_quarter"] = panel.entry_date.dt.to_period("Q").dt.start_time
    members = market.membership.rename(columns={"quarter_start": "membership_quarter"})
    panel = panel.merge(members, on=["symbol", "membership_quarter"], how="inner", validate="many_to_one")
    if not panel.selection_date.le(panel.date).all():
        raise ValueError("Universe selection was unavailable at the signal close")

    events = financial_events(market.fundamentals)
    panel = pd.merge_asof(panel.sort_values("date"), events.sort_values("available_date"),
                          left_on="date", right_on="available_date", by="symbol", direction="backward")
    assert (panel.available_date.isna() | panel.available_date.le(panel.date)).all()
    stale = (panel.date - panel.fiscal_date).dt.days.gt(STALE_FINANCIAL_DAYS) | panel.fiscal_date.isna()
    financial_columns = FINANCIAL_FEATURES + ["shares", "total_equity", "ttm_net_income",
                                              "ttm_revenue", "ttm_free_cash_flow"]
    panel.loc[stale, financial_columns] = np.nan
    panel["filing_age_days"] = (panel.date - panel.available_date).dt.days.where(~stale)

    mcap = (panel.close * panel.shares).where(lambda x: x > 0)
    panel["earnings_yield"] = panel.ttm_net_income / mcap
    panel["book_to_market"] = panel.total_equity / mcap
    panel["sales_to_price"] = panel.ttm_revenue / mcap
    panel["fcf_yield"] = panel.ttm_free_cash_flow / mcap
    panel["log_mcap"] = np.log(mcap)
    panel["sector"] = panel.symbol.map(market.sectors).fillna("Unknown")

    # Label: entry at the next session's close, exit at the first quote on/after 90 calendar days.
    t, j = panel._t.to_numpy(), panel._j.to_numpy()
    entry_price = px[t + 1, j]
    due = panel.entry_date + pd.Timedelta(days=holding_days)
    due_pos = calendar.searchsorted(due)
    exit_pos = np.full(len(panel), -1)
    for delay in range(max_exit_delay, -1, -1):
        candidate = due_pos + delay
        inside = candidate < len(calendar)
        quoted = np.zeros(len(panel), dtype=bool)
        quoted[inside] = np.isfinite(px[candidate[inside], j[inside]])
        exit_pos = np.where(quoted, candidate, exit_pos)
    has_exit = (exit_pos >= 0) & np.isfinite(entry_price)
    exit_price = np.where(has_exit, px[np.clip(exit_pos, 0, None), j], np.nan)
    panel["label_end"] = pd.Series(calendar[np.clip(exit_pos, 0, None)], index=panel.index).where(has_exit)
    panel["target"] = exit_price / entry_price - 1

    numeric = NUMERIC_FEATURES + ["target"]
    panel[numeric] = panel[numeric].replace([np.inf, -np.inf], np.nan)
    keep = (["date", "symbol", "sector", "universe_rank", "entry_date", "label_end", "target"]
            + NUMERIC_FEATURES + ["max_abs_logret_252"])
    return panel[keep].sort_values(["date", "symbol"]).reset_index(drop=True)


def cross_sectional_ranks(panel, columns=NUMERIC_FEATURES):
    """Percentile rank of each feature among the same date's eligible members (NaN stays NaN)."""
    ranked = panel.copy()
    ranked[columns] = panel.groupby("date")[columns].rank(pct=True)
    return ranked
