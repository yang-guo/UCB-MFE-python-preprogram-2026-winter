"""Data and feature construction for HW4.

Adapted from the lecture helpers in Lectures/modeling/course_data.py (Lectures 7 and 8).
I reuse these lecture functions without change, and I cite them here:
  load_data, load_membership, price_features, financial_events, join_financials,
  apply_membership, make_labels, date_folds, make_pipeline, prediction_metrics.
What is new in this file
  * `build_panel` works in batches of stocks, so the 3,849-stock database fits in memory.
  * It keeps only every 5th market session as a signal date. The strategy enters every
    5 sessions, so these are the only dates where a score is used.
  * `add_relative_features` adds two sector-relative features (my own).
  * `prefix_check` tests that a feature never uses future prices.
Round 2 additions (see README, "Refinement round 2")
  * `SIZE_RISK_FEATURES`: market cap, price level and trading volume, all known at the signal date.
  * `risk_features` builds the volume and price-level features from raw closes and volumes.
  * `STUCK_COLUMNS` and the `stuck` flag mark a signal whose lot would have no exit quote.
Never use `is_active_snapshot` or `delisted_at_snapshot` from the `universe` table as features.
They are written at the END of the data (the snapshot) and so they leak the future.
"""

import sys
import warnings
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
sys.path.insert(0, str(ROOT / "Lectures" / "modeling"))
import course_data as cd  # noqa: E402  (lecture code, reused and cited)

DB_FULL = ROOT / "Lectures/modeling/data/equities_top2000.duckdb"
DB_CLASS = ROOT / "Lectures/modeling/data/equities_class.duckdb"

RELATIVE_FEATURES = ["mom63_vs_sector", "mom21_vs_sector"]
NUMERIC = cd.NUMERIC_FEATURES + RELATIVE_FEATURES
FEATURES = NUMERIC + ["sector"]
KEEP = ["date", "symbol", "sector", "universe_rank", "entry_date", "label_end", "target"] + cd.NUMERIC_FEATURES

# Round 2. Size, price level and liquidity. All of them are known at the signal close.
SIZE_RISK_FEATURES = ["log_mcap", "log_dvol_63", "log_price", "turnover_63"]
# Extra columns that the round 2 panel keeps. They describe the future outcome and are NOT features.
STUCK_COLUMNS = ["entry_price", "scheduled_exit", "resolved_date", "stuck"]
KEEP_V2 = KEEP + ["market_cap_proxy"] + SIZE_RISK_FEATURES + STUCK_COLUMNS


def all_symbols(database):
    with duckdb.connect(str(database), read_only=True) as con:
        return con.execute("SELECT symbol FROM universe ORDER BY symbol").fetchdf().symbol.tolist()


def signal_dates(calendar, first_test_day, step=5, start="2010-01-01"):
    """Every `step`-th session, anchored so that the first test entry has a signal date.

    The first test entry is the first session on or after first_test_day. Its signal is
    the session before. The strategy enters every `step` sessions from there, so every
    entry uses a signal date in this set.
    """
    calendar = pd.DatetimeIndex(calendar)
    anchor = calendar.searchsorted(pd.Timestamp(first_test_day)) - 1
    idx = np.arange(len(calendar))
    chosen = calendar[(idx - anchor) % step == 0]
    return chosen[chosen >= pd.Timestamp(start)]


def risk_features(prices, calendar):
    """Price level and trading volume, per stock and session. They use the past only.

    log_dvol_63 is the log of the average daily dollar volume over 63 sessions (raw close
    times volume). log_price is the log of the raw close (low-priced stocks are riskier).
    """
    calendar = pd.DatetimeIndex(calendar)
    close = prices.pivot(index="date", columns="symbol", values="close").reindex(calendar)
    volume = prices.pivot(index="date", columns="symbol", values="volume").reindex(calendar)
    dollars = (close * volume).where(close > 0)
    parts = {
        "log_dvol_63": np.log1p(dollars.rolling(63, min_periods=21).mean()),
        "log_price": np.log(close.where(close > 0)),
        "dvol_63": dollars.rolling(63, min_periods=21).mean(),
    }
    frames = []
    for name, wide in parts.items():
        frames.append(wide.rename_axis(columns="symbol").reset_index().melt(
            id_vars="date", value_name=name).set_index(["date", "symbol"])[name])
    return pd.concat(frames, axis=1).reset_index()


def load_size(database, end):
    """Market cap proxy (price times shares) known at each quarterly selection date."""
    with duckdb.connect(str(database), read_only=True) as con:
        size = con.execute("""SELECT symbol, quarter_start AS membership_quarter, market_cap_proxy
            FROM universe_membership WHERE quarter_start<=?""", [end]).fetchdf()
    size["membership_quarter"] = pd.to_datetime(size.membership_quarter).astype("datetime64[ns]")
    return size


def add_stuck_flag(panel, calendar, max_exit_delay=5, hold_days=90):
    """Mark signals that would be stuck, and say when that fact becomes known.

    stuck = True when the stock has a next-session quote, the 90 day exit date is inside
    the data, and no quote appears within `max_exit_delay` sessions after the due date.
    It is the case where the sale would be delayed. This is the FUTURE outcome, so it is
    used only as a training label, never as a feature.
    resolved_date is the first session when the outcome is known: the exit date when there
    is a target, and the end of the waiting window when the signal is stuck.
    """
    calendar = pd.DatetimeIndex(calendar)
    out = panel.copy()
    due_pos = calendar.searchsorted(out.scheduled_exit.to_numpy().astype("datetime64[ns]"))
    in_data = out.scheduled_exit.le(calendar[-1]) & out.entry_price.notna()
    window_end_pos = np.minimum(due_pos + max_exit_delay, len(calendar) - 1)
    window_end = pd.Series(calendar[window_end_pos], index=out.index)
    waited = in_data & window_end.le(calendar[-1]) & (due_pos + max_exit_delay <= len(calendar) - 1)
    out["stuck"] = waited & out.target.isna()
    out["resolved_date"] = out.label_end.where(out.target.notna(), window_end.where(out.stuck))
    return out


def build_panel(database, end, first_test_day, step=5, batch=120, start="2010-01-01", verbose=True, round2=False):
    """Return eligible signal-date rows with features and targets, plus a wide price table.

    round2=True also keeps market cap, price level, volume, and the stuck flag.
    """
    database = Path(database)
    with duckdb.connect(str(database), read_only=True) as con:
        bench = con.execute("SELECT date,adj_close FROM benchmark WHERE date<=? ORDER BY date", [end]).fetchdf()
    bench["date"] = pd.to_datetime(bench.date).astype("datetime64[ns]")
    calendar = pd.DatetimeIndex(bench.date)
    dates = signal_dates(calendar, first_test_day, step, start)
    membership = cd.load_membership(database, end=end)
    size = load_size(database, end) if round2 else None
    symbols = all_symbols(database)
    panels, wides = [], []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for b in range(0, len(symbols), batch):
            chunk = symbols[b:b + batch]
            prices, fundamentals, universe, _ = cd.load_data(database, end=end, symbols=chunk)
            wides.append(prices.pivot(index="date", columns="symbol", values="adj_close").reindex(calendar))
            feats = cd.price_features(prices, calendar)
            feats = feats[feats.date.isin(dates)]
            if feats.empty:
                continue
            panel = cd.join_financials(feats, cd.financial_events(fundamentals), universe)
            panel = cd.apply_membership(panel, membership, calendar)
            panel = cd.make_labels(panel, prices, calendar)
            if round2:
                risk = risk_features(prices, calendar)
                risk = risk[risk.date.isin(dates)]
                panel = panel.merge(risk, on=["date", "symbol"], how="left", validate="one_to_one")
                panel = panel.merge(size, on=["symbol", "membership_quarter"], how="left", validate="many_to_one")
                panel["log_mcap"] = np.log(panel.market_cap_proxy.where(panel.market_cap_proxy > 0))
                panel["turnover_63"] = panel.dvol_63 / panel.market_cap_proxy.where(panel.market_cap_proxy > 0)
                panel = add_stuck_flag(panel, calendar)
            panels.append(panel.loc[panel.eligible, KEEP_V2 if round2 else KEEP])
            if verbose:
                print(f"batch {b // batch + 1}/{-(-len(symbols) // batch)} done, rows so far {sum(map(len, panels)):,}", flush=True)
    panel = pd.concat(panels, ignore_index=True).sort_values(["date", "symbol"]).reset_index(drop=True)
    wide = pd.concat(wides, axis=1).sort_index(axis=1)
    return add_relative_features(panel), wide, bench.set_index("date").adj_close


def add_relative_features(panel):
    """Momentum minus the same-date median of the stock's sector. Peers are eligible stocks only."""
    out = panel.copy()
    for src, name in [("momentum_63", "mom63_vs_sector"), ("momentum_21", "mom21_vs_sector")]:
        out[name] = out[src] - out.groupby(["date", "sector"])[src].transform("median")
    return out


def prefix_check(database, symbols, cutoff, end):
    """Look-ahead test. Features from prices cut at `cutoff` must equal full-history features."""
    cutoff = pd.Timestamp(cutoff)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prices, _, _, bench = cd.load_data(database, end=end, symbols=symbols)
    cols = cd.PRICE_FEATURES
    full = cd.price_features(prices, bench.date)
    cut = cd.price_features(prices[prices.date <= cutoff], bench.date[bench.date <= cutoff])
    a = full[full.date <= cutoff].set_index(["symbol", "date"])[cols]
    b = cut.set_index(["symbol", "date"])[cols]
    a, b = a.align(b, join="inner")
    return float(np.nanmax(np.abs(a.to_numpy() - b.to_numpy()))), len(a)


def screen_mask(panel):
    """Skip stocks with extreme recent price behavior. It uses past information only.

    Such rows are often data errors (for example, a price that collapses by 99.99% and
    later 'returns' 100,000%). The rule is the same for training and for trading.
    """
    return (panel.volatility_63 <= 3) & panel.momentum_63.between(-0.8, 5) & (panel.drawdown_252 > -0.95)
