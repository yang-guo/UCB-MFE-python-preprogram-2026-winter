"""Loads the database and builds two things the other scripts need:
a weekly table of features and 90 day outcomes, and a daily grid of closes.

From Lectures/modeling/course_data.py I call load_data, load_membership,
price_features, financial_events, join_financials, apply_membership and
make_labels without changes. The rest of this file is mine.

Look-ahead: a feature on signal date d only sees closes through d and filings
public by d. Columns that peek forward (entry_price, exit_price, label_end,
target) are outcomes. They are used for training labels and IC, never as an
input to a score.
"""

import re
import sys
import warnings
from pathlib import Path

import duckdb
import pandas as pd

HW_DIR = Path(__file__).parent.resolve()
REPO_ROOT = HW_DIR.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "Lectures" / "modeling"))
import course_data as cd  # noqa: E402

DB_FOLDER = REPO_ROOT / "Lectures" / "modeling" / "data"
BIG_DB = DB_FOLDER / "equities_top2000.duckdb"
SMALL_DB = DB_FOLDER / "equities_class.duckdb"
PARQUET_DIR = HW_DIR / "outputs"

SIZE_CUTOFF = 500          # only the 500 largest names each quarter
BAD_TICK_RATIO = 5.0       # a 5x move between two prints is a data error, not news
BOND_LIKE = re.compile(r"%|\bnotes?\b|debenture|warrant|cap secs|preferred", re.IGNORECASE)


def _query(db, sql, params=()):
    con = duckdb.connect(str(db), read_only=True)
    try:
        return con.execute(sql, list(params)).fetchdf()
    finally:
        con.close()


def trading_days(db, last_day):
    # the benchmark trades every session, so its dates double as the calendar
    got = _query(db, "SELECT date FROM benchmark WHERE date <= ? ORDER BY date", [last_day])
    return pd.DatetimeIndex(pd.to_datetime(got["date"])).astype("datetime64[ns]")


def index_levels(db, last_day):
    got = _query(db, "SELECT date, adj_close FROM benchmark WHERE date <= ? ORDER BY date", [last_day])
    levels = got["adj_close"]
    levels.index = pd.DatetimeIndex(pd.to_datetime(got["date"])).astype("datetime64[ns]")
    return levels


def week_end_sessions(calendar, from_day, also_include=()):
    """The final trading day of every Mon-Fri week, plus any extra days asked for."""
    days = calendar[calendar >= pd.Timestamp(from_day)]
    last_in_week = pd.Series(days, index=days).groupby(days.to_period("W-FRI")).max()
    wanted = set(last_in_week) | {pd.Timestamp(x) for x in also_include}
    return pd.DatetimeIndex(sorted(wanted & set(calendar)))


def names_to_load(db, cutoff=SIZE_CUTOFF):
    """Every symbol that made the size cutoff in some quarter. Notes, warrants
    and preferreds are thrown out by name and handed back so I can list them."""
    listing = _query(db, """SELECT symbol, name FROM universe
                            WHERE symbol IN (SELECT DISTINCT symbol FROM universe_membership
                                             WHERE universe_rank <= ?)
                            ORDER BY symbol""", [cutoff])
    looks_like_bond = listing["name"].fillna("").str.contains(BOND_LIKE)
    return listing.loc[~looks_like_bond, "symbol"].tolist(), listing[looks_like_bond]


def bad_tick_mask(prices, calendar, lookback=252):
    """True on day d if, within the last `lookback` sessions, some print was
    over 5x or under 1/5 of the print before it. Compares each print with the
    previous real print, so a jump across a trading gap still gets caught."""
    own = prices.pivot(index="date", columns="symbol", values="adj_close").sort_index()
    prev_print = own.ffill().shift(1)
    move = (own / prev_print).where(own.notna())
    bad = (move > BAD_TICK_RATIO) | (move < 1 / BAD_TICK_RATIO)
    return bad.astype(float).reindex(calendar).fillna(0.0).rolling(lookback, min_periods=1).max().astype(bool)


def assemble(db=BIG_DB, last_day="2026-09-30", from_day="2009-01-01", chunk_size=150,
             also_signal=("2016-03-31",), cutoff=SIZE_CUTOFF, chatty=True):
    """Returns (weekly table, daily close grid, benchmark levels, names skipped)."""
    calendar = trading_days(db, last_day)
    signal_days = week_end_sessions(calendar, from_day, also_signal)
    quarterly = cd.load_membership(db, end=last_day)
    to_load, skipped_names = names_to_load(db, cutoff)
    if db == SMALL_DB:
        cutoff = 2000   # the class database is too small for a top 500 cut

    chunks, close_blocks = [], []
    for offset in range(0, len(to_load), chunk_size):
        chunk_syms = to_load[offset: offset + chunk_size]
        # course_data warns about the small class file, nothing that matters here
        with warnings.catch_warnings(action="ignore"):
            px, funda, uni, _ = cd.load_data(db, end=last_day, symbols=chunk_syms)
        if len(px) == 0:
            continue
        feats = cd.price_features(px, calendar)
        closes = px.pivot(index="date", columns="symbol", values="adj_close").reindex(calendar)
        six_month = (closes / closes.shift(126) - 1).where(closes.notna())
        glitch = bad_tick_mask(px, calendar)

        snap = feats[feats.date.isin(signal_days)].copy()
        keys = pd.MultiIndex.from_arrays([snap.date, snap.symbol])
        snap["trend_6m"] = six_month.stack(future_stack=True).reindex(keys).to_numpy()
        snap["bad_tick"] = glitch.stack(future_stack=True).reindex(keys).fillna(False).to_numpy()

        snap = cd.join_financials(snap, cd.financial_events(funda), uni)
        snap = cd.make_labels(snap, px, calendar)
        chunks.append(snap)
        close_blocks.append(closes)
        if chatty:
            print(f"  {offset + len(chunk_syms):>5} of {len(to_load)} symbols done", flush=True)

    weekly = cd.apply_membership(pd.concat(chunks).reset_index(drop=True), quarterly, calendar)
    weekly["tradable"] = (weekly.eligible
                          & (weekly.universe_rank <= cutoff)
                          & ~weekly.bad_tick.astype(bool)
                          & weekly.momentum_252_21.notna())
    close_grid = pd.concat(close_blocks, axis=1).sort_index(axis=1)
    return weekly, close_grid, index_levels(db, last_day), skipped_names


def load_or_build(db=BIG_DB, rebuild=False, **opts):
    """First call builds and saves parquet files in outputs/ (git-ignored).
    Later calls just read them back."""
    stem = Path(db).stem
    paths = {part: PARQUET_DIR / f"{stem}_{part}.parquet" for part in ("panel", "prices", "benchmark")}
    if not rebuild and all(p.exists() for p in paths.values()):
        return (pd.read_parquet(paths["panel"]), pd.read_parquet(paths["prices"]),
                pd.read_parquet(paths["benchmark"])["adj_close"])
    PARQUET_DIR.mkdir(exist_ok=True)
    weekly, close_grid, levels, skipped_names = assemble(db, **opts)
    weekly.to_parquet(paths["panel"])
    close_grid.to_parquet(paths["prices"])
    levels.to_frame("adj_close").to_parquet(paths["benchmark"])
    skipped_names.to_csv(PARQUET_DIR / f"{stem}_dropped_non_common.csv", index=False)
    return weekly, close_grid, levels


if __name__ == "__main__":
    db = SMALL_DB if "--class" in sys.argv else BIG_DB
    weekly, close_grid, _ = load_or_build(db, rebuild="--refresh" in sys.argv,
                                          last_day="2015-12-31" if db == SMALL_DB else "2026-09-30")
    print("weekly table", weekly.shape, "| close grid", close_grid.shape,
          "| tradable rows", int(weekly.tradable.sum()))
    print("tradable stocks per signal date:")
    print(weekly[weekly.tradable].groupby("date").size().describe())
