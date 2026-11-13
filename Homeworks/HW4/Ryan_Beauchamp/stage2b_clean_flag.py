"""HW4 stage 2b: causal data-quality measures for a `clean` flag.

Why: a few price series in the database are broken (AAM mixes two securities,
VBIV has an unadjusted 800x jump, LCI's adjusted price jumps 17x while its raw
close jumps 4x). Their 90-day targets reach +14,000,000% and swamp any
squared-error metric. We screen them out using only information available at
each signal close; no target or future price is used to define the flag.

Measures (all known at the close of the signal date t):
  * max_abs_log_move_252: largest |ln(P_t / P_prev)| between consecutive observed
    adjusted closes over the trailing 252 observed sessions, including today.
    A log move is symmetric: a crash from $25,000 to $0.40 counts as large,
    whereas a plain percentage return cannot fall below -100%.
  * close: the supplied close on t (price floor).
  * cap_proxy: the capitalization proxy from the quarterly selection used by the
    row (known at that quarter's selection date, which is <= t).

This script saves the measures (not a chosen flag) to outputs/clean_measures/,
prints exclusion counts per year for candidate thresholds, and runs a prefix
test. The stage 1 panel is read only, never modified.

Run from the repository root:
    development (default): .venv/Scripts/python -B Homeworks/HW4/Ryan_Beauchamp/stage2b_clean_flag.py
    full period:           .venv/Scripts/python -B Homeworks/HW4/Ryan_Beauchamp/stage2b_clean_flag.py --full

Thresholds are fixed (chosen on 2010-2015 counts); the full run only applies the
same measures to the longer panel and repeats the prefix tests.

Credits: no lecture code reused in this file (it reads the stage 1 panel, which
uses Lectures/modeling/course_data.py). Written with AI assistance (Claude Code)
and reviewed by the author.
"""

import argparse
import shutil
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
DATABASE = REPO / "Lectures" / "modeling" / "data" / "equities_top2000.duckdb"
PANEL_DIR = HERE / "outputs" / "panel_stage1"
MEASURES_DIR = HERE / "outputs" / "clean_measures"
DATA_END = pd.Timestamp("2016-03-31")
WINDOW = 252

# Candidate thresholds; each criterion flags a row when it is breached.
MOVE_CANDIDATES = {"move > ln 3 (x3 or /3)": np.log(3), "move > ln 5": np.log(5), "move > ln 10": np.log(10)}
PRICE_CANDIDATES = {"close < $1": 1.0, "close < $2": 2.0, "close < $5": 5.0}
CAP_CANDIDATES = {"cap proxy > $1T": 1e12, "cap proxy > $2T": 2e12}

PREFIX_CUTOFFS = ["2011-06-30", "2013-12-31"]


def load_membership_caps(database, available_by):
    """Quarterly cap proxies whose selection date is on or before `available_by`."""
    with duckdb.connect(str(database), read_only=True) as con:
        caps = con.execute("""SELECT symbol, quarter_start, selection_date, market_cap_proxy
                              FROM universe_membership WHERE selection_date <= ?""",
                           [available_by]).fetchdf()
    for column in ["quarter_start", "selection_date"]:
        caps[column] = pd.to_datetime(caps[column]).astype("datetime64[ns]")
    return caps.rename(columns={"quarter_start": "membership_quarter", "market_cap_proxy": "cap_proxy"})


def compute_measures(panel, caps):
    """Return one row per (symbol, date) with the three causal measures."""
    p = panel[["symbol", "date", "close", "adj_close", "membership_quarter"]].sort_values(["symbol", "date"])
    p = p.reset_index(drop=True)
    log_price = np.log(p.adj_close)
    move = (log_price - log_price.groupby(p.symbol).shift()).abs()
    p["max_abs_log_move_252"] = (move.groupby(p.symbol).rolling(WINDOW, min_periods=1).max()
                                 .reset_index(level=0, drop=True))
    p = p.merge(caps[["symbol", "membership_quarter", "cap_proxy", "selection_date"]],
                on=["symbol", "membership_quarter"], how="left", validate="many_to_one")
    # The quarter's ranking must already be known at the signal close.
    assert (p.selection_date.isna() | p.selection_date.le(p.date)).all()
    return p[["symbol", "date", "close", "max_abs_log_move_252", "cap_proxy"]]


def same_values(a, b):
    return (a.isna() & b.isna()) | (a == b)


def prefix_test(panel, full_measures, cutoff):
    """Measures from data truncated at `cutoff` must equal the full-data measures on earlier dates."""
    cutoff = pd.Timestamp(cutoff)
    truncated = compute_measures(panel.loc[panel.date.le(cutoff)], load_membership_caps(DATABASE, cutoff))
    full = full_measures.loc[full_measures.date.le(cutoff)]
    merged = truncated.merge(full, on=["symbol", "date"], suffixes=("_cut", "_full"), validate="one_to_one")
    assert len(merged) == len(truncated) == len(full)
    for column in ["close", "max_abs_log_move_252", "cap_proxy"]:
        a, b = merged[f"{column}_cut"], merged[f"{column}_full"]
        same = (a.isna() & b.isna()) | (a == b)
        assert same.all(), f"{column} changes when future data is added ({(~same).sum()} rows)"
    return len(merged)


def exclusion_counts(frame):
    """Rows and stocks each candidate would exclude among eligible 2010-2015 rows."""
    year = frame.date.dt.year
    flags = {}
    for name, value in MOVE_CANDIDATES.items():
        flags[name] = frame.max_abs_log_move_252.gt(value)
    for name, value in PRICE_CANDIDATES.items():
        flags[name] = frame.close.lt(value)
    for name, value in CAP_CANDIDATES.items():
        flags[name] = frame.cap_proxy.gt(value)
    rows, stocks = {}, {}
    for name, flag in flags.items():
        rows[name] = (flag.groupby(year).mean() * 100).round(2)
        stocks[name] = frame.loc[flag].groupby(year[flag]).symbol.nunique()
    rows = pd.DataFrame(rows).T
    stocks = pd.DataFrame(stocks).T.fillna(0).astype(int)
    rows.columns, stocks.columns = rows.columns.astype(str), stocks.columns.astype(str)
    return rows, stocks, flags


def main(full=False):
    import psutil
    panel_dir = HERE / "outputs" / ("panel_full" if full else "panel_stage1")
    measures_dir = HERE / "outputs" / ("clean_measures_full" if full else "clean_measures")
    data_end = pd.Timestamp("2026-09-30") if full else DATA_END
    cutoffs = PREFIX_CUTOFFS + (["2016-03-31", "2020-06-30", "2024-12-31"] if full else [])
    tag = "full" if full else "stage2b"

    start = time.perf_counter()
    panel = pd.read_parquet(panel_dir, columns=["symbol", "date", "close", "adj_close", "membership_quarter",
                                                "eligible"])
    assert panel.date.max() <= data_end

    t = time.perf_counter()
    measures = compute_measures(panel, load_membership_caps(DATABASE, data_end))
    print(f"Measures for {len(measures):,} rows in {time.perf_counter() - t:.1f}s")

    t = time.perf_counter()
    for cutoff in cutoffs:
        n = prefix_test(panel, measures, cutoff)
        print(f"Prefix test at {cutoff}: {n:,} earlier rows identical with and without later data  OK")
    print(f"Prefix tests took {time.perf_counter() - t:.1f}s")

    if full:
        # The development measures (built from data ending 2016-03-31) must be reproduced exactly.
        dev = pd.read_parquet(HERE / "outputs" / "clean_measures" / "measures.parquet")
        both = dev.merge(measures.loc[measures.date.le(DATA_END)], on=["symbol", "date"],
                         suffixes=("_dev", "_full"), how="outer", indicator=True)
        assert both._merge.eq("both").all(), "row sets differ before 2016-04-01"
        for column in ["close", "max_abs_log_move_252"]:
            assert same_values(both[f"{column}_dev"], both[f"{column}_full"]).all(), column
        # On the dev panel's last day (2016-03-31) there was no next session, so no
        # membership quarter and no cap proxy; the full panel knows Q2. Compare before it.
        before = both.date.lt(DATA_END)
        assert same_values(both.loc[before, "cap_proxy_dev"], both.loc[before, "cap_proxy_full"]).all()
        print(f"  cap proxy differs only on the dev boundary day {DATA_END.date()}: "
              f"{int((~same_values(both.cap_proxy_dev, both.cap_proxy_full)).sum())} rows")
        print(f"Full-period measures equal the development measures on all {len(both):,} rows through {DATA_END.date()}  OK")

    if measures_dir.exists():
        shutil.rmtree(measures_dir)
    measures_dir.mkdir(parents=True)
    measures.to_parquet(measures_dir / "measures.parquet", index=False)

    # Exclusion counts among eligible rows from 2010 (data preparation, not performance).
    dev = panel.loc[panel.eligible & panel.date.ge("2010-01-01"),
                    ["symbol", "date"]].merge(measures, on=["symbol", "date"], validate="one_to_one")
    rows, stocks, flags = exclusion_counts(dev)
    print(f"\nEligible rows from 2010: {len(dev):,}")
    print("\n% of eligible rows excluded per year:")
    print(rows.to_string())
    print("\nDistinct stocks excluded per year:")
    print(stocks.to_string())

    if not full:
        # Audit only (not used to choose thresholds): do the known broken names get caught?
        known = ["AAM", "VBIV", "LCI", "PALT", "FBC", "MULN", "POAI", "CEAD"]
        audit = pd.DataFrame({name: dev.loc[dev.symbol.isin(known) & flag].groupby("symbol").size()
                              for name, flag in flags.items()}).reindex(known).fillna(0).astype(int)
        audit.insert(0, "eligible rows", dev.loc[dev.symbol.isin(known)].groupby("symbol").size().reindex(known))
        print("\nAudit: rows flagged for the broken names found in stage 2")
        print(audit.T.to_string())

    rows.to_csv(HERE / "outputs" / f"{tag}_pct_rows_excluded.csv")
    stocks.to_csv(HERE / "outputs" / f"{tag}_stocks_excluded.csv")
    info = psutil.Process().memory_info()
    print(f"\nTotal {time.perf_counter() - start:.1f}s; peak memory {getattr(info, 'peak_wset', info.rss) / 1e9:.1f} GB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="use panel_full (through 2026-09-30)")
    main(full=parser.parse_args().full)
