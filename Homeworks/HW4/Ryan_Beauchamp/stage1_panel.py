"""HW4 stage 1: build the feature/target panel from the full homework database.

What this does
--------------
For every stock in ``equities_top2000.duckdb`` it builds one row per market
session with:
  * causal price features (momentum, volatility, drawdown),
  * availability-dated financial features (as-of joined on filing date + 1 day),
  * dated quarterly universe membership (``is_universe_member`` / ``eligible``),
  * the 90-calendar-day target: next-session entry close -> first close on or
    after entry + 90 days.

Reused code
-----------
The feature, membership and label logic is imported unchanged from
``Lectures/modeling/course_data.py`` (Lecture 7/8 course code). This script
only adds ticker batching, a hard data cutoff, dtype normalisation, timing,
and checks. Nothing under ``Lectures/`` is modified.

Why batching is safe here
-------------------------
Every step used below works within one stock at a time (rolling windows,
as-of joins, label lookups) or row by row (membership). So splitting the
stocks into batches gives exactly the same rows as one big run, while
keeping memory bounded. Peer-relative (same-date cross-sectional) features
are *not* computed here; they must be built after all batches are combined.

Data cutoff
-----------
We read nothing dated after ``DATA_END`` (2016-03-31). Development ends
2015-12-31 and the Jan-Mar 2016 buffer lets those targets finish. Targets
whose exit would fall after the cutoff stay missing (censored).

Extra features
--------------
The panel is keyed by (date, symbol), unique. Additional feature columns are
computed in a separate file with the same key and merged on (date, symbol);
the panel never needs rebuilding for them.

Run from the repository root:
    development panel (default, reads nothing after 2016-03-31):
        .venv/Scripts/python -B Homeworks/HW4/Ryan_Beauchamp/stage1_panel.py
    full-period panel for the final test (separate folder):
        .venv/Scripts/python -B Homeworks/HW4/Ryan_Beauchamp/stage1_panel.py --end 2026-09-30 --name panel_full
    (macOS/Linux: .venv/bin/python)

Credits: feature/membership/label functions are from Lectures/modeling/course_data.py
(imported, unmodified). Script written with AI assistance (Claude Code) and
reviewed by the author.
"""

import argparse
import re
import shutil
import sys
import time
import warnings
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

# Importing course_data must not write a __pycache__ folder under Lectures/.
sys.dont_write_bytecode = True

import duckdb
import numpy as np
import pandas as pd
import psutil

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
MODELING_DIR = REPO / "Lectures" / "modeling"
sys.path.insert(0, str(MODELING_DIR))
import course_data as cd  # noqa: E402  (path must be set first)

DATABASE = MODELING_DIR / "data" / "equities_top2000.duckdb"
CLASS_DATABASE = MODELING_DIR / "data" / "equities_class.duckdb"
OUTPUT_DIR = HERE / "outputs"
PANEL_DIR = OUTPUT_DIR / "panel_stage1"  # one parquet file per ticker batch

DATA_END = "2016-03-31"  # last date read from the database (end of buffer)
BATCH_SIZE = 250         # tickers per batch: ~11 batches, a few hundred MB each

# Hand-checkable worked example from Lecture 7, notebook 3.
CHECK = {"symbol": "AAPL", "date": "2012-12-03",
         "momentum_63": -0.1127, "volatility_63": 0.3032, "target": -0.2672}

DATE_COLUMNS = ["date", "fiscal_date", "filing_date", "available_date", "membership_quarter",
                "quarter_end", "universe_selection_date", "entry_date", "scheduled_exit", "label_end"]
FLOAT_COLUMNS = (["close", "adj_close", "volume", "universe_rank", "entry_price", "exit_price", "target"]
                 + cd.NUMERIC_FEATURES)


class StepTimer:
    """Accumulate wall-clock seconds per named step across all batches."""

    def __init__(self):
        self.seconds = defaultdict(float)

    @contextmanager
    def __call__(self, name):
        start = time.perf_counter()
        yield
        self.seconds[name] += time.perf_counter() - start


def list_symbols(database, end):
    """Every ticker with at least one price row on or before the cutoff."""
    with duckdb.connect(str(database), read_only=True) as con:
        rows = con.execute("SELECT DISTINCT symbol FROM prices WHERE date <= ? ORDER BY symbol",
                           [end]).fetchall()
    return [r[0] for r in rows]


def load_calendar(database, end):
    """Market sessions = benchmark dates, the same calendar course_data uses."""
    with duckdb.connect(str(database), read_only=True) as con:
        dates = con.execute("SELECT date FROM benchmark WHERE date <= ? ORDER BY date", [end]).fetchdf().date
    return pd.DatetimeIndex(pd.to_datetime(dates).astype("datetime64[ns]"))


def normalize_dtypes(panel):
    """Give every batch the same column types so the parquet parts line up."""
    for column in DATE_COLUMNS:
        panel[column] = pd.to_datetime(panel[column]).astype("datetime64[ns]")
    for column in FLOAT_COLUMNS:
        panel[column] = pd.to_numeric(panel[column], errors="coerce").astype("float64")
    for column in ["eligible", "is_universe_member"]:
        panel[column] = panel[column].astype(bool)
    panel["symbol"] = panel.symbol.astype(str)
    panel["sector"] = panel.sector.astype(str)
    return panel


def build_batch(batch, calendar, membership, timer, quarantined, end=DATA_END):
    """Run the course pipeline on one ticker batch (same steps as cd.prepare_panel)."""
    with timer("1 load prices + financials (SQL)"):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            prices, fundamentals, universe, _ = cd.load_data(DATABASE, end=end, symbols=batch)
        for w in caught:  # course loader warns when it quarantines bad price rows
            m = re.search(r"Quarantining (\d+) (invalid adjusted-price|price rows outside)", str(w.message))
            if m:
                quarantined["zero/invalid adj_close" if m.group(2).startswith("invalid") else "off-calendar date"] += int(m.group(1))
    with timer("2 price features"):
        panel = cd.price_features(prices, calendar)
    with timer("3 financial features (as-of join)"):
        n_before = len(panel)
        panel = cd.join_financials(panel, cd.financial_events(fundamentals), universe)
        assert len(panel) == n_before, "financial join must not add or drop rows"
    with timer("4 quarterly membership"):
        panel = cd.apply_membership(panel, membership, calendar)
        assert len(panel) == n_before
    with timer("5 90-day targets"):
        panel = cd.make_labels(panel, prices, calendar)
        assert len(panel) == n_before
    return normalize_dtypes(panel)


def summarize_batch(panel):
    """Per-year counts and per-date eligible counts for one batch."""
    has_target = panel.target.notna()
    by_year = (panel.assign(year=panel.date.dt.year,
                            eligible_with_target=panel.eligible & has_target)
               .groupby("year")
               .agg(rows=("symbol", "size"),
                    universe_member_rows=("is_universe_member", "sum"),
                    eligible_rows=("eligible", "sum"),
                    eligible_with_target=("eligible_with_target", "sum")))
    eligible_per_date = panel.loc[panel.eligible].groupby("date").size()
    return by_year, eligible_per_date


def check_worked_example(row):
    """Compare AAPL 2012-12-03 with the Lecture 7 values (rounded to 0.01%)."""
    print("\nWorked example: AAPL signal 2012-12-03")
    ok = True
    for column in ["momentum_63", "volatility_63", "target"]:
        got = float(row[column])
        match = round(got, 4) == CHECK[column]
        ok &= match
        print(f"  {column:<14} got {got:+.4%}   expected {CHECK[column]:+.2%}   {'OK' if match else 'MISMATCH'}")
    print(f"  eligible={bool(row.eligible)}  member={bool(row.is_universe_member)}  "
          f"entry={row.entry_date.date()}  exit={row.label_end.date()}")
    return ok


def compare_with_class_db(panel_dir, timer):
    """Cross-check: the 30 class stocks should get identical values in 2010-2015.

    The class extract starts in 2008, so by 2010 every lookback is warm and the
    two databases should agree exactly. Class targets stop at 2015-12-31, so we
    only compare targets that exist in both.
    """
    with timer("7 cross-check vs class database"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            small = cd.prepare_panel(CLASS_DATABASE, end="2015-12-31")
        small = small.loc[small.date.between("2010-01-01", "2015-12-31")]
        columns = ["symbol", "date", "eligible", "is_universe_member", "target"] + cd.NUMERIC_FEATURES
        full = pd.read_parquet(panel_dir, columns=columns,
                               filters=[("symbol", "in", sorted(small.symbol.unique())),
                                        ("date", ">=", pd.Timestamp("2010-01-01")),
                                        ("date", "<=", pd.Timestamp("2015-12-31"))])
        both = small[columns].merge(full, on=["symbol", "date"], suffixes=("_class", "_full"),
                                    how="outer", indicator=True)
        report = {"class rows": len(small), "matched rows": int(both._merge.eq("both").sum())}
        both = both.loc[both._merge.eq("both")]
        for column in ["eligible", "is_universe_member"]:
            report[f"{column} differences"] = int((both[f"{column}_class"] != both[f"{column}_full"]).sum())
        worst = 0.0
        for column in cd.NUMERIC_FEATURES:
            a, b = both[f"{column}_class"], both[f"{column}_full"]
            report[f"{column} missing-pattern differences"] = int((a.isna() != b.isna()).sum())
            diff = (a - b).abs().max()
            if pd.notna(diff):
                worst = max(worst, float(diff))
        t = both.target_class.notna() & both.target_full.notna()
        report["targets compared"] = int(t.sum())
        report["max |target diff|"] = float((both.target_class[t] - both.target_full[t]).abs().max())
        report["max |feature diff|"] = worst
    return report


def peak_memory_gb():
    """Peak resident memory of this process so far (Windows: peak working set)."""
    info = psutil.Process().memory_info()
    return getattr(info, "peak_wset", info.rss) / 1e9


def main(end=DATA_END, panel_name="panel_stage1", tag="stage1"):
    panel_dir = OUTPUT_DIR / panel_name
    timer = StepTimer()
    quarantined = defaultdict(int)
    total_start = time.perf_counter()

    with timer("0 setup (symbols, calendar, membership)"):
        symbols = list_symbols(DATABASE, end)
        calendar = load_calendar(DATABASE, end)
        membership = cd.load_membership(DATABASE, end=end)
    print(f"{len(symbols):,} tickers with prices on/before {end}; "
          f"{len(calendar):,} sessions {calendar[0].date()} to {calendar[-1].date()}; "
          f"{len(membership):,} membership rows through {membership.quarter_start.max().date()}")

    if panel_dir.exists():  # rerun: replace only this panel's own previous output
        shutil.rmtree(panel_dir)
    panel_dir.mkdir(parents=True)

    yearly, per_date = [], []
    example = None
    n_rows = 0
    batches = [symbols[i:i + BATCH_SIZE] for i in range(0, len(symbols), BATCH_SIZE)]
    for number, batch in enumerate(batches, start=1):
        batch_start = time.perf_counter()
        panel = build_batch(batch, calendar, membership, timer, quarantined, end=end)

        # Hard guard on the cutoff rule: no row or outcome after `end`.
        assert panel.date.max() <= pd.Timestamp(end)
        assert panel.label_end.max() <= pd.Timestamp(end)
        assert not panel.duplicated(["date", "symbol"]).any(), "(date, symbol) must be a unique merge key"

        with timer("6 write parquet"):
            panel.to_parquet(panel_dir / f"part-{number:03d}.parquet", index=False)
        by_year, eligible_per_date = summarize_batch(panel)
        yearly.append(by_year)
        per_date.append(eligible_per_date)
        n_rows += len(panel)
        hit = panel.loc[panel.symbol.eq(CHECK["symbol"]) & panel.date.eq(CHECK["date"])]
        if len(hit):
            example = hit.iloc[0]
        print(f"  batch {number:>2}/{len(batches)}: {batch[0]}..{batch[-1]}  "
              f"{len(panel):>9,} rows  {time.perf_counter() - batch_start:5.1f}s  "
              f"peak mem {peak_memory_gb():.1f} GB")
        del panel

    # --- Combine batch summaries ---------------------------------------------------
    counts = pd.concat(yearly).groupby(level=0).sum().astype(int)
    eligible_per_date = pd.concat(per_date).groupby(level=0).sum()
    counts["avg_eligible_per_session"] = (eligible_per_date.groupby(eligible_per_date.index.year).mean()
                                          .reindex(counts.index).round(0))
    counts.loc["total"] = counts.sum(numeric_only=True)
    counts.loc["total", "avg_eligible_per_session"] = np.nan
    assert eligible_per_date.max() <= 2000, "more eligible stocks on a date than the universe allows"

    # Count rows from parquet metadata instead of loading the whole panel (memory).
    with timer("8 count rows from parquet metadata"):
        import pyarrow.parquet as pq
        stored = sum(pq.ParquetFile(f).metadata.num_rows for f in sorted(panel_dir.glob("*.parquet")))
    assert stored == n_rows

    print("\nQuarantined price rows (course loader):", dict(quarantined) or "none")
    example_ok = check_worked_example(example) if example is not None else False
    cross = compare_with_class_db(panel_dir, timer)

    print("\nCross-check vs class database (30 stocks, 2010-2015):")
    for key, value in cross.items():
        print(f"  {key:<45} {value:,}" if isinstance(value, int) else f"  {key:<45} {value:.2e}")

    print("\nRows per year (signal date):")
    print(counts.to_string())
    print(f"\nPanel: {n_rows:,} rows saved to {panel_dir}; peak process memory {peak_memory_gb():.1f} GB")

    timings = pd.Series(timer.seconds).sort_index()
    timings.loc["total wall clock"] = time.perf_counter() - total_start
    print("\nTime per step (seconds, summed over batches):")
    print(timings.round(1).to_string())

    counts.to_csv(OUTPUT_DIR / f"{tag}_rows_per_year.csv")
    timings.rename("seconds").to_csv(OUTPUT_DIR / f"{tag}_timings.csv")
    if not example_ok:
        raise SystemExit("Worked example did not match; stop and investigate.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--end", default=DATA_END, help="last date read from the database")
    parser.add_argument("--name", default="panel_stage1", help="output folder under outputs/")
    args = parser.parse_args()
    if args.name == "panel_stage1" and args.end != DATA_END:
        raise SystemExit("Refusing to overwrite the development panel with a different end date; pass --name.")
    main(end=args.end, panel_name=args.name, tag="stage1" if args.name == "panel_stage1" else args.name)
