"""HW4 stage 3: development-period dry run of the backtester (2015 only).

PLUMBING SMOKE TEST, NOT A RESULT. It checks that scores, prices, the account
and analytics fit together on real data. One year of one weak model says
nothing about which strategy to choose.

Steps
1. Fit the stage 2c rank-target Ridge (alpha=1000) for the 2015 fold: training
   rows are clean eligible signals from 2010 whose outcome ended before the
   first 2015 signal (2015-01-02).
2. Score every clean eligible row dated in 2015, whether or not its target is
   known, so selection never depends on future label completeness.
3. Backtest signals and fills inside 2015: first decision 2015-01-02, last
   session 2015-12-31. Lots still open at the end are valued at their last mark.
4. Benchmark: $1,000,000 in ^SP500TR at the first fill close (2015-01-05).

Run from the repository root:
    .venv/Scripts/python -B Homeworks/HW4/Ryan_Beauchamp/stage3_dry_run.py

Credits: date_folds and make_pipeline come from Lectures/modeling/course_data.py
(imported, unmodified). Written with AI assistance (Claude Code) and reviewed
by the author.
"""

import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True

import duckdb
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import stage2_validation as s2  # noqa: E402  (puts course_data on the path)
import stage2c_clean_validation as s2c  # noqa: E402
import course_data as cd  # noqa: E402
from backtester import (Backtester, ScoreStrategy, accounting_checks, benchmark_nav,  # noqa: E402
                        performance)

DATABASE = HERE.parents[2] / "Lectures" / "modeling" / "data" / "equities_top2000.duckdb"
OUTPUT_DIR = HERE / "outputs"
START, END = pd.Timestamp("2015-01-02"), pd.Timestamp("2015-12-31")
PARAMS = dict(top_n=50, cohort_fraction=1 / 3, per_name_cap=0.02)   # strategy
BT = dict(fee_bps=10, entry_every=21, max_stale_sessions=5, holding_days=90)  # backtester


def make_scores(timer):
    t = time.perf_counter()
    frame = s2c.load_clean_rows()
    tr, _ = cd.date_folds(frame, validation_years=(2015,), train_start=s2.DEV_START)[0]
    train = frame.iloc[tr]
    assert train.label_end.max() < START
    model = cd.make_pipeline(Ridge(alpha=1000)).fit(train[cd.FEATURES], s2c.date_rank(train, train.target))
    score_rows = frame.loc[frame.date.between(START, END)]           # all clean eligible rows, target or not
    scores = score_rows[["date", "symbol"]].assign(score=model.predict(score_rows[cd.FEATURES]))
    scores.to_parquet(OUTPUT_DIR / "stage3_scores_2015.parquet", index=False)
    timer["fit + score 2015"] = time.perf_counter() - t
    print(f"Fit on {len(train):,} rows (last label_end {train.label_end.max().date()}); "
          f"scored {len(scores):,} rows on {scores.date.nunique()} dates "
          f"({score_rows.target.isna().sum():,} of them have no target yet: kept)")
    return {d: g.set_index("symbol").score for d, g in scores.groupby("date")}


def load_prices_and_benchmark():
    with duckdb.connect(str(DATABASE), read_only=True) as con:
        bench = con.execute("SELECT date, adj_close FROM benchmark WHERE date BETWEEN ? AND ? ORDER BY date",
                            [START.date(), END.date()]).fetchdf()
    bench = bench.assign(date=pd.to_datetime(bench.date)).set_index("date").adj_close
    calendar = pd.DatetimeIndex(bench.index)
    # Quotes from the stage 1 panel (course loader already quarantined zero and off-calendar rows).
    long = pd.read_parquet(HERE / "outputs" / "panel_stage1", columns=["date", "symbol", "adj_close"],
                           filters=[("date", ">=", START), ("date", "<=", END)])
    prices = long.pivot(index="date", columns="symbol", values="adj_close").reindex(calendar)
    return prices, bench


def main():
    timer = {}
    start = time.perf_counter()
    scores = make_scores(timer)

    t = time.perf_counter()
    prices, bench = load_prices_and_benchmark()
    timer["load prices"] = time.perf_counter() - t
    print(f"Price table: {prices.shape[0]} sessions x {prices.shape[1]:,} symbols")

    t = time.perf_counter()
    result = Backtester(prices, ScoreStrategy(scores, **PARAMS), **BT).run()
    timer["backtest loop"] = time.perf_counter() - t

    h = result.history
    first_fill = result.fills.date.min()
    bench_nav = benchmark_nav(bench, h.index, first_fill)

    print("\nAccounting checks:")
    for name, ok in accounting_checks(result).items():
        print(f"  {'OK  ' if ok else 'FAIL'} {name}")
    assert h.nav.iloc[0] == result.initial_cash and h.buys.iloc[0] == 0, "first row must be initial capital"
    print(f"  OK   first history row {h.index[0].date()} is initial capital (${h.nav.iloc[0]:,.0f}, no fills); "
          f"first fill {first_fill.date()}; benchmark invests that close")

    o = result.orders
    print("\nActivity:")
    print(f"  decisions: {o.signal_date.nunique()}  orders: {len(o)}  filled: {(o.status == 'filled').sum()}  "
          f"rejected (no entry quote): {(o.status != 'filled').sum()}")
    print(f"  sells: {(result.fills.side == 'sell').sum()}  open lots at end: {len(result.open_lots)}  "
          f"fees paid: ${result.fills.fee.sum():,.0f}")
    print(f"  average invested share of NAV: {(h.holdings / h.nav).mean():.1%}  "
          f"(max {(h.holdings / h.nav).max():.1%})")
    print(f"  stale-mark rows: {len(result.stale)} (max run {result.stale.stale_sessions.max() if len(result.stale) else 0})  "
          f"jump flags on held stocks: {len(result.jumps)}")
    if len(result.jumps):
        print(result.jumps.to_string(index=False))

    metrics = pd.DataFrame({"strategy (dry run)": performance(h.nav), "^SP500TR": performance(bench_nav)})
    print("\nMetrics, 2015 only (smoke test, NOT a strategy result):")
    print(metrics.to_string())

    result.history.to_csv(OUTPUT_DIR / "stage3_dry_run_history.csv")
    result.fills.to_csv(OUTPUT_DIR / "stage3_dry_run_fills.csv", index=False)
    result.orders.to_csv(OUTPUT_DIR / "stage3_dry_run_orders.csv", index=False)
    result.jumps.to_csv(OUTPUT_DIR / "stage3_dry_run_jumps.csv", index=False)
    result.stale.to_csv(OUTPUT_DIR / "stage3_dry_run_stale.csv", index=False)
    metrics.to_csv(OUTPUT_DIR / "stage3_dry_run_metrics.csv")

    timer["total"] = time.perf_counter() - start
    print("\nTime (seconds):", {k: round(v, 1) for k, v in timer.items()})


if __name__ == "__main__":
    main()
