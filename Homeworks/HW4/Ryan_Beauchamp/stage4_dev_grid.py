"""HW4 task 2: development-only strategy grid and retraining comparison (2012-2015).

Uses only the development panel (data through 2016-03-31) and signals through
2015-12-31. Backtests run from the 2012-01-03 close (initial-capital row, first
decision) to 2015-12-31; positions still open then are valued at that close.

a. Walk-forward rank-target Ridge scores, three retraining rules:
     once      - one fit on 2012-01-03, scores 2012-2015
     annual    - refit at the first session of each year (= stage 2c folds)
     quarterly - refit at the first session of each quarter
   Each fit uses only outcomes with label_end < fit date; every clean eligible
   row is scored, with or without a known target.
b. Grid (exactly as specified): top N {20, 50} x per-name cap {2%, 5%} x entry
   every {5, 21} sessions, fee 10 bps, on the annual scores.
   Cohort fraction = entry_every / 63 (about 63 sessions in 90 calendar days),
   so each cohort is an equal slice and the book is roughly fully invested.
c. The same grid on the "once" and "quarterly" scores, plus rank IC and the
   top-5% gap for each rule.
Every backtest is appended to outputs/stage4_run_log.csv, including failures.

Run from the repository root:
    .venv/Scripts/python -B Homeworks/HW4/Ryan_Beauchamp/stage4_dev_grid.py

Credits: make_pipeline/FEATURES from Lectures/modeling/course_data.py (via
walkforward.py). Written with AI assistance (Claude Code) and reviewed by the author.
"""

import itertools
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

sys.dont_write_bytecode = True

import duckdb
import numpy as np
import pandas as pd
import psutil

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import walkforward as wf  # noqa: E402
from backtester import Backtester, ScoreStrategy, accounting_checks, benchmark_nav, performance  # noqa: E402

OUT = HERE / "outputs"
DATABASE = HERE.parents[2] / "Lectures" / "modeling" / "data" / "equities_top2000.duckdb"
PANEL = OUT / "panel_stage1"
MEASURES = OUT / "clean_measures" / "measures.parquet"
DATA_END = pd.Timestamp("2016-03-31")
BT_START, BT_END = pd.Timestamp("2012-01-03"), pd.Timestamp("2015-12-31")
RULES = ["once", "annual", "quarterly"]
GRID = list(itertools.product([20, 50], [0.02, 0.05], [5, 21]))   # top_n, cap, entry_every
FEE_BPS, MAX_STALE, SESSIONS_PER_HOLD = 10, 5, 63
RUN_LOG = OUT / "stage4_run_log.csv"


def peak_gb():
    info = psutil.Process().memory_info()
    return getattr(info, "peak_wset", info.rss) / 1e9


def get_scores(frame, rule):
    path = OUT / f"stage4_scores_{rule}.parquet"
    if path.exists():
        print(f"  {rule}: cached {path.name}")
        return pd.read_parquet(path), pd.read_csv(OUT / f"stage4_fits_{rule}.csv")
    fits = wf.fit_dates(frame.date.unique(), BT_START, BT_END, rule)
    print(f"  {rule}: {len(fits)} fit(s)")
    scores, fit_log = wf.walk_forward_scores(frame, fits, BT_END, log=print)
    scores.to_parquet(path, index=False)
    fit_log.to_csv(OUT / f"stage4_fits_{rule}.csv", index=False)
    return scores, fit_log


def prediction_quality(frame, scores):
    """Daily rank IC and top-5% gap (raw and per-date 1/99% winsorized) on rows with targets."""
    s = scores.merge(frame[["date", "symbol", "target"]], on=["date", "symbol"]).dropna(subset=["target"])
    lo = s.groupby("date").target.transform(lambda x: x.quantile(0.01))
    hi = s.groupby("date").target.transform(lambda x: x.quantile(0.99))
    s["wins"] = s.target.clip(lo, hi)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ic = s.groupby("date")[["score", "target"]].apply(lambda x: x.score.corr(x.target, method="spearman"))
    s["top"] = s.score.ge(s.groupby("date").score.transform(lambda x: x.quantile(0.95)))
    gap = {}
    for column in ["target", "wins"]:
        m = s.groupby(["date", "top"])[column].mean().unstack()
        gap[column] = m[True] - m[False]
    by_year = pd.DataFrame({"rank_IC": ic, "top5_gap_raw": gap["target"], "top5_gap_wins": gap["wins"]})
    out = by_year.groupby(by_year.index.year).mean()
    out.loc["2012-2015"] = by_year.mean()
    return out


def run_backtest(scores_map, prices, bench, rule, top_n, cap, every):
    params = dict(rule=rule, top_n=top_n, per_name_cap=cap, entry_every=every,
                  cohort_fraction=round(every / SESSIONS_PER_HOLD, 4), fee_bps=FEE_BPS)
    row = {"logged_at": datetime.now().isoformat(timespec="seconds"), **params}
    t = time.perf_counter()
    try:
        strategy = ScoreStrategy(scores_map, top_n=top_n, cohort_fraction=every / SESSIONS_PER_HOLD,
                                 per_name_cap=cap)
        r = Backtester(prices, strategy, fee_bps=FEE_BPS, entry_every=every,
                       max_stale_sessions=MAX_STALE).run()
        checks = accounting_checks(r)
        m = performance(r.history.nav)
        row.update(status="ok" if all(checks.values()) else "ACCOUNTING CHECK FAILED",
                   final_value=m.final_value, CAGR=m.CAGR, max_drawdown=m.max_drawdown, sharpe=m.sharpe,
                   ann_vol=m.ann_volatility, avg_invested=(r.history.holdings / r.history.nav).mean(),
                   fills=len(r.fills), rejected=int((r.orders.status != "filled").sum()),
                   stale_rows=len(r.stale), jumps=len(r.jumps), open_lots_end=len(r.open_lots),
                   fees=r.fills.fee.sum(), checks_failed=[k for k, v in checks.items() if not v] or "")
        if len(r.jumps):
            r.jumps.assign(**params).to_csv(OUT / f"stage4_jumps_{rule}_{top_n}_{cap}_{every}.csv", index=False)
    except Exception as exc:  # log bad runs too
        row.update(status=f"failed: {type(exc).__name__}: {exc}")
    row["seconds"] = round(time.perf_counter() - t, 1)
    pd.DataFrame([row]).to_csv(RUN_LOG, mode="a", header=not RUN_LOG.exists(), index=False)
    return row


def main():
    start = time.perf_counter()
    frame, n_broken = wf.load_model_rows(PANEL, MEASURES, "2010-01-01", "2015-12-31")
    assert frame.date.max() <= BT_END and frame.label_end.max() <= DATA_END
    print(f"Clean eligible rows 2010-2015: {len(frame):,} (excluded by clean flag: {n_broken:,}); "
          f"peak mem {peak_gb():.1f} GB")

    print("\n(a, c) Walk-forward scores:")
    scores, fits = {}, {}
    for rule in RULES:
        scores[rule], fits[rule] = get_scores(frame, rule)

    # Consistency: annual 2015 scores must equal the stage 3 dry-run scores (same fold, same model).
    dry = pd.read_parquet(OUT / "stage3_scores_2015.parquet")
    a15 = scores["annual"].loc[scores["annual"].date.dt.year.eq(2015)]
    both = dry.merge(a15, on=["date", "symbol"], suffixes=("_dry", "_wf"), how="outer", indicator=True)
    assert both._merge.eq("both").all()
    print(f"  annual 2015 scores vs stage 3 dry-run scores: max |diff| = "
          f"{(both.score_dry - both.score_wf).abs().max():.2e} on {len(both):,} rows")

    quality = pd.concat({rule: prediction_quality(frame, scores[rule]) for rule in RULES}, axis=1)
    print("\nPrediction quality by retraining rule (rows with known targets):")
    print(quality.to_string(float_format=lambda x: f"{x:+.4f}"))
    quality.to_csv(OUT / "stage4_prediction_quality.csv")

    with duckdb.connect(str(DATABASE), read_only=True) as con:
        bench = con.execute("SELECT date, adj_close FROM benchmark WHERE date BETWEEN ? AND ? ORDER BY date",
                            [BT_START.date(), BT_END.date()]).fetchdf()
    bench = bench.assign(date=pd.to_datetime(bench.date)).set_index("date").adj_close
    long = pd.read_parquet(PANEL, columns=["date", "symbol", "adj_close"],
                           filters=[("date", ">=", BT_START), ("date", "<=", BT_END)])
    prices = long.pivot(index="date", columns="symbol", values="adj_close").reindex(bench.index)
    print(f"\nPrice table {prices.shape[0]} sessions x {prices.shape[1]:,} symbols; peak mem {peak_gb():.1f} GB")
    bench_metrics = performance(benchmark_nav(bench, bench.index, bench.index[1]))
    print(f"Benchmark ^SP500TR (invested {bench.index[1].date()} close): final {bench_metrics.final_value:,.0f}, "
          f"CAGR {bench_metrics.CAGR:.2%}, maxDD {bench_metrics.max_drawdown:.2%}, Sharpe {bench_metrics.sharpe:.2f}")

    print("\n(b, c) Backtests:")
    rows = []
    for rule in RULES:
        smap = wf.scores_by_date(scores[rule])
        for top_n, cap, every in GRID:
            row = run_backtest(smap, prices, bench, rule, top_n, cap, every)
            rows.append(row)
            print(f"  {rule:<9} N={top_n:<2} cap={cap:.0%} every={every:<2}  {row['status'][:60]:<12} "
                  + (f"CAGR {row['CAGR']:+.2%}  Sharpe {row['sharpe']:.2f}  maxDD {row['max_drawdown']:.1%}"
                     if row["status"] == "ok" else ""))
    results = pd.DataFrame(rows)
    results.to_csv(OUT / "stage4_grid_results.csv", index=False)
    pd.Series(bench_metrics).to_csv(OUT / "stage4_benchmark_2012_2015.csv")
    print(f"\nTotal {time.perf_counter() - start:.0f}s; peak memory {peak_gb():.1f} GB")


if __name__ == "__main__":
    main()
