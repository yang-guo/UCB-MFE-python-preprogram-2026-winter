"""Run B: EXPLORATORY, run after seeing Run A. Not the headline result.

Same procedure as final_run.py (Run A) with one change: the model uses only six
GARP columns (fcf_yield, ebitda_yield, earnings_yield, rev_growth_ttm,
eps_change_yield, op_margin_ttm), without sector and without the 13 baseline
features. Everything else comes unchanged from config_run_b.json, which copies
Run A's frozen settings: clean flag, rank-target Ridge (alpha 1000), training
from 2010, annual refits, top 50, 2% total cap, every 5 sessions, 10 bps,
"hold" stale-quote rule, first signal 2016-03-31, first fill and benchmark
2016-04-01, end 2026-09-30.

Reuses walkforward.py and backtester.py unchanged; writes only to outputs/run_b/
(or outputs/run_b_smoke/ with --smoke, development data 2013-06-28 to 2015-12-31).
Run A's files are never written.

Run from the repository root:
    .venv/Scripts/python -B Homeworks/HW4/Ryan_Beauchamp/run_b.py --smoke
    .venv/Scripts/python -B Homeworks/HW4/Ryan_Beauchamp/run_b.py

Credits: make_pipeline from Lectures/modeling/course_data.py (via walkforward.py);
backtester structure modeled on Lectures/modeling/backtest.py. Written with AI
assistance (Claude Code) and reviewed by the author.
"""

import argparse
import hashlib
import json
import sys
import time
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

CONFIG = HERE / "config_run_b.json"
DATABASE = HERE.parents[2] / "Lectures" / "modeling" / "data" / "equities_top2000.duckdb"
LABEL = "Run B: exploratory, run after seeing Run A"
GARP6 = ["fcf_yield", "ebitda_yield", "earnings_yield", "rev_growth_ttm", "eps_change_yield", "op_margin_ttm"]
SMOKE = {"dates": {"first_signal": "2013-06-28", "first_fill": "2013-07-01", "end": "2015-12-31"},
         "data": {"panel": "panel_stage1", "measures": "clean_measures/measures.parquet",
                  "extra_features_path": "garp_dev/garp_features.parquet", "output_dir": "run_b_smoke"}}


def peak_gb():
    info = psutil.Process().memory_info()
    return getattr(info, "peak_wset", info.rss) / 1e9


def yearly_returns(navs):
    """Calendar-year returns for each NAV column (first year from the first history row)."""
    rows = {}
    for y in sorted(navs.index.year.unique()):
        prev = navs.loc[navs.index < pd.Timestamp(f"{y}-01-01")]
        base = prev.iloc[-1] if len(prev) else navs.loc[navs.index.year == y].iloc[0]
        rows[y] = navs.loc[navs.index.year == y].iloc[-1] / base - 1
    out = pd.DataFrame(rows).T
    out.loc["whole period"] = navs.iloc[-1] / navs.iloc[0] - 1
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    smoke = parser.parse_args().smoke
    raw = CONFIG.read_bytes()
    config, digest = json.loads(raw), hashlib.sha256(raw).hexdigest()
    assert config["status"] == "EXPLORATORY" and config["features"] == GARP6 and not config["include_sector"]
    assert np.isclose(config["clean_flag"]["max_abs_log_move_252"], wf.MAX_LOG_MOVE)
    assert config["clean_flag"]["min_close"] == wf.MIN_CLOSE
    if smoke:
        config["dates"], config["data"] = dict(SMOKE["dates"]), dict(SMOKE["data"])
    out = HERE / "outputs" / config["data"]["output_dir"]
    assert out.name in ("run_b", "run_b_smoke"), "Run B must never write into Run A's folders"
    out.mkdir(parents=True, exist_ok=True)
    start, timer = time.perf_counter(), {}
    print(f"=== {LABEL}{' (SMOKE, development data)' if smoke else ''} ===")

    first_signal, end = pd.Timestamp(config["dates"]["first_signal"]), pd.Timestamp(config["dates"]["end"])
    s, bt = config["strategy"], config["backtest"]
    with duckdb.connect(str(DATABASE), read_only=True) as con:
        bench = con.execute("SELECT date, adj_close FROM benchmark WHERE date BETWEEN ? AND ? ORDER BY date",
                            [first_signal.date(), end.date()]).fetchdf()
    bench = bench.assign(date=pd.to_datetime(bench.date)).set_index("date").adj_close
    sessions = pd.DatetimeIndex(bench.index)
    assert sessions[0] == first_signal and sessions[1] == pd.Timestamp(config["dates"]["first_fill"])
    last_signal = sessions[-2]

    t = time.perf_counter()
    frame, n_broken = wf.load_model_rows(HERE / "outputs" / config["data"]["panel"],
                                         HERE / "outputs" / config["data"]["measures"],
                                         config["model"]["train_start"], last_signal,
                                         extra_path=HERE / "outputs" / config["data"]["extra_features_path"],
                                         extra_columns=GARP6)
    timer["load model rows"] = time.perf_counter() - t
    print(f"Model rows: {len(frame):,} (clean flag excluded {n_broken:,}); GARP coverage "
          + ", ".join(f"{c} {frame[c].notna().mean():.0%}" for c in GARP6) + f"; peak mem {peak_gb():.1f} GB")

    t = time.perf_counter()
    fits = wf.fit_dates(sessions, first_signal, last_signal, config["model"]["retraining_rule"])
    scores, fit_log = wf.walk_forward_scores(frame, fits, last_signal, alpha=config["model"]["alpha"],
                                             train_start=config["model"]["train_start"],
                                             numeric_features=GARP6, include_sector=False)
    assert pd.Timestamp(fit_log.iloc[0].last_train_label_end) < first_signal
    assert scores.date.min() == first_signal and scores.date.max() <= last_signal
    print("\nFit-date table (each fit uses only outcomes with label_end < fit date):")
    print(fit_log[["fit_date", "train_rows", "last_train_signal", "last_train_label_end",
                   "scored_from", "scored_to", "scored_rows"]].to_string(index=False))
    scores.to_parquet(out / "scores.parquet", index=False)
    fit_log.to_csv(out / "fit_log.csv", index=False)
    timer["fit + score"] = time.perf_counter() - t
    del frame

    long = pd.read_parquet(HERE / "outputs" / config["data"]["panel"], columns=["date", "symbol", "adj_close"],
                           filters=[("date", ">=", first_signal), ("date", "<=", end)])
    prices = long.pivot(index="date", columns="symbol", values="adj_close").reindex(sessions)
    del long

    t = time.perf_counter()
    strategy = ScoreStrategy(wf.scores_by_date(scores), top_n=s["top_n"],
                             cohort_fraction=s["entry_every"] / s["sessions_per_hold"], per_name_cap=s["per_name_cap"])
    result = Backtester(prices, strategy, initial_cash=bt["initial_cash"], fee_bps=bt["fee_bps"],
                        holding_days=bt["holding_days"], entry_every=s["entry_every"],
                        max_stale_sessions=bt["max_stale_sessions"],
                        jump_threshold=config["clean_flag"]["max_abs_log_move_252"],
                        long_gap_policy=bt["long_gap_policy"]).run()
    timer["backtest"] = time.perf_counter() - t

    h, f, o = result.history, result.fills, result.orders
    checks = accounting_checks(result)
    checks[f"history starts {first_signal.date()} with initial capital, no fills"] = bool(
        h.index[0] == first_signal and h.nav.iloc[0] == bt["initial_cash"] and h.buys.iloc[0] == 0)
    checks[f"first fills on {sessions[1].date()}"] = bool(f.date.min() == sessions[1])
    checks["no fill after the end date"] = bool(f.date.max() <= end)
    checks["no order planned on the last session"] = bool(o.signal_date.max() < sessions[-1])
    checks["every refit used outcomes ended before its fit date"] = bool(
        (pd.to_datetime(fit_log.last_train_label_end) < pd.to_datetime(fit_log.fit_date)).all())

    bench_nav = benchmark_nav(bench, h.index, sessions[1], bt["initial_cash"])
    frozen = float(result.long_gaps.value_at_end.sum()) if len(result.long_gaps) else 0.0
    columns = {"Run B (exploratory)": performance(h.nav, annual_rf=bt["sharpe_rf"])}
    navs = pd.DataFrame({"Run B (exploratory)": h.nav})
    run_a_path = HERE / "outputs" / "final" / "history.csv"
    if not smoke:
        a = pd.read_csv(run_a_path, parse_dates=["date"], index_col="date")       # read only
        assert a.index.equals(h.index)
        columns["Run A (frozen headline)"] = performance(a.nav, annual_rf=bt["sharpe_rf"])
        navs["Run A (frozen headline)"] = a.nav
    columns["^SP500TR"] = performance(bench_nav, annual_rf=bt["sharpe_rf"])
    navs["^SP500TR"] = bench_nav
    metrics = pd.DataFrame(columns)
    metrics.loc["frozen_lots_value_at_end"] = [frozen] + [np.nan] * (metrics.shape[1] - 1)
    metrics.loc["final_value_if_frozen_lots_worth_zero"] = [h.nav.iloc[-1] - frozen] + [np.nan] * (metrics.shape[1] - 1)
    if not smoke:
        a_metrics = pd.read_csv(HERE / "outputs" / "final" / "metrics.csv", index_col=0)
        metrics.loc["frozen_lots_value_at_end", "Run A (frozen headline)"] = float(a_metrics.loc["frozen_lots_value_at_end", "strategy"])
        metrics.loc["final_value_if_frozen_lots_worth_zero", "Run A (frozen headline)"] = float(
            a_metrics.loc["final_value_if_frozen_lots_worth_zero", "strategy"])
    years = yearly_returns(navs)

    h.assign(benchmark_nav=bench_nav).to_csv(out / "history.csv")
    for name, df in [("fills", f), ("orders", o), ("closed_lots", result.closed_lots), ("open_lots", result.open_lots),
                     ("stale_marks", result.stale), ("jump_flags", result.jumps), ("long_gaps", result.long_gaps)]:
        df.to_csv(out / f"{name}.csv", index=False)
    metrics.to_csv(out / "metrics.csv")
    years.to_csv(out / "yearly_returns.csv")
    (out / "accounting_checks.json").write_text(json.dumps(
        {"label": LABEL, "config": CONFIG.name, "config_sha256": digest, "checks": checks}, indent=2), encoding="utf-8")
    (out / "README.txt").write_text(LABEL + ". Not the headline; Run A (outputs/final, results/) is the frozen result.\n",
                                    encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 5))
    for name in navs.columns:
        ax.plot(navs.index, navs[name] / bt["initial_cash"] - 1, label=name, lw=1.3,
                ls="--" if name.startswith("Run B") else "-")
    ax.axhline(0, color="grey", lw=0.6)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_title(f"{LABEL}\nCumulative return from the {first_signal.date()} close to {h.index[-1].date()}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "cumulative_returns_run_b.png", dpi=150)

    timer["total"] = time.perf_counter() - start
    print("\nAccounting and timing checks:")
    for name, ok in checks.items():
        print(f"  {'OK  ' if ok else 'FAIL'} {name}")
    print(f"\nOrders {len(o):,} (rejected, no entry quote: {(o.status != 'filled').sum():,}); fills {len(f):,}; "
          f"open lots at end {len(result.open_lots)}; stale-mark rows {len(result.stale):,}; "
          f"jump flags {len(result.jumps)}; long quote gaps {len(result.long_gaps)} "
          f"({result.long_gaps.symbol.nunique() if len(result.long_gaps) else 0} stocks); fees ${f.fee.sum():,.0f}")
    if len(result.jumps):
        print(result.jumps.to_string(index=False))
    print("\nMetrics:")
    print(metrics.to_string())
    print("\nCalendar-year returns:")
    print(years.to_string(float_format=lambda x: f"{x:+.2%}"))
    print("\nTime (s):", {k: round(v) for k, v in timer.items()}, f"peak mem {peak_gb():.1f} GB")
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
