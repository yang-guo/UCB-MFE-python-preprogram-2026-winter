"""HW4 final test: one run, April 1 2016 to September 30 2026, from the frozen config.

DO NOT RUN until the strategy is frozen. The script refuses to start unless
frozen_config.json has "status": "FROZEN" and --i-confirm-frozen is passed.

Convention
* First signal after the 2016-03-31 close; the history's initial-capital row is
  dated 2016-03-31. First fills, and the benchmark purchase, at the 2016-04-01 close.
* The model scoring those first signals is fitted only on outcomes that ended
  before 2016-03-31 (label_end < fit date). Later refits follow the frozen
  retraining rule with the same restriction at each fit date.
* Every clean eligible row is scored, with or without a known target.
* No order is planned on the last session (2026-09-30); nothing fills after it.
  Lots still open on 2026-09-30 are valued at that close (or their last mark).
* Benchmark: $1,000,000 in ^SP500TR at the 2016-04-01 close, held to 2026-09-30.

Outputs (outputs/final/): scores, fit log, daily history, fills, orders, open
lots, stale marks, jump flags, accounting checks, metrics, and a chart of both
cumulative-return curves.

Run from the repository root (after freezing):
    .venv/Scripts/python -B Homeworks/HW4/Ryan_Beauchamp/final_run.py --i-confirm-frozen

Credits: make_pipeline/FEATURES from Lectures/modeling/course_data.py (via
walkforward.py); backtester structure modeled on Lectures/modeling/backtest.py.
Written with AI assistance (Claude Code) and reviewed by the author.
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
from backtester import (Backtester, ScoreStrategy, StaleQuoteError, accounting_checks,  # noqa: E402
                        benchmark_nav, performance)

CONFIG = HERE / "frozen_config.json"
OUT = HERE / "outputs" / "final"
DATABASE = HERE.parents[2] / "Lectures" / "modeling" / "data" / "equities_top2000.duckdb"


def peak_gb():
    info = psutil.Process().memory_info()
    return getattr(info, "peak_wset", info.rss) / 1e9


# Smoke window spans two year boundaries so the annual refits actually happen (fits 2013-06-28,
# 2014-01-02, 2015-01-02). Development data only.
SMOKE = {"dates": {"first_signal": "2013-06-28", "first_fill": "2013-07-01", "end": "2015-12-31"},
         "data": {"panel": "panel_stage1", "measures": "clean_measures/measures.parquet",
                  "extra_features_path": "garp_dev/garp_features.parquet"}}


def load_config(confirmed, smoke=False):
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    if smoke:
        # Same code path on development data only (signals and fills inside 2015 H2).
        config["dates"], config["data"] = dict(SMOKE["dates"]), dict(SMOKE["data"])
        return config, "smoke-test"
    if config.get("status") != "FROZEN" or not confirmed:
        raise SystemExit("Refusing to run: frozen_config.json must have \"status\": \"FROZEN\" "
                         "and --i-confirm-frozen must be passed. Nothing from April 2016 on was read.")
    digest = hashlib.sha256(CONFIG.read_bytes()).hexdigest()
    return config, digest


def check_config_matches_code(config):
    """The frozen thresholds must be the ones the shared code applies."""
    flag = config["clean_flag"]
    assert np.isclose(flag["max_abs_log_move_252"], wf.MAX_LOG_MOVE), "clean-flag move threshold mismatch"
    assert flag["min_close"] == wf.MIN_CLOSE, "clean-flag price floor mismatch"
    assert config["features"] == wf.cd.FEATURES, "feature list mismatch"
    from garp_features import GARP_SET
    assert set(config.get("extra_features", [])) <= set(GARP_SET), "unknown extra feature"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--i-confirm-frozen", action="store_true")
    parser.add_argument("--smoke", action="store_true",
                        help="development-data smoke test of this script (2015 H2 only); allowed while DRAFT")
    args = parser.parse_args()
    config, digest = load_config(args.i_confirm_frozen, smoke=args.smoke)
    check_config_matches_code(config)
    global OUT
    if args.smoke:
        OUT = HERE / "outputs" / "final_smoke"
    timer, start = {}, time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)

    first_signal = pd.Timestamp(config["dates"]["first_signal"])     # 2016-03-31
    end = pd.Timestamp(config["dates"]["end"])                       # 2026-09-30
    s, bt = config["strategy"], config["backtest"]

    # 1. Market calendar and benchmark for the test window.
    with duckdb.connect(str(DATABASE), read_only=True) as con:
        bench = con.execute("SELECT date, adj_close FROM benchmark WHERE date BETWEEN ? AND ? ORDER BY date",
                            [first_signal.date(), end.date()]).fetchdf()
    bench = bench.assign(date=pd.to_datetime(bench.date)).set_index("date").adj_close
    sessions = pd.DatetimeIndex(bench.index)
    assert sessions[0] == first_signal and sessions[1] == pd.Timestamp(config["dates"]["first_fill"])
    assert sessions[-1] <= end
    last_signal = sessions[-2]           # no order is planned on the last session

    # 2. Model rows (clean eligible, 2010 onward, needed columns only).
    t = time.perf_counter()
    extra = list(config.get("extra_features", []))
    extra_path = HERE / "outputs" / config["data"]["extra_features_path"] if extra else None
    frame, n_broken = wf.load_model_rows(HERE / "outputs" / config["data"]["panel"],
                                         HERE / "outputs" / config["data"]["measures"],
                                         config["model"]["train_start"], last_signal,
                                         extra_path=extra_path, extra_columns=extra)
    timer["load model rows"] = time.perf_counter() - t
    print(f"Model rows: {len(frame):,} (clean flag excluded {n_broken:,}); peak mem {peak_gb():.1f} GB")

    # 3. Walk-forward fits and scores.
    t = time.perf_counter()
    fits = wf.fit_dates(sessions, first_signal, last_signal, config["model"]["retraining_rule"])
    assert fits[0] == first_signal
    scores, fit_log = wf.walk_forward_scores(frame, fits, last_signal, alpha=config["model"]["alpha"],
                                             train_start=config["model"]["train_start"],
                                             numeric_features=list(wf.cd.NUMERIC_FEATURES) + extra,
                                             include_sector=True)
    print("\nFit-date table (each fit uses only outcomes with label_end < fit date):")
    print(fit_log[["fit_date", "train_rows", "last_train_signal", "last_train_label_end",
                   "scored_from", "scored_to", "scored_rows"]].to_string(index=False))
    first_fit = fit_log.iloc[0]
    assert pd.Timestamp(first_fit.last_train_label_end) < first_signal
    assert scores.date.min() == first_signal and scores.date.max() <= last_signal
    scores.to_parquet(OUT / "scores.parquet", index=False)
    fit_log.to_csv(OUT / "fit_log.csv", index=False)
    timer["fit + score"] = time.perf_counter() - t
    del frame

    # 4. Quotes for every stock that trades in the window (holdings need marks even after leaving the universe).
    t = time.perf_counter()
    long = pd.read_parquet(HERE / "outputs" / config["data"]["panel"], columns=["date", "symbol", "adj_close"],
                           filters=[("date", ">=", first_signal), ("date", "<=", end)])
    prices = long.pivot(index="date", columns="symbol", values="adj_close").reindex(sessions)
    del long
    timer["load prices"] = time.perf_counter() - t
    print(f"Prices: {prices.shape[0]} sessions x {prices.shape[1]:,} symbols; peak mem {peak_gb():.1f} GB")

    # 5. Backtest.
    t = time.perf_counter()
    strategy = ScoreStrategy(wf.scores_by_date(scores), top_n=s["top_n"],
                             cohort_fraction=s["entry_every"] / s["sessions_per_hold"],
                             per_name_cap=s["per_name_cap"])
    try:
        result = Backtester(prices, strategy, initial_cash=bt["initial_cash"], fee_bps=bt["fee_bps"],
                            holding_days=bt["holding_days"], entry_every=s["entry_every"],
                            max_stale_sessions=bt["max_stale_sessions"],
                            jump_threshold=config["clean_flag"]["max_abs_log_move_252"],
                            long_gap_policy=bt["long_gap_policy"]).run()
    except StaleQuoteError as exc:
        raise SystemExit(f"STOPPED: {exc}\nNo result is reported; investigate this data gap first.")
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

    # 6. Benchmark and metrics.
    bench_nav = benchmark_nav(bench, h.index, sessions[1], bt["initial_cash"])
    metrics = pd.DataFrame({"strategy": performance(h.nav, annual_rf=bt["sharpe_rf"]),
                            "^SP500TR": performance(bench_nav, annual_rf=bt["sharpe_rf"])})
    metrics.loc["excess_final_value"] = [metrics.loc["final_value", "strategy"] - metrics.loc["final_value", "^SP500TR"], np.nan]
    # Sensitivity: lots frozen by a long quote gap (policy "hold") valued at zero instead of their last mark.
    frozen_value = float(result.long_gaps.value_at_end.sum()) if len(result.long_gaps) else 0.0
    metrics.loc["frozen_lots_value_at_end"] = [frozen_value, np.nan]
    metrics.loc["final_value_if_frozen_lots_worth_zero"] = [h.nav.iloc[-1] - frozen_value, np.nan]
    result.long_gaps.to_csv(OUT / "long_gaps.csv", index=False)

    # 7. Save everything.
    h.assign(benchmark_nav=bench_nav).to_csv(OUT / "history.csv")
    f.to_csv(OUT / "fills.csv", index=False)
    o.to_csv(OUT / "orders.csv", index=False)
    result.closed_lots.to_csv(OUT / "closed_lots.csv", index=False)
    result.open_lots.to_csv(OUT / "open_lots.csv", index=False)
    result.stale.to_csv(OUT / "stale_marks.csv", index=False)
    result.jumps.to_csv(OUT / "jump_flags.csv", index=False)
    metrics.to_csv(OUT / "metrics.csv")
    (OUT / "accounting_checks.json").write_text(json.dumps(
        {"config_sha256": digest, "checks": checks}, indent=2), encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(h.index, h.nav / bt["initial_cash"] - 1, label="Strategy", lw=1.4)
    ax.plot(h.index, bench_nav / bt["initial_cash"] - 1, label="^SP500TR", lw=1.4)
    ax.axhline(0, color="grey", lw=0.6)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_title(f"Cumulative return, $1,000,000 from the {first_signal.date()} close to {h.index[-1].date()}")
    ax.set_ylabel("Cumulative return")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "cumulative_returns.png", dpi=150)

    timer["total"] = time.perf_counter() - start
    print("\nAccounting and timing checks:")
    for name, ok in checks.items():
        print(f"  {'OK  ' if ok else 'FAIL'} {name}")
    print(f"\nOrders {len(o):,} (rejected, no entry quote: {(o.status != 'filled').sum():,}); "
          f"fills {len(f):,}; open lots at end {len(result.open_lots)}; "
          f"stale-mark rows {len(result.stale):,}; jump flags {len(result.jumps)}; "
          f"long quote gaps {len(result.long_gaps)}; fees ${f.fee.sum():,.0f}")
    print(f"\nPositions held under the stale-quote rule (no quote for > {bt['max_stale_sessions']} sessions): "
          f"{len(result.long_gaps)}")
    if len(result.long_gaps):
        lg = result.long_gaps.copy()
        lg["value_at_last_price"] = lg.units * lg.last_mark
        print(lg[["symbol", "lot_id", "last_quote_date", "flagged_on", "last_mark", "value_at_last_price",
                  "still_open_at_end", "value_at_end"]].to_string(index=False))
        print(f"  tickers: {', '.join(sorted(lg.symbol.unique()))}")
    print(f"  final NAV at last prices: {h.nav.iloc[-1]:,.2f}; with still-frozen positions at zero: "
          f"{h.nav.iloc[-1] - frozen_value:,.2f}")
    print("\nMetrics:")
    print(metrics.to_string())
    print("\nTime (s):", {k: round(v) for k, v in timer.items()}, f"peak mem {peak_gb():.1f} GB")
    print(f"Saved to {OUT}")


if __name__ == "__main__":
    main()
