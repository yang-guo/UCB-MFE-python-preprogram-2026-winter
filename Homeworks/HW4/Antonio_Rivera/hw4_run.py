"""Run one strategy variant end to end and save its report to outputs/run_<tag>/.

Usage (from the repo root):
    uv run python Homeworks/HW4/Antonio_Rivera/hw4_run.py <cache> <scores.parquet> <variant> <start> <end>
e.g. dev:   hw4_run.py dev outputs/scores_dev_ridge_F1_baseline.parquet S1 2012-01-03 2015-12-31
     final: hw4_run.py final outputs/scores_final_ridge_F1_baseline.parquet S1 2016-04-01 2026-09-30
"""

import sys
from pathlib import Path

import pandas as pd

import hw4_backtest as bt
import hw4_build_cache as hc
import hw4_data as hd

VARIANTS = {
    # S0: the original monthly top-20 cohorts (kept as the cash-drag baseline).
    "S0": dict(top_n=20, cohort_fraction=1 / 3, max_weight=0.05, frequency="M"),
    # S1: broad weekly cohorts; 13 overlapping cohorts of NAV/12 -> ~fully invested.
    "S1": dict(top_n=100, cohort_fraction=1 / 12, max_weight=0.05, frequency="W"),
    # S1R: S1 with linear rank weights inside each cohort (best name n shares, n-th 1 share).
    "S1R": dict(top_n=100, cohort_fraction=1 / 12, max_weight=0.05, frequency="W", weighting="rank"),
}
PANEL_COLS = ["date", "symbol", "adj_close", "volume", "eligible", "mcap_suspect"]
WARMUP_DAYS = 60    # calendar days of history kept before `start` (liquidity window)


def load_run_panel(cache, scores_path, start, end):
    panel = pd.read_parquet(hc.cache_path(cache), columns=PANEL_COLS)
    panel = panel.loc[panel.date.le(end)]
    database = hc.TARGETS[cache][0]
    calendar = bt.load_benchmark(database, end).index
    panel = bt.add_traded_days(panel, calendar)
    panel = panel.loc[panel.date.ge(pd.Timestamp(start) - pd.Timedelta(days=WARMUP_DAYS))]
    scores = pd.read_parquet(scores_path, columns=["date", "symbol", "score"])
    panel = panel.merge(scores, on=["date", "symbol"], how="left", validate="one_to_one")
    return panel.reset_index(drop=True), calendar, database


def run(cache, scores_path, variant, start, end, fee_bps=bt.DEFAULT_FEE_BPS):
    panel, calendar, database = load_run_panel(cache, scores_path, start, end)
    strat = bt.TopNScoreStrategy(**VARIANTS[variant])
    result = bt.BacktestEngine(panel, calendar, strat, bt.OrderManager(1_000_000, fee_bps=fee_bps),
                               start, end).run()
    report = bt.PerformanceReport(result, bt.load_benchmark(database, end))
    return result, report


def concentration_checks(result):
    """Guards for bad-print PnL and stale end marks (see data audit)."""
    t = result.trades
    total = t.pnl.sum()
    top5 = t.nlargest(5, "pnl")
    ol = result.open_lots
    stale_end = ol.loc[ol.mark_date < result.daily.index[-1]] if len(ol) else ol
    return {"total_realized_pnl": total,
            "top5_lots_pnl": top5.pnl.sum(),
            "top5_share_of_realized": top5.pnl.sum() / total if total else float("nan"),
            "max_single_lot_return": t["return"].max() if len(t) else float("nan"),
            "open_lots_stale_at_end": len(stale_end),
            "stale_end_value": stale_end.end_value.sum() if len(stale_end) else 0.0}


if __name__ == "__main__":
    cache, scores_path, variant, start, end = sys.argv[1:6]
    fee = float(sys.argv[6]) if len(sys.argv) > 6 else bt.DEFAULT_FEE_BPS
    tag = f"{cache}_{variant}_{int(fee)}bps"
    out = hc.OUT / f"run_{tag}"
    out.mkdir(parents=True, exist_ok=True)
    result, report = run(cache, scores_path, variant, start, end, fee)
    summary = report.summary()
    trading = pd.Series(report.trading())
    checks = pd.Series(concentration_checks(result))
    verdict = pd.Series(report.verdict())
    summary.to_csv(out / "metrics.csv")
    pd.concat([trading, checks, verdict]).to_csv(out / "trading.csv", header=["value"])
    result.daily.to_parquet(out / "daily.parquet")
    result.trades.to_csv(out / "trades.csv", index=False)
    result.diagnostics.to_csv(out / "diagnostics.csv", index=False)
    report.plot(out / "chart.png", title=f"Cumulative return: {variant} vs ^SP500TR ({start} to {end})")
    pd.set_option("display.width", 160)
    print(summary.to_string())
    print(pd.concat([verdict, trading, checks]).to_string())
    print("saved to", out)
