"""Step 3: choose the trading rule in development (2013-01 to 2015-12) only.

    uv run python run_strategy_dev.py --db top2000 [--model NAME]   # ~2 min

Uses the walk-forward validation predictions (each year scored by a model
fitted before that year) and my backtester. Grid (kept small on purpose):
top N in {10, 25, 50} x cadence {weekly 1/13, monthly 1/3} x universe
{all eligible, 1000 most liquid}. Writes results/strategy_dev_grid.csv.
"""

import argparse
import json

import duckdb
import numpy as np
import pandas as pd

from hw4 import analytics as A, config
from hw4.backtester import Backtester, TopNCohortStrategy

GRID = [{"top_n": n, "cadence": c, "universe": u}
        for n in (10, 25, 50) for c in ("weekly", "monthly") for u in ("all", "liquid")]


def load_prices_wide(cache, start, end):
    px = cache.execute("SELECT date, symbol, adj_close FROM px_daily WHERE date BETWEEN ? AND ?",
                       [start, end]).fetchdf()
    wide = px.pivot(index="date", columns="symbol", values="adj_close").astype("float64")
    cal = pd.DatetimeIndex(cache.execute("SELECT date FROM bench WHERE date BETWEEN ? AND ? ORDER BY date",
                                         [start, end]).fetchdf().date)
    assert wide.index.isin(cal).all()
    return wide.reindex(cal)                    # sessions with no quote stay NaN (no filling)


def schedule_for(signal_dates, cadence):
    s = pd.DatetimeIndex(sorted(signal_dates))
    if cadence == "weekly":
        return s
    return pd.DatetimeIndex(pd.Series(s, index=s).groupby(s.to_period("M")).max().values)


def run_grid(scores, prices, bench, start, end, fee_bps=config.FEE_BPS, grid=GRID):
    rows, runs = [], {}
    weekly = schedule_for(scores.date.unique(), "weekly")
    first = weekly[weekly >= start][0]
    for g in grid:
        # Both cadences start at the same first signal (same benchmark window);
        # monthly then trades on the last weekly signal of each later month.
        sched = schedule_for(scores.date.unique(), g["cadence"])
        sched = pd.DatetimeIndex([first]).append(sched[(sched > first) & (sched < end)])
        strat = TopNCohortStrategy(scores, sched, top_n=g["top_n"],
                                   cohort_fraction=1 / 13 if g["cadence"] == "weekly" else 1 / 3,
                                   universe=g["universe"])
        res = Backtester(prices, strat, None, fee_bps=fee_bps).run(sched[0], end)
        first_entry = prices.index[prices.index.get_loc(sched[0]) + 1]
        bnav = A.benchmark_nav(bench, sched[0], first_entry, end)
        perf = A.performance(res.history.nav, bnav, history=res.history)
        bperf = A.performance(bnav)
        rows.append({**g, **{k: perf[k] for k in ["final_value", "CAGR", "volatility", "sharpe", "max_drawdown",
                                                   "beta", "tracking_error", "information_ratio",
                                                   "avg_cash_fraction", "annual_turnover", "total_fees", "avg_names"]},
                     "excess_CAGR": perf["CAGR"] - bperf["CAGR"], "bench_CAGR": bperf["CAGR"],
                     "bench_sharpe": bperf["sharpe"]})
        runs[(g["top_n"], g["cadence"], g["universe"])] = (res, bnav)
        print(f"N={g['top_n']:>2} {g['cadence']:<7} {g['universe']:<6} CAGR={perf['CAGR']:+.3f} "
              f"Sharpe={perf['sharpe']:.2f} excess={perf['CAGR'] - bperf['CAGR']:+.3f} "
              f"cash={perf['avg_cash_fraction']:.3f}")
    return pd.DataFrame(rows), runs


def main(db_key, model_name=None):
    cache = duckdb.connect(str(config.OUTPUT_DIR / f"cache_{db_key}.duckdb"), read_only=True)
    preds = duckdb.connect(str(config.OUTPUT_DIR / f"preds_{db_key}.duckdb"), read_only=True)
    sel = json.loads((config.OUTPUT_DIR / f"cv_selection_{db_key}.json").read_text())
    model_name = model_name or sel["selected"]
    pv = preds.execute("SELECT date, symbol, score FROM pred_val WHERE model=?", [model_name]).fetchdf()
    meta = cache.execute("SELECT date, symbol, sector, liq_rank FROM panel WHERE date BETWEEN ? AND ?",
                         ["2013-01-01", config.DEV_END]).fetchdf()
    scores = pv.merge(meta, on=["date", "symbol"], validate="one_to_one")
    assert scores.date.max() <= pd.Timestamp(config.DEV_END)
    prices = load_prices_wide(cache, "2012-12-01", config.DEV_END)
    bench = cache.execute("SELECT date, adj_close FROM bench").fetchdf().set_index("date").adj_close
    start, end = pd.Timestamp("2013-01-01"), pd.Timestamp(config.DEV_END)
    print(f"strategy grid on validation predictions of {model_name}")
    grid, runs = run_grid(scores, prices, bench, start, end)
    grid.insert(0, "model", model_name)
    print(grid.round(3).to_string(index=False))
    grid.to_csv(config.OUTPUT_DIR / f"strategy_dev_{db_key}_{model_name}.csv", index=False)
    if db_key == "top2000":
        config.RESULTS_DIR.mkdir(exist_ok=True)
        grid.round(4).to_csv(config.RESULTS_DIR / f"strategy_dev_grid_{model_name}.csv", index=False)
    return grid


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", choices=list(config.DATABASES), default="class")
    ap.add_argument("--model", default=None)
    a = ap.parse_args()
    main(a.db, a.model)
