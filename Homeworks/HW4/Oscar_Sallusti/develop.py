"""Development-period research: model validation and strategy selection.

Uses data through 2016-03-31 only (development + buffer). Nothing from the
final test period is loaded.

    .venv/bin/python Homeworks/HW4/Oscar_Sallusti/develop.py [path/to/database]

Validation design (expanding walk-forward, yearly refits):
    fit on first signal of 2012 -> predict 2012
    fit on first signal of 2013 -> predict 2013   ... through 2015
Each fit uses rows from 2010 onwards whose 90-day exit is before the fit date.
Writes dev_prediction_report.csv and dev_strategy_grid.csv to results/.
"""

from __future__ import annotations

import itertools
import sys
import time
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from backtester import Backtester, benchmark_nav, check_accounting, performance_metrics  # noqa: E402
from features import build_panel, load_fundamental_events, load_market, signal_dates  # noqa: E402
from model import walk_forward, prediction_report  # noqa: E402
from strategy import TopScoreStrategy  # noqa: E402

DEFAULT_DB = HERE.parents[2] / "Lectures/modeling/data/equities_top2000.duckdb"
OUT = HERE / "results"  # small tables, committed
DEV_DATA_END = "2016-03-31"
DEV_BT_START, DEV_BT_END = "2012-01-01", "2015-12-31"
VALID_YEARS = [2012, 2013, 2014, 2015]


def main(db=DEFAULT_DB):
    OUT.mkdir(exist_ok=True)
    t = time.time()
    m = load_market(db, end=DEV_DATA_END)
    events = load_fundamental_events(db, end=DEV_DATA_END)
    panel = build_panel(m, events, "2010-01-01", DEV_DATA_END)
    print(f"dev panel {panel.shape} in {time.time() - t:.0f}s")

    sig = signal_dates(m["calendar"], "2012-01-01", DEV_BT_END)
    fit_dates = [sig[sig.year == y][0] for y in VALID_YEARS]

    kinds = ["size", "momentum", "ridge", "hgb", "hgb_small"]
    preds, reports = {}, []
    for kind in kinds:
        t = time.time()
        pred, log = walk_forward(panel, fit_dates, DEV_BT_END, kind)
        preds[kind] = pred
        print(f"\n{kind} ({time.time() - t:.0f}s), train rows per fit: {log.train_rows.tolist()}")
        for scope in [None, 500]:
            rep = prediction_report(pred, max_rank=scope).assign(model=kind, scope=scope or "all")
            reports.append(rep)
            print(f"  universe={scope or 'all'}:\n" + rep[["rank_IC", "top_minus_all",
                                                         "decile_spread"]].round(4).T.to_string())
    pd.concat(reports).to_csv(OUT / "dev_prediction_report.csv")

    prices = m["A"].loc[:DEV_BT_END]
    bench = benchmark_nav(m["bench"], DEV_BT_START, DEV_BT_END)
    rows = [performance_metrics(bench).rename("SP500TR")]
    yearly = {"SP500TR": bench.resample("YE").last().pct_change().fillna(bench.resample("YE").last().iloc[0] / bench.iloc[0] - 1)}
    universes = {"top500": dict(max_universe_rank=500), "top1000": dict(max_universe_rank=1000),
                 "liquid": dict(min_dollar_vol=10e6)}
    for kind, top_n, uni in itertools.product(kinds, [20, 50], universes):
        strat = TopScoreStrategy(preds[kind], top_n=top_n, **universes[uni])
        res = Backtester(prices).run(strat, DEV_BT_START, DEV_BT_END)
        chk = check_accounting(res, prices)
        assert chk.max_cash_error < 1e-4 and chk.bad_buy_timing == 0 and chk.min_cash >= 0
        name = f"{kind}|top{top_n}|{uni}"
        met = performance_metrics(res.history.nav)
        met["max_stale_value"] = res.history.stale_value.max()
        met["cancelled"] = len(res.cancelled)
        rows.append(met.rename(name))
        ye = res.history.nav.resample("YE").last()
        yearly[name] = ye.pct_change().fillna(ye.iloc[0] / res.history.nav.iloc[0] - 1)
    grid = pd.DataFrame(rows)
    yr = pd.DataFrame(yearly).T
    yr.columns = yr.columns.year
    grid = grid.join(yr.sub(yr.loc["SP500TR"]).add_prefix("excess_"))
    grid.to_csv(OUT / "dev_strategy_grid.csv")
    pd.set_option("display.width", 250)
    print("\nDevelopment backtests 2012-2015 (excess_YYYY = calendar-year return minus SP500TR):")
    print(grid.drop(columns=["final_value"]).round(3).sort_values("sharpe", ascending=False).to_string())


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DB)
