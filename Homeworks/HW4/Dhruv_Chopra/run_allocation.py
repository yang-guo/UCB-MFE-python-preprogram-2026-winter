"""Development test of portfolio construction (2012-2015 backtests only).

    ../../../.venv/bin/python run_allocation.py

The spec leaves sizing to us ("You decide how many stocks to buy, how much to
allocate"). Grid, run on the development period only:
  * weighting: equal / rank (best name ~2x average) / inverse 63-day volatility
  * stocks per cohort: 30 / 50 / 100
  * features: v1, v1 + skew_180, v1 + skew_180 + buybacks + net issuance
Writes results/allocation_dev_grid.csv.

    ../../../.venv/bin/python run_allocation.py --redistribute

re-runs the main cells with the name-cap redistribution (money a capped name
cannot take goes to the other picks, then to the next-ranked names) and writes
results/allocation_redistribute_dev.csv.
"""

import argparse

import pandas as pd

from features import FEATURES
from pipeline import DEV_BACKTEST, OUTPUTS, RESULTS, compare, run_strategy, setup

DEV_END = pd.Timestamp("2015-12-31")
FEATURE_SETS = {
    "v1": FEATURES,
    "v1 + skew_180": FEATURES + ["skew_180"],
    "v1 + skew_180 + issuance": FEATURES + ["skew_180", "buyback_to_assets", "net_issuance_yoy"],
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--redistribute", action="store_true")
    args = parser.parse_args()
    feature_sets = ({k: FEATURE_SETS[k] for k in ["v1", "v1 + skew_180"]} if args.redistribute
                    else FEATURE_SETS)
    sizes = [30, 50] if args.redistribute else [30, 50, 100]
    out_name = "allocation_redistribute_dev.csv" if args.redistribute else "allocation_dev_grid.csv"
    OUTPUTS.mkdir(exist_ok=True)
    data, feats, target, label_end, panel = setup()
    dev = panel.loc[panel.date.le(DEV_END)].copy()
    del panel
    rows = []
    for fs, cols in feature_sets.items():
        for weighting in ["equal", "rank", "inverse_vol"]:
            for top_n in sizes:
                result, _, _, nav, bench = run_strategy(data, feats, dev, *DEV_BACKTEST, features=cols,
                                                        top_n=top_n, weighting=weighting,
                                                        redistribute=args.redistribute)
                m = compare(nav, bench).iloc[:, 0]
                diag = result.daily.iloc[1:]
                rows.append({"features": fs, "weighting": weighting, "top_n": top_n, "redistribute": args.redistribute, **m.drop(["start", "end"]),
                             "avg_cash_fraction": (diag.cash / diag.nav).mean(), "fees": result.fills.fee.sum()})
                print(f"{fs:26s} {weighting:12s} top{top_n:<4d} CAGR {m.CAGR:.2%}  Sharpe {m.Sharpe:.2f}  "
                      f"maxDD {m.max_drawdown:.2%}  cash {(diag.cash / diag.nav).mean():.1%}", flush=True)
    bench_m = compare(nav, bench).iloc[:, 1]
    grid = pd.DataFrame(rows)
    grid.loc[len(grid)] = {"features": "S&P 500 TR", **bench_m.drop(["start", "end"])}
    grid.to_csv(RESULTS / out_name, index=False)
    print(grid.round(4).to_string())


if __name__ == "__main__":
    main()
