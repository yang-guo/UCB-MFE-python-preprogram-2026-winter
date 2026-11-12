"""Development test of a signal cutoff: buy a top-30 name only if its predicted
score clears a threshold, otherwise keep that slot in cash. 2012-2015 only.

    ../../../.venv/bin/python run_cutoff.py

The score is the model's predicted 90-day return percentile (Ridge on ranks).
Cutoffs are set from the development baseline's own top-30 scores (25th, 50th,
75th percentiles), i.e. so that roughly 75% / 50% / 25% of slots get filled.
Also checks whether weak-signal dates came before weak markets: cash only
helps if it does. Writes results/cutoff_dev_grid.csv and cutoff_timing.csv.
"""

import numpy as np
import pandas as pd

from features import FEATURES
from pipeline import DEV_BACKTEST, OUTPUTS, RESULTS, compare, run_strategy, setup

DEV_END = pd.Timestamp("2015-12-31")
COLS = FEATURES + ["skew_180"]


def main():
    OUTPUTS.mkdir(exist_ok=True)
    data, feats, target, label_end, panel = setup()
    dev = panel.loc[panel.date.le(DEV_END)].copy()
    del panel

    # 1. Baseline (no cutoff): the distribution of top-30 predicted scores.
    _, base_strategy, _, _, _ = run_strategy(data, feats, dev, *DEV_BACKTEST, features=COLS)
    log = pd.DataFrame(base_strategy.log)
    pooled = np.concatenate(log.top_scores.to_numpy())
    cutoffs = {f"q{q}": float(np.quantile(pooled, q / 100)) for q in (25, 50, 75)}
    print("Top-30 predicted scores, development cohorts: min %.3f, median %.3f, max %.3f"
          % (pooled.min(), np.median(pooled), pooled.max()))
    print("Cutoffs:", {k: round(v, 4) for k, v in cutoffs.items()})

    # 2. Timing check: does a weak signal (few names above the median cutoff)
    #    come before weak 90-day market or top-30 returns?
    bench = data.benchmark
    rows = []
    for _, r in log.iterrows():
        t = data.calendar.get_loc(r.signal_date)
        exit_t = data.calendar.searchsorted(r.signal_date + pd.Timedelta(days=91))
        if exit_t >= len(data.calendar) or t + 1 >= len(data.calendar):
            continue
        rows.append({"signal_date": r.signal_date,
                     "share_above_median_cutoff": np.mean(np.array(r.top_scores) >= cutoffs["q50"]),
                     "score_30th": r.score_nth,
                     "sp500tr_next_90d": bench.iloc[exit_t] / bench.iloc[t + 1] - 1})
    timing = pd.DataFrame(rows)
    corr = timing[["share_above_median_cutoff", "score_30th"]].corrwith(timing.sp500tr_next_90d, method="spearman")
    print("\nSpearman correlation with the next-90-day S&P 500 TR return (cohort dates):\n", corr.round(3))
    timing.to_csv(RESULTS / "cutoff_timing.csv", index=False)

    # 3. Backtests: weighting x cutoff.
    out = []
    for weighting in ["equal", "rank", "inverse_vol"]:
        for label, c in [("none", None), *cutoffs.items()]:
            result, strategy, _, nav, bench_nav = run_strategy(
                data, feats, dev, *DEV_BACKTEST, features=COLS, weighting=weighting,
                redistribute=weighting != "equal", min_score=c)
            m = compare(nav, bench_nav).iloc[:, 0]
            diag = result.daily.iloc[1:]
            filled = pd.DataFrame(strategy.log).slots_filled.mean()
            out.append({"weighting": weighting, "cutoff": label, "min_score": c, **m.drop(["start", "end"]),
                        "avg_cash_fraction": (diag.cash / diag.nav).mean(), "avg_slots_filled": filled,
                        "fees": result.fills.fee.sum()})
            print(f"{weighting:12s} cutoff {label:5s}: CAGR {m.CAGR:.2%}  Sharpe {m.Sharpe:.2f}  "
                  f"maxDD {m.max_drawdown:.2%}  cash {(diag.cash / diag.nav).mean():.1%}  slots {filled:.0%}",
                  flush=True)
    grid = pd.DataFrame(out)
    grid.loc[len(grid)] = {"weighting": "S&P 500 TR", **compare(nav, bench_nav).iloc[:, 1].drop(["start", "end"])}
    grid.to_csv(RESULTS / "cutoff_dev_grid.csv", index=False)


if __name__ == "__main__":
    main()
