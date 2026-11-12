"""Development test of correlation-limited selection (2012-2015 only).

    ../../../.venv/bin/python run_correlation.py

Walk down the model's ranking and add a stock only if the correlation of its
past-year daily returns with the current portfolio (held names + names already
accepted for this cohort, equal-weighted) is below k. Search depth is capped at
the top 300 names; unfilled slots stay in cash. v1 + skew_180, top 30.
Writes results/correlation_dev_grid.csv.
"""

import numpy as np
import pandas as pd

from features import FEATURES
from pipeline import DEV_BACKTEST, OUTPUTS, RESULTS, compare, run_strategy, setup

DEV_END = pd.Timestamp("2015-12-31")
COLS = FEATURES + ["skew_180"]
AGREEMENT = 0.7324


def main():
    OUTPUTS.mkdir(exist_ok=True)
    data, feats, target, label_end, panel = setup()
    dev = panel.loc[panel.date.le(DEV_END)].copy()
    del panel
    configs = [(w, k, None) for w in ["equal", "rank"] for k in [None, 0.8, 0.7, 0.6, 0.5, 0.4]]
    configs += [("rank", k, AGREEMENT) for k in [None, 0.7, 0.6, 0.5]]
    rows = []
    for weighting, k, cutoff in configs:
        result, strategy, _, nav, bench = run_strategy(
            data, feats, dev, *DEV_BACKTEST, features=COLS, weighting=weighting,
            redistribute=True, min_agreement=cutoff, corr_cap=k)
        m = compare(nav, bench).iloc[:, 0]
        diag = result.daily.iloc[1:]
        r, rb = nav.pct_change().dropna(), bench.pct_change().dropna()
        r, rb = r.align(rb, join="inner")
        log = pd.DataFrame(strategy.log)
        rows.append({"weighting": weighting, "corr_cap": k, "agreement_cutoff": cutoff is not None,
                     **m.drop(["start", "end"]), "beta_to_sp500": np.cov(r, rb)[0, 1] / rb.var(),
                     "corr_to_sp500": r.corr(rb), "avg_cash_fraction": (diag.cash / diag.nav).mean(),
                     "avg_names_held": diag.names.mean(),
                     "mean_corr_of_picks": log.get("mean_corr_of_picks", pd.Series(dtype=float)).mean(),
                     "mean_search_depth": log.get("search_depth", pd.Series(dtype=float)).mean()})
        x = rows[-1]
        print(f"{weighting:5s} k={str(k):4s} cutoff={'yes' if cutoff else 'no ':3s}: CAGR {m.CAGR:.2%}  "
              f"Sharpe {m.Sharpe:.2f}  maxDD {m.max_drawdown:.2%}  vol {m.annual_volatility:.1%}  "
              f"beta {x['beta_to_sp500']:.2f}  cash {x['avg_cash_fraction']:.1%}  "
              f"pick corr {x['mean_corr_of_picks']:.2f}  depth {x['mean_search_depth']:.0f}", flush=True)
    grid = pd.DataFrame(rows)
    grid.loc[len(grid)] = {"weighting": "S&P 500 TR", **compare(nav, bench).iloc[:, 1].drop(["start", "end"])}
    grid.to_csv(RESULTS / "correlation_dev_grid.csv", index=False)


if __name__ == "__main__":
    main()
