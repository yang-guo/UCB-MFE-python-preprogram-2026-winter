"""Development test of sector-neutral selection (2012-2015 only).

    ../../../.venv/bin/python run_sector.py

Each cohort buys the k best-scoring stocks in *every* one of the 11 sectors
(k = 1 is "only the highest signal within each sector"), instead of the
overall top 30. Grid: k in {1, 2, 3, 5} x equal / rank weights x with / without
the feature-agreement cutoff, on v1 + skew_180, plus the top-30 baselines.
Writes results/sector_dev_grid.csv.
"""

import pandas as pd

from features import FEATURES
from pipeline import DEV_BACKTEST, OUTPUTS, RESULTS, compare, run_strategy, setup

DEV_END = pd.Timestamp("2015-12-31")
COLS = FEATURES + ["skew_180"]
AGREEMENT = 0.7324        # 25th percentile of 2012-2015 picks (run_agreement.py)


def main():
    OUTPUTS.mkdir(exist_ok=True)
    data, feats, target, label_end, panel = setup()
    dev = panel.loc[panel.date.le(DEV_END)].copy()
    del panel
    configs = [("top 30 (baseline)", None, w, c) for w in ["equal", "rank"] for c in [None, AGREEMENT]]
    configs += [(f"best {k} per sector", k, w, c) for k in [1, 2, 3, 5] for w in ["equal", "rank"]
                for c in [None, AGREEMENT]]
    rows = []
    for label, k, weighting, cutoff in configs:
        result, strategy, _, nav, bench = run_strategy(
            data, feats, dev, *DEV_BACKTEST, features=COLS, weighting=weighting,
            redistribute=weighting != "equal" or cutoff is not None, min_agreement=cutoff, per_sector=k)
        m = compare(nav, bench).iloc[:, 0]
        diag = result.daily.iloc[1:]
        rows.append({"selection": label, "weighting": weighting, "agreement_cutoff": cutoff is not None,
                     **m.drop(["start", "end"]), "avg_cash_fraction": (diag.cash / diag.nav).mean(),
                     "avg_names_held": diag.names.mean(), "fees": result.fills.fee.sum()})
        print(f"{label:20s} {weighting:6s} cutoff={'yes' if cutoff else 'no ':3s}: CAGR {m.CAGR:.2%}  "
              f"Sharpe {m.Sharpe:.2f}  maxDD {m.max_drawdown:.2%}  vol {m.annual_volatility:.1%}  "
              f"cash {(diag.cash / diag.nav).mean():.1%}  names {diag.names.mean():.0f}", flush=True)
    grid = pd.DataFrame(rows)
    grid.loc[len(grid)] = {"selection": "S&P 500 TR", **compare(nav, bench).iloc[:, 1].drop(["start", "end"])}
    grid.to_csv(RESULTS / "sector_dev_grid.csv", index=False)


if __name__ == "__main__":
    main()
