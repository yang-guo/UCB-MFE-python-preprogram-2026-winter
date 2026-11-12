"""Development test of an industry-rotation strategy (2012-2015 only).

    ../../../.venv/bin/python run_rotation.py

Each cohort ranks industries (snapshot labels, >= 10 tradeable stocks) by the
equal-weighted past-`lookback`-session return of their tradeable stocks and
buys 10 stocks of each of the top n industries, chosen either by the model's
combined score (v1 + skew_180) or by the stocks' own past return (no model).
Also measures directly whether the chosen industry beat the tradeable universe
over the next 90 days. Writes results/rotation_dev_grid.csv.
"""

import numpy as np
import pandas as pd

from features import FEATURES
from model import tradeable_mask
from pipeline import DEV_BACKTEST, OUTPUTS, RESULTS, compare, run_strategy, setup

DEV_END = pd.Timestamp("2015-12-31")
COLS = FEATURES + ["skew_180"]


def main():
    OUTPUTS.mkdir(exist_ok=True)
    data, feats, target, label_end, panel = setup()
    dev = panel.loc[panel.date.le(DEV_END)].copy()
    del panel
    industries = data.industry.reindex(data.symbols).fillna("Unknown").to_numpy()
    configs = [None] + [{"n_industries": n, "lookback": L, "per_industry": 10, "within": w, "min_members": 10}
                        for w in ["score", "momentum"] for L in [63, 126] for n in [1, 2, 3]]
    rows = []
    for rot in configs:
        result, strategy, _, nav, bench = run_strategy(data, feats, dev, *DEV_BACKTEST, features=COLS,
                                                       redistribute=True, rotation=rot)
        m = compare(nav, bench).iloc[:, 0]
        diag = result.daily.iloc[1:]
        r, rb = nav.pct_change().dropna().align(bench.pct_change().dropna(), join="inner")
        label = "top 30 by model (baseline)" if rot is None else \
            f"top {rot['n_industries']} industry x10 by {rot['within']}, lookback {rot['lookback']}"
        row = {"strategy": label, **m.drop(["start", "end"]), "beta_to_sp500": np.cov(r, rb)[0, 1] / rb.var(),
               "avg_cash_fraction": (diag.cash / diag.nav).mean(), "avg_names_held": diag.names.mean()}
        if rot is not None:
            # Did the top-ranked industry beat the universe over the next 90 days? (dev labels only)
            log = pd.DataFrame(strategy.log)
            ex = []
            for _, c in log.iterrows():
                t = data.calendar.get_loc(c.signal_date)
                trade = tradeable_mask(data, feats, t)
                top_ind = c.industries.split("|")[0]
                in_ind = trade & (industries == top_ind)
                ex.append(np.nanmean(target[t, in_ind]) - np.nanmean(target[t, trade]))
            ex = pd.Series(ex)
            row.update({"top_industry_excess_90d": ex.mean(), "top_industry_beat_universe": (ex > 0).mean(),
                        "distinct_industries": log.industries.str.split("|").explode().nunique()})
        rows.append(row)
        print(f"{label:48s}: CAGR {m.CAGR:.2%}  Sharpe {m.Sharpe:.2f}  maxDD {m.max_drawdown:.2%}  "
              f"vol {m.annual_volatility:.1%}  beta {row['beta_to_sp500']:.2f}"
              + (f"  top-industry excess {row['top_industry_excess_90d']:+.2%} (beat {row['top_industry_beat_universe']:.0%})"
                 if rot else ""), flush=True)
    grid = pd.DataFrame(rows)
    grid.loc[len(grid)] = {"strategy": "S&P 500 TR", **compare(nav, bench).iloc[:, 1].drop(["start", "end"])}
    grid.to_csv(RESULTS / "rotation_dev_grid.csv", index=False)


if __name__ == "__main__":
    main()
