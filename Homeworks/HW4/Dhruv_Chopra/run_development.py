"""Development work: Jan 2010 - Dec 2015 only. Nothing here reads a label or
price after 2015-12-31 for evaluation (labels of late-2015 signals end in the
2016 buffer, which is what the buffer is for).

    ../../../.venv/bin/python run_development.py

Writes results/dev_*.csv and prints the tables quoted in the README.
"""

import numpy as np
import pandas as pd

from features import FEATURES
from model import daily_ic, walk_forward
from pipeline import DEV_BACKTEST, OUTPUTS, RESULTS, compare, run_strategy, setup

VALIDATION_YEARS = [2012, 2013, 2014, 2015]
DEV_END = pd.Timestamp("2015-12-31")


def top_n_excess(wf, n=30):
    """Mean 90-day return of the top-n tradeable names minus the tradeable average."""
    s = wf[wf.tradeable].copy()
    s["r"] = s.groupby("date").score.rank(ascending=False, method="first")
    top = s[s.r <= n].groupby("date").target.mean()
    return top - s.groupby("date").target.mean()


def main():
    OUTPUTS.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    data, feats, target, label_end, panel = setup()
    dev = panel.loc[panel.date.le(DEV_END)].copy()
    print(f"Development panel: {len(dev):,} stock-weeks, {dev.date.nunique()} weekly cross-sections")

    # ---- 1. Walk-forward model comparison (expanding window, purged) ----
    rows = []
    for kind in ["ridge", "hgb"]:
        wf = walk_forward(kind, dev, data, VALIDATION_YEARS)
        ic = daily_ic(wf[wf.tradeable])
        ex = top_n_excess(wf)
        rows.append({"score": kind, "rank_IC": ic.mean(), "IC_t_stat": ic.mean() / ic.std() * np.sqrt(len(ic) / 13),
                     "top30_excess_90d": ex.mean(),
                     **{f"top30_excess_{y}": ex[ex.index.year == y].mean() for y in VALIDATION_YEARS}})
        if kind == "ridge":
            ridge_wf = wf
    base = dev[dev.date.dt.year.isin(VALIDATION_YEARS) & dev.target.notna()].copy()
    for f in ["mom_252_21", "drawdown_252", "gross_profitability", "vol_63"]:
        base["score"] = base[f].fillna(0.5) * (-1 if f == "vol_63" else 1)
        ic = daily_ic(base[base.tradeable])
        ex = top_n_excess(base)
        rows.append({"score": f"single feature: {'-' if f == 'vol_63' else ''}{f}", "rank_IC": ic.mean(),
                     "IC_t_stat": ic.mean() / ic.std() * np.sqrt(len(ic) / 13), "top30_excess_90d": ex.mean(),
                     **{f"top30_excess_{y}": ex[ex.index.year == y].mean() for y in VALIDATION_YEARS}})
    table = pd.DataFrame(rows).set_index("score")
    # IC t-stat divides by sqrt(13): weekly samples overlap ~13 weeks of a 90-day label.
    print("\nWalk-forward validation 2012-2015 (tradeable stocks):\n", table.round(4).to_string())
    table.to_csv(RESULTS / "dev_model_comparison.csv")

    # Ridge coefficients from the last development fit, for interpretation.
    from model import fit
    last = fit("ridge", dev, int(dev.loc[dev.date.ge("2015-01-01"), "t"].min()))
    coef = pd.Series(last.model.coef_[:len(FEATURES)], index=FEATURES).sort_values()
    print("\nRidge coefficients (per unit of within-date percentile rank):\n", coef.round(4).to_string())
    coef.rename("coefficient").to_csv(RESULTS / "dev_ridge_coefficients.csv")

    # ---- 2. Development backtests: choose the portfolio size ----
    rows = []
    for model, top_n in [("ridge", 20), ("ridge", 30), ("ridge", 50), ("momentum", 30)]:
        result, strategy, schedule, nav, bench = run_strategy(data, feats, dev, *DEV_BACKTEST,
                                                             model=model, top_n=top_n)
        m = compare(nav, bench, f"{model} top{top_n}")
        diag = result.daily.iloc[1:]
        rows.append(m.iloc[:, 0].rename(f"{model} top{top_n}")
                    .to_frame().T.assign(avg_cash_fraction=(diag.cash / diag.nav).mean(),
                                         fees=result.fills.fee.sum() if len(result.fills) else 0.0))
        bench_row = m.iloc[:, 1]
        c = m.iloc[:, 0]
        print(f"{model} top{top_n}: CAGR {c.CAGR:.2%}, Sharpe {c.Sharpe:.2f}, max drawdown {c.max_drawdown:.2%}")
    rows.append(bench_row.rename("S&P 500 TR").to_frame().T)
    bt = pd.DataFrame(pd.concat(rows))
    print("\nDevelopment backtests", DEV_BACKTEST, ":\n", bt.to_string())
    bt.to_csv(RESULTS / "dev_backtests.csv")


if __name__ == "__main__":
    main()
