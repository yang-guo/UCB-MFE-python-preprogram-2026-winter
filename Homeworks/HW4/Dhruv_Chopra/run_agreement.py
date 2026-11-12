"""Development test of a feature-agreement cutoff (2012-2015 only).

    ../../../.venv/bin/python run_agreement.py

agreement = positive feature contributions / absolute contributions for each
top-30 pick (ModelSchedule.agreement). Three checks before trusting it:
  1. drift: does agreement trend over time / jump at the quarterly refits?
     (the raw-score cutoff failed exactly here);
  2. stock picking: within the top 30, do high-agreement picks earn more over
     90 days than low-agreement picks?
  3. backtests: picks below the cutoff are not bought; the slot stays in cash.
Writes results/agreement_*.csv.
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

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

    # Baseline run (no cutoff) logs every top-30 pick with its agreement.
    _, base, _, _, _ = run_strategy(data, feats, dev, *DEV_BACKTEST, features=COLS)
    picks = []
    for r in base.log:
        t = data.calendar.get_loc(r["signal_date"])
        tradeable = tradeable_mask(data, feats, t)
        mean_ret = np.nanmean(target[t, tradeable])          # 90-day return of the tradeable universe
        for n, a, s in zip(r["top_picked"], r["top_agreement"], r["top_scores"]):
            picks.append({"signal_date": r["signal_date"], "n": n, "agreement": a, "score": s,
                          "excess_90d": target[t, n] - mean_ret})
    picks = pd.DataFrame(picks)
    picks["quarter"] = picks.signal_date.dt.to_period("Q")
    picks.to_csv(RESULTS / "agreement_picks.csv", index=False)

    # 1. Drift.
    by_q = picks.groupby("quarter").agreement.mean()
    within = (picks.agreement - picks.groupby("quarter").agreement.transform("mean")).var()
    print("Agreement of top-30 picks: mean %.3f, 10th-90th pct %.3f-%.3f"
          % (picks.agreement.mean(), picks.agreement.quantile(0.1), picks.agreement.quantile(0.9)))
    print("Share of variance between refit quarters: %.0f%%" % (100 * (1 - within / picks.agreement.var())))
    print("Quarterly mean agreement:\n", by_q.round(3).to_string())
    rho, p = spearmanr(np.arange(len(by_q)), by_q.to_numpy())
    print("Trend (Spearman vs time): %.2f (p=%.2f)" % (rho, p))

    # 2. Stock picking within the top 30: agreement terciles per date.
    d = picks.dropna(subset=["excess_90d", "agreement"]).copy()
    d["tercile"] = d.groupby("signal_date").agreement.transform(
        lambda a: pd.qcut(a.rank(method="first"), 3, labels=["low", "mid", "high"]))
    terc = d.groupby(["signal_date", "tercile"], observed=True).excess_90d.mean().unstack()
    spread = terc["high"] - terc["low"]
    ic = d.groupby("signal_date").apply(lambda g: g.agreement.corr(g.excess_90d, method="spearman"),
                                        include_groups=False)
    print("\nWithin top 30, mean 90-day excess by agreement tercile:\n", terc.mean().round(4).to_string())
    print("High-minus-low: %.4f, positive on %.0f%% of %d cohort dates, t (overlap-adjusted) %.2f"
          % (spread.mean(), 100 * (spread > 0).mean(), len(spread),
             spread.mean() / spread.std() * np.sqrt(len(spread) / 3)))
    print("Within-top-30 IC of agreement: %.3f" % ic.mean())
    pd.DataFrame({"high_minus_low": spread, "ic": ic}).to_csv(RESULTS / "agreement_stock_picking.csv")

    # 3. Backtests with agreement cutoffs from the baseline's distribution.
    cutoffs = {f"q{q}": float(np.nanquantile(picks.agreement, q / 100)) for q in (25, 50, 75)}
    print("\nCutoffs:", {k: round(v, 3) for k, v in cutoffs.items()})
    out = []
    for weighting in ["equal", "rank", "inverse_vol"]:
        for label, c in [("none", None), *cutoffs.items()]:
            result, strategy, _, nav, bnav = run_strategy(
                data, feats, dev, *DEV_BACKTEST, features=COLS, weighting=weighting,
                redistribute=weighting != "equal", min_agreement=c)
            m = compare(nav, bnav).iloc[:, 0]
            diag = result.daily.iloc[1:]
            log = pd.DataFrame(strategy.log)
            filled = log.slots_filled.mean()
            fill_q = log.assign(q=log.signal_date.dt.to_period("Q")).groupby("q").slots_filled.mean()
            out.append({"weighting": weighting, "cutoff": label, "min_agreement": c, **m.drop(["start", "end"]),
                        "avg_cash_fraction": (diag.cash / diag.nav).mean(), "avg_slots_filled": filled,
                        "min_quarter_fill": fill_q.min(), "max_quarter_fill": fill_q.max(),
                        "fees": result.fills.fee.sum()})
            print(f"{weighting:12s} cutoff {label:5s}: CAGR {m.CAGR:.2%}  Sharpe {m.Sharpe:.2f}  "
                  f"maxDD {m.max_drawdown:.2%}  cash {(diag.cash / diag.nav).mean():.1%}  slots {filled:.0%} "
                  f"(quarters {fill_q.min():.0%}-{fill_q.max():.0%})", flush=True)
    grid = pd.DataFrame(out)
    grid.loc[len(grid)] = {"weighting": "S&P 500 TR", **compare(nav, bnav).iloc[:, 1].drop(["start", "end"])}
    grid.to_csv(RESULTS / "agreement_dev_grid.csv", index=False)


if __name__ == "__main__":
    main()
