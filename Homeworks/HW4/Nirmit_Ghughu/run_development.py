"""Development experiment: choose the model and strategy using 2010-2015 only.

The database is read with ``end='2015-12-31'``, so no code path here can see a
price or filing after the development period.

Validation design (chronological walk-forward):
  * Training signals start 2010-01-01 (earlier data only feeds lookbacks).
  * Quarterly refits from 2012-Q1 to 2015-Q4.  Each fit uses only rows whose
    90-day outcome ended before the fit date, so there is a ~90-day purge gap.
  * Out-of-sample predictions for 2012-2015 are scored two ways:
      - prediction quality: weekly rank IC and top-decile excess 90-day return;
      - portfolio quality: a full backtest (same rules as the final test) vs ^SP500TR.

Run:  python run_development.py      (about 5-10 minutes)
"""

from pathlib import Path
import time

import pandas as pd

from backtester import Backtester, TopNStrategy, performance
from data import FULL_DB, build_panel
from model import add_rank_features, prediction_metrics, walk_forward_predictions

OUT = Path(__file__).parent / "results"
OUT.mkdir(exist_ok=True)
DEV_END = "2015-12-31"
VAL_START = "2012-01-01"


def dev_backtest(pred, market, top_n, start="2012-01-03", end=DEV_END):
    strat = TopNStrategy(pred, market.sectors, top_n=top_n)
    res = Backtester(market.adj_close, strat).run(start, end)
    bench = market.benchmark.loc[res.history.index]
    bench = 1_000_000 * bench / bench.iloc[0]
    return res, performance(res.history.nav), performance(bench)


def main():
    t0 = time.time()
    panel, market = build_panel(FULL_DB, end=DEV_END)
    ranked = add_rank_features(panel)
    print(f"panel {panel.shape}, tradable rows {len(ranked):,}  ({time.time() - t0:.0f}s)")

    candidates = {
        "ridge_raw": ("ridge", "raw"), "ridge_rank": ("ridge", "rank"),
        "hgb_raw": ("hgb", "raw"), "hgb_rank": ("hgb", "rank"),
        "ridge_core_rank": ("ridge_core", "rank"), "hgb_core_rank": ("hgb_core", "rank"),
    }
    preds, metric_rows = {}, []
    for name, (kind, target) in candidates.items():
        print(f"walk-forward {name}")
        pred, fits = walk_forward_predictions(ranked, VAL_START, DEV_END, kind=kind, target=target, verbose=False)
        preds[name] = pred
        m = prediction_metrics(pred)
        for year, g in pred.groupby(pred.date.dt.year):
            m[f"IC_{year}"] = prediction_metrics(g).mean_rank_IC
        metric_rows.append(m.rename(name))
    # A no-model baseline: rank by 12-1 momentum (already a percentile rank).
    mom = ranked.loc[ranked.date >= VAL_START, ["date", "symbol", "sector", "target", "label_end"]].assign(
        score=ranked.loc[ranked.date >= VAL_START, "mom_252_21"].fillna(0).to_numpy())
    preds["momentum_12_1"] = mom
    # Blend: average of the per-date percentile ranks of a model and of momentum.
    blend = preds["ridge_core_rank"].merge(mom[["date", "symbol", "score"]], on=["date", "symbol"], suffixes=("", "_mom"))
    g = blend.groupby("date")
    blend["score"] = g.score.rank(pct=True) + g.score_mom.rank(pct=True)
    preds["blend_core_mom"] = blend.drop(columns="score_mom")
    for name in ("momentum_12_1", "blend_core_mom"):
        m = prediction_metrics(preds[name])
        for year, gy in preds[name].groupby(preds[name].date.dt.year):
            m[f"IC_{year}"] = prediction_metrics(gy).mean_rank_IC
        metric_rows.append(m.rename(name))
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(OUT / "dev_prediction_metrics.csv", float_format="%.4f")
    print(metrics.round(4).to_string())

    rows = []
    for name, pred in preds.items():
        for top_n in (10, 25, 50):
            res, perf, bperf = dev_backtest(pred, market, top_n)
            rows.append(dict(model=name, top_n=top_n, **perf.add_prefix("strat_"),
                             bench_CAGR=bperf.CAGR, bench_Sharpe=bperf.Sharpe,
                             avg_cash_frac=(res.history.cash / res.history.nav).mean(),
                             fees=res.fills.fee.sum()))
            print(f"  {name:14s} N={top_n:<3d} CAGR {perf.CAGR:7.2%}  Sharpe {perf.Sharpe:5.2f}  "
                  f"MDD {perf.max_drawdown:7.2%}   bench CAGR {bperf.CAGR:7.2%}")
    bt = pd.DataFrame(rows)
    bt.to_csv(OUT / "dev_backtests.csv", index=False, float_format="%.4f")
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
