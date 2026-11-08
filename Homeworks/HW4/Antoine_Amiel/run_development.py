"""Step 1 – development only (signals 2010-01-01 .. 2015-12-31).

Compares models with expanding-window yearly folds, then backtests a small
grid of trading rules on 2012–2015 with the same quarterly refit schedule that
the final test uses. Writes results/development/* and the frozen choice to
results/chosen_config.json. Nothing here reads a signal after 2015-12-31.

    python run_development.py
"""

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from hw4.backtester import Backtester, performance  # noqa: E402
from hw4.data import load_market_data  # noqa: E402
from hw4.features import FeatureStore  # noqa: E402
from hw4.model import (WalkForwardScorer, build_panel, daily_ic, fit, glitch_flags, predict,  # noqa: E402
                       top_bucket_excess, training_dates)
from hw4.strategy import Ranked, SleeveTopN  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)
OUT = HERE / "results/development"
DEV_END = pd.Timestamp("2015-12-31")
VALIDATION_YEARS = [2012, 2013, 2014, 2015]
SCREENS = {"top500": 500, "top1000": 1000, "top2000": None}
BASELINES = {"momentum_12_1": ("r_mom_252_21", 1.0), "low_volatility": ("r_vol_63", -1.0),
             "value_earnings_yield": ("r_earnings_yield", 1.0), "quality_roa": ("r_roa", 1.0)}


def fold_validation(panel, glitch, calendar):
    """Expanding-window yearly folds; rows the strategy would screen out are dropped."""
    flagged = glitch.to_numpy()[calendar.get_indexer(panel.date), glitch.columns.get_indexer(panel.symbol)]
    panel = panel.loc[~flagged]
    rows = []
    for year in VALIDATION_YEARS:
        valid = panel.loc[panel.date.dt.year.eq(year) & panel.target.notna()].copy()
        first = valid.date.min()
        train = panel.loc[panel.date.lt(first) & panel.label_end.lt(first) & panel.target.notna()]
        assert train.label_end.max() < first
        scored = {name: valid[col].fillna(0.5) * sign for name, (col, sign) in BASELINES.items()}
        for kind in ["ridge", "hgb"]:
            scored[kind] = predict(fit(kind, train), valid)
        for screen, max_rank in SCREENS.items():
            sub = valid if max_rank is None else valid.loc[valid.universe_rank.le(max_rank)]
            for name, score in scored.items():
                frame = sub.assign(score=np.asarray(score)[valid.index.get_indexer(sub.index)])
                rows.append({"year": year, "screen": screen, "model": name, "train_rows": len(train),
                             "rank_IC": daily_ic(frame).mean(),
                             "top_decile_excess_90d": top_bucket_excess(frame, q=0.1).mean()})
    return pd.DataFrame(rows)


def dev_backtests(store, panel, glitch, adj_dev):
    start, end = pd.Timestamp("2012-01-03"), DEV_END
    cal = store.calendar
    fit_dates = [cal[cal >= q][0] for q in pd.date_range("2012-01-01", "2015-10-01", freq="QS")]
    # The first signal (2011-12-30) precedes the first scheduled fit, so add it.
    fit_dates = [cal[cal.get_loc(start) - 1]] + fit_dates[1:]
    shared = {kind: ({}, []) for kind in ["ridge", "hgb"]}
    results, navs = [], {}
    bench = store.md.benchmark.loc[start:end]
    configs = [(m, s, n, "equal") for m in ["hgb", "ridge", "momentum_12_1"] for s in SCREENS for n in [30, 60]]
    configs += [(m, s, n, w) for m in ["ridge", "momentum_12_1"] for s in ["top500", "top1000"]
                for n, w in [(100, "equal"), (30, "cap"), (100, "cap")]]
    for model, screen, n, weighting in configs:
        if model in shared:
            scorer = WalkForwardScorer(store, panel, model, fit_dates, SCREENS[screen], glitch,
                                       shared_models=shared[model][0], shared_audit=shared[model][1])
        else:
            col, sign = BASELINES[model]
            scorer = Ranked(store, col.removeprefix("r_"), SCREENS[screen], glitch, sign)
        strat = SleeveTopN(scorer, cal, start, n_names=n, weighting=weighting)
        res = Backtester(adj_dev, strat).run(start, end)
        nav = res.history.nav
        b = 1_000_000 * bench / bench.iloc[0]
        b = pd.concat([pd.Series({nav.index[0]: 1_000_000.0}), b])
        perf, bperf = performance(nav), performance(b)
        name = f"{model}|{screen}|{n}|{weighting}"
        navs[name] = nav
        results.append({"model": model, "screen": screen, "names_per_sleeve": n, "weighting": weighting,
                        **perf.to_dict(),
                        "excess_CAGR": perf.CAGR - bperf.CAGR, "bench_CAGR": bperf.CAGR,
                        "bench_Sharpe": bperf.Sharpe, "avg_cash_frac": (res.history.cash / nav).iloc[63:].mean(),
                        "price_incidents": len(res.incidents),
                        "incident_pnl_frac": res.incidents.pnl_impact.sum() / 1_000_000})
        print(f"{name:28s} CAGR {perf.CAGR:6.1%}  Sharpe {perf.Sharpe:5.2f}  MDD {perf.max_drawdown:6.1%}"
              f"  bench CAGR {bperf.CAGR:6.1%}  incidents {len(res.incidents)}", flush=True)
    return pd.DataFrame(results), pd.DataFrame(navs), shared


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    md = load_market_data()
    store = FeatureStore(md)
    glitch = glitch_flags(store)
    dates = training_dates(store.calendar, "2010-01-01", DEV_END)
    panel = build_panel(store, dates)
    assert panel.date.max() <= DEV_END
    print(f"data + features + panel: {time.time() - t0:.0f}s, panel rows {len(panel):,}", flush=True)

    folds = fold_validation(panel, glitch, store.calendar)
    folds.to_csv(OUT / "fold_validation.csv", index=False)
    summary = folds.groupby(["screen", "model"])[["rank_IC", "top_decile_excess_90d"]].mean().unstack(0)
    summary.to_csv(OUT / "fold_validation_summary.csv")
    print(summary.round(4).to_string(), flush=True)

    # Backtests only ever see quotes through the end of development.
    adj_dev = md.adj.loc[:DEV_END]
    table, navs, shared = dev_backtests(store, panel, glitch, adj_dev)
    table.to_csv(OUT / "dev_backtests.csv", index=False)
    navs.to_csv(OUT / "dev_backtest_navs.csv")
    for kind, (_, audit) in shared.items():
        pd.DataFrame(audit).to_csv(OUT / f"fit_audit_{kind}.csv", index=False)
    print(table.sort_values("Sharpe", ascending=False).round(3).to_string(), flush=True)


if __name__ == "__main__":
    main()
