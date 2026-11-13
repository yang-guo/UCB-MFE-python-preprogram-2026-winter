"""Step 1: choose the model and strategy using the development period only.

Run:  python validate.py            (full database)
      python validate.py --class    (small class database, to test the code)

Only signals from 2010-2015 are used here. Walk-forward validation: before
each year 2012-2015 I refit on outcomes that had already ended, then predict
that year's weekly signals. Nothing after the 2016 Q1 buffer is touched.
"""

import argparse
import json

import pandas as pd

import research
from backtester import Backtester, benchmark_nav, performance
from pipeline import RESULTS, load, log, scores_for, sectors, tradeable

DEV_END = "2015-12-31"
VAL_YEARS = [2012, 2013, 2014, 2015]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--class", dest="class_db", action="store_true")
    args = ap.parse_args()
    db = research.CLASS_DB if args.class_db else research.FULL_DB
    md, panel = load(db)
    dev = panel.loc[panel.date.le(DEV_END)].reset_index(drop=True)
    X = research.rank_features(dev)

    # Refit at the first (January) signal of each validation year, using only
    # outcomes that ended before that date, and predict that whole year.
    first_val = [dev.loc[dev.date.dt.year.eq(y), "date"].min() for y in VAL_YEARS]

    # ---- 1. signal quality of each model/target pair
    rows, all_preds = [], {}
    for model in ["ridge", "hgb"]:
        for target in ["rank", "excess"]:
            preds, fits = research.walk_forward(dev, X, first_val, model, target)
            q = research.signal_quality(dev, preds)
            q["model"], q["target"] = model, target
            rows.append(q)
            all_preds[(model, target)] = preds
            log(f"{model}/{target}: IC {q.mean_rank_IC:.4f}  top50-all {q['top50_minus_all']:.4f}")
    ic_table = pd.DataFrame(rows).set_index(["model", "target"])
    ic_table.to_csv(RESULTS / "validation_signal_quality.csv")
    print(ic_table.round(4).to_string())

    # ---- 2. dev-period backtests of the strategy settings
    prices = md.prices_frame()
    sec = sectors(md)
    start, end = "2012-01-01", DEV_END
    bench = performance(benchmark_nav(md.benchmark, start, end))
    bt_rows = []
    for (model, target), preds in all_preds.items():
        for min_dv in [1e6, 5e6, 25e6]:
            for top_n in [25, 50, 100]:
                s = scores_for(dev, preds, tradeable(dev, min_dv))
                strategy = research.WeeklyTopN(s, sec, top_n=top_n)
                res = Backtester(prices, cost_bps=10).run(strategy, start, end)
                p = performance(res.nav)
                bt_rows.append({"model": model, "target": target, "min_dollar_volume": min_dv, "top_n": top_n,
                                "CAGR": p.CAGR, "Sharpe": p.Sharpe, "max_drawdown": p.max_drawdown,
                                "final_value": p.final_value, "avg_cash_share": (res.daily.cash / res.daily.nav).mean(),
                                "skipped_buys": int(res.events.event.str.startswith("buy_skipped").sum())})
    bt = pd.DataFrame(bt_rows)
    bt.loc[len(bt)] = {"model": "S&P 500 TR", "CAGR": bench.CAGR, "Sharpe": bench.Sharpe,
                       "max_drawdown": bench.max_drawdown, "final_value": bench.final_value}
    bt.to_csv(RESULTS / "validation_backtests.csv", index=False)
    print(bt.round(3).to_string())

    # ---- 3. pick: best model/target by mean IC, then best setting by dev Sharpe
    best_model, best_target = ic_table.mean_rank_IC.idxmax()
    cand = bt[(bt.model == best_model) & (bt.target == best_target)]
    best = cand.loc[cand.Sharpe.idxmax()]
    choice = {"model": best_model, "target": best_target, "top_n": int(best.top_n),
              "min_dollar_volume": float(best.min_dollar_volume), "cohort_fraction": 1 / 14, "name_cap": 0.05,
              "sector_cap": 0.30, "cost_bps": 10, "retrain": "every 12 months, at the last signal in March"}
    print("chosen:", choice)
    if not args.class_db:
        (RESULTS / "chosen_config.json").write_text(json.dumps(choice, indent=2))


if __name__ == "__main__":
    main()
