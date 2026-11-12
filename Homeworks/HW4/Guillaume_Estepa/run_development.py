"""Step 1 - development (2010-01-01 to 2015-12-31 only).

Loads data cut at 2015-12-31, compares models and portfolio rules with a
walk-forward validation, and writes the single chosen configuration to
``results/chosen_config.json``. ``run_final.py`` reads that file and never
changes it, so the test period plays no part in any choice.

Run from the repository root:
    uv run python Homeworks/HW4/Guillaume_Estepa/run_development.py
"""

import itertools
import json
from pathlib import Path
import sys

import pandas as pd

from backtester import Backtester
from data import CLASS_DB, FULL_DB, load_market_data
from features import build_panel, weekly_grid
from metrics import buy_and_hold, performance_metrics, rank_ic
from model import MODELS, refit_dates, walk_forward_predict
from strategy import TopNTrancheStrategy, build_schedule

RESULTS = Path(__file__).parent / "results"
DEV_END = "2015-12-31"
VALIDATION_START = "2012-01-01"  # 2010-2011 is the first training window

# Fixed by design (not tuned): 3 tranches 21 sessions apart, 10 bp costs,
# 30% sector cap. Compared on validation: model, N stocks, liquidity screen.
N_STOCKS = [25, 50, 100]
LIQUIDITY = [500, 1000, 2000]
N_TRANCHES, COST_RATE, SECTOR_CAP = 3, 0.001, 0.30


def main(database=FULL_DB):
    RESULTS.mkdir(exist_ok=True)
    md = load_market_data(database, end=DEV_END)
    sessions = md.sessions

    # Validation backtest: first decision at the last close of 2011.
    first_entry = sessions[sessions.searchsorted(pd.Timestamp(VALIDATION_START))]
    first_signal = sessions[sessions.get_loc(first_entry) - 1]
    schedule = build_schedule(sessions, first_signal, DEV_END, N_TRANCHES)
    panel = build_panel(md, weekly_grid(sessions).union(pd.DatetimeIndex(list(schedule))))
    print(f"Development panel: {len(panel):,} rows, {panel.date.nunique()} signal dates, "
          f"{panel.symbol.nunique()} stocks, last label end {panel.label_end.max().date()}")

    fits = refit_dates(panel.date, first_signal)
    benchmark = buy_and_hold(md.benchmark, first_entry, DEV_END)
    benchmark_metrics = performance_metrics(benchmark)

    ic_rows, backtest_rows = [], []
    for model_name in MODELS:
        panel["prediction"], fit_log, _ = walk_forward_predict(panel, model_name, fits)
        print(f"\n[{model_name}] walk-forward fits\n{fit_log.to_string(index=False)}")

        # Prediction quality: same-date rank correlation, by validation year.
        ic = rank_ic(panel)
        top = panel.dropna(subset=["prediction", "target"]).copy()
        top["decile"] = top.groupby("date").prediction.rank(pct=True)
        spread = (top[top.decile > 0.9].groupby("date").target.mean()
                  - top[top.decile <= 0.1].groupby("date").target.mean())
        for year, values in ic.groupby(ic.index.year):
            ic_rows.append({"model": model_name, "year": year, "mean_rank_ic": values.mean(),
                            "ic_hit_rate": (values > 0).mean(), "dates": values.notna().sum(),
                            "top_minus_bottom_decile": spread[spread.index.year == year].mean()})
        ic_rows.append({"model": model_name, "year": "all", "mean_rank_ic": ic.mean(),
                        "ic_hit_rate": (ic > 0).mean(), "dates": ic.notna().sum(),
                        "top_minus_bottom_decile": spread.mean()})

        # Portfolio outcome of each rule on the validation years.
        for n_stocks, liquidity in itertools.product(N_STOCKS, LIQUIDITY):
            strategy = TopNTrancheStrategy(panel, schedule, n_stocks, N_TRANCHES, liquidity, SECTOR_CAP)
            result = Backtester(md.adj_close, first_entry, DEV_END, cost_rate=COST_RATE).run(strategy)
            metrics = performance_metrics(result.daily.total_value)
            failed = [k for k, v in result.audit.items() if isinstance(v, bool) and not v]
            assert not failed, failed
            backtest_rows.append({"model": model_name, "n_stocks": n_stocks,
                                  "max_liquidity_rank": liquidity, **metrics,
                                  "excess_cagr": metrics.cagr - benchmark_metrics.cagr})

    ic_table = pd.DataFrame(ic_rows)
    backtests = pd.DataFrame(backtest_rows)
    ic_table.to_csv(RESULTS / "development_rank_ic.csv", index=False)
    backtests.to_csv(RESULTS / "development_backtests.csv", index=False)
    print("\nValidation rank IC\n", ic_table.to_string(index=False))
    print("\nValidation backtests (2012-2015)\n", backtests.round(4).to_string(index=False))
    print("\nBenchmark over the same window\n", benchmark_metrics.round(4).to_string())

    # Selection rule, written down before looking at any test data:
    # 1) model with the higher mean validation rank IC;
    # 2) for that model, the (N, liquidity) pair with the best Sharpe ratio.
    overall = ic_table[ic_table.year == "all"].set_index("model").mean_rank_ic
    best_model = overall.idxmax()
    best = backtests[backtests.model == best_model].sort_values("sharpe", ascending=False).iloc[0]
    config = {"model": best_model, "n_stocks": int(best.n_stocks),
              "max_liquidity_rank": int(best.max_liquidity_rank),
              "n_tranches": N_TRANCHES, "cost_rate": COST_RATE, "sector_cap": SECTOR_CAP,
              "refit": "at test start, then first signal of each calendar year",
              "chosen_on": f"validation {VALIDATION_START}..{DEV_END}", "database": Path(database).name}
    (RESULTS / "chosen_config.json").write_text(json.dumps(config, indent=2) + "\n")
    print("\nChosen configuration\n", json.dumps(config, indent=2))


if __name__ == "__main__":
    main(CLASS_DB if "--class-db" in sys.argv else FULL_DB)
