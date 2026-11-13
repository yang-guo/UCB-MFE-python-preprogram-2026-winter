"""Step 6 (POST-HOC, after the one final test): diagnostics that explain the result.

    uv run python run_posthoc.py        # ~2 min

Nothing here changes the headline result. It answers "why did we lose to the
benchmark?" with evidence:
  1. random-pick null: identical rules/costs/caps, but the 25 names are drawn at
     random from the same eligible universe (20 seeds). If random picks also lag
     SPX TR, the gap is the universe/weighting, not the model's ranking.
  2. score-decile table and per-year top-25 vs universe returns.
  3. forced-exit lots classified (acquisition / SPAC liquidation / ticker issue).
  4. what the benchmark's biggest names did (mega-cap concentration).
"""

import json

import duckdb
import numpy as np
import pandas as pd

from hw4 import analytics as A, config
from hw4.backtester import Backtester, TopNCohortStrategy
from run_strategy_dev import load_prices_wide, schedule_for

FORCED_EXIT_NOTES = {   # my notes from public deal news, NOT in the database (no delisting proceeds there)
    "AEL": "cash+stock acquisition by Brookfield (2024)", "AHL": "likely acquisition (unverified)",
    "CERN": "cash acquisition by Oracle (2022)", "CHK": "merger with Southwestern -> Expand Energy (ticker change)",
    "DCP": "bought in by Phillips 66 (2023)", "EA": "take-private (2026, announced 2025)", "FOE": "cash acquisition by Prince (2022)",
    "HIBB": "cash acquisition by JD Sports (2024)", "MEG": "2017 Media General -> Nexstar; ticker later reused",
    "MMP": "acquired by ONEOK (2023)", "MPLXP": "preferred units, likely redeemed/converted (unverified)",
    "PDCE": "acquired by Chevron (2023)", "PGTI": "cash acquisition by MITER (2024)",
    "PSTH": "SPAC trust liquidated, cash returned (2022)", "REVG": "likely merger/acquisition (unverified)",
    "SAFM": "cash acquisition by Cargill/Continental Grain (2022)", "SC": "cash buy-in by Santander (2022)",
    "SEE": "likely take-private (unverified)", "UFS": "cash acquisition by Paper Excellence (2021)",
    "UNVR": "cash take-private by Apollo (2023)", "VRTV": "cash take-private by CD&R (2023)",
    "WSTC": "likely take-private (unverified)",
}


def main():
    cfg = json.loads((config.HW_DIR / "frozen_config.json").read_text())
    st = cfg["strategy"]
    cache = duckdb.connect(str(config.OUTPUT_DIR / "cache_top2000.duckdb"), read_only=True)
    scores = pd.read_csv(config.OUTPUT_DIR / "final_scores.csv", parse_dates=["date"])
    meta = cache.execute("SELECT date, symbol, sector, liq_rank, target FROM panel WHERE date >= ? AND date < ?",
                         [config.TEST_FIRST_SIGNAL, config.TEST_END]).fetchdf()
    scores = scores.merge(meta, on=["date", "symbol"])
    prices = load_prices_wide(cache, "2016-03-01", config.TEST_END)
    bench = cache.execute("SELECT date, adj_close FROM bench").fetchdf().set_index("date").adj_close
    first, end = pd.Timestamp(config.TEST_FIRST_SIGNAL), pd.Timestamp(config.TEST_END)
    sched = schedule_for(scores.date.unique(), st["cadence"])
    sched = pd.DatetimeIndex([first]).append(sched[(sched > first + pd.Timedelta(days=1)) & (sched < end)])
    bnav = A.benchmark_nav(bench, first, prices.index[prices.index.get_loc(first) + 1], end)
    hist = pd.read_csv(config.OUTPUT_DIR / "final_history.csv", index_col=0, parse_dates=True)
    R = config.RESULTS_DIR

    # 1. random-pick null distribution -------------------------------------------
    rows = []
    for seed in range(20):
        rng = np.random.default_rng(seed)
        s = scores[["date", "symbol", "sector", "liq_rank"]].assign(score=rng.random(len(scores)))
        strat = TopNCohortStrategy(s, sched, top_n=st["top_n"], cohort_fraction=st["cohort_fraction"],
                                   name_cap=st["name_cap"], sector_cap=st["sector_cap"])
        r = Backtester(prices, strat, None).run(first, end)
        p = A.performance(r.history.nav, bnav, history=r.history)
        rows.append({"seed": seed, **{k: p[k] for k in ["final_value", "CAGR", "sharpe", "max_drawdown", "beta"]}})
        print(f"random seed {seed:>2}: final {p['final_value']:,.0f}  CAGR {p['CAGR']:.4f}")
    null = pd.DataFrame(rows)
    strat_perf = A.performance(hist.nav, bnav)
    pct = (null.CAGR < strat_perf["CAGR"]).mean()
    summary = pd.DataFrame({
        "strategy": {k: strat_perf[k] for k in ["final_value", "CAGR", "sharpe", "max_drawdown"]},
        "random_25_mean": null[["final_value", "CAGR", "sharpe", "max_drawdown"]].mean(),
        "random_25_min": null[["final_value", "CAGR", "sharpe", "max_drawdown"]].min(),
        "random_25_max": null[["final_value", "CAGR", "sharpe", "max_drawdown"]].max(),
        "benchmark": {k: A.performance(bnav)[k] for k in ["final_value", "CAGR", "sharpe", "max_drawdown"]},
    })
    summary.loc["strategy_CAGR_percentile_vs_random"] = [pct, np.nan, np.nan, np.nan, np.nan]
    summary.round(4).to_csv(R / "posthoc_random_pick_null.csv")
    print(summary.round(4).to_string())

    # 2. deciles and top-25 vs universe by year -----------------------------------
    lab = scores.loc[scores.target.notna()].copy()
    lab["excess"] = lab.target - lab.groupby("date").target.transform("mean")
    lab["decile"] = lab.groupby("date").score.transform(lambda x: pd.qcut(x.rank(method="first"), 10, labels=False) + 1)
    dec = lab.groupby("decile").agg(mean_excess=("excess", "mean"), median_excess=("excess", "median"),
                                    mean_ret=("target", "mean"), hit_rate=("excess", lambda x: (x > 0).mean()))
    dec.round(4).to_csv(R / "posthoc_decile_table.csv")
    print(dec.round(4).to_string())
    lab["top25"] = lab.groupby("date").score.rank(ascending=False, method="first") <= 25
    yr = lab.groupby([lab.date.dt.year, "top25"]).target.mean().unstack()
    yr.columns = ["rest_of_universe", "top25_picks"]
    yr["picks_minus_rest"] = yr.top25_picks - yr.rest_of_universe
    yr.round(4).to_csv(R / "posthoc_top25_vs_universe_by_year.csv")
    print(yr.round(4).to_string())

    # 3. forced exits ------------------------------------------------------------
    fx = pd.read_csv(R / "forced_exit_lots.csv")
    g = fx.groupby("symbol").agg(lots=("lot_id", "count"), cost=("cost", "sum"),
                                 exit_value=("proceeds", "sum"), pnl=("realized_pnl", "sum"))
    g["likely_event_not_from_db"] = g.index.map(FORCED_EXIT_NOTES).fillna("unknown")
    g.round(2).to_csv(R / "forced_exits_classified.csv")
    print(g.round(0).to_string())


if __name__ == "__main__":
    main()
