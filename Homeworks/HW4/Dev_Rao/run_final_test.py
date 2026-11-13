"""Step 5: the ONE final test, Apr 1 2016 - Sep 30 2026, with the frozen config.

    uv run python run_final_test.py          # ~5-20 min depending on the model

Refuses to run if frozen_config.json is missing or the database fingerprint
changed, and refuses a second run (outputs/final_test_ran.json) unless
--rerun-reason is given (the reason is logged and must be reported).
Sensitivities (fees 0/25 bps, 0% recovery on forced exits, 2% cash interest)
are computed in the same invocation and are labelled as such.
"""

import argparse
import json
import time
from datetime import datetime, timezone

import duckdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hw4 import analytics as A, config, data, model as M
from hw4.backtester import Backtester, TopNCohortStrategy
from hw4.features import MODEL_CATEGORICAL, MODEL_NUMERIC
from hw4.labels import add_bench_forward
from run_strategy_dev import load_prices_wide, schedule_for

# The real test window, and a dry-run window inside development used only to
# exercise this exact code path before the one real run (no 2016+ data read).
WINDOWS = {
    "test": {"first_signal": config.TEST_FIRST_SIGNAL, "first_entry": config.TEST_START,
             "end": config.TEST_END, "prices_from": "2016-03-01"},
    "dryrun": {"first_signal": "2013-01-04", "first_entry": "2013-01-07",
               "end": config.DEV_END, "prices_from": "2012-12-01"},
}
W = WINDOWS["test"]

EXPOSURE_COLS = ["r_log_mcap", "r_log_dv_63", "r_beta_252", "r_vol_63", "r_mom_12_1", "r_bm", "r_ey",
                 "r_gp_a", "r_roe", "r_asset_growth", "r_share_iss"]


def load_panel(cache):
    cols = list(dict.fromkeys(["date", "symbol", "sector", "target", "entry_date", "label_end", "liq_rank",
                               "beta_252"] + MODEL_NUMERIC + EXPOSURE_COLS))
    p = cache.execute(f"SELECT {', '.join(cols)} FROM panel WHERE date >= ? AND date < ? ORDER BY date, symbol",
                      [config.TRAIN_START, W["end"]]).fetchdf()
    assert p.date.max() < pd.Timestamp(W["end"])
    for c in MODEL_NUMERIC:
        p[c] = p[c].astype("float32")
    bench = cache.execute("SELECT date, adj_close FROM bench").fetchdf().set_index("date").adj_close
    return add_bench_forward(p, bench), bench


def quarterly_fit_dates(test_signals):
    s = pd.DatetimeIndex(sorted(test_signals))
    first = pd.Timestamp(W["first_signal"])
    later = pd.Series(s[s > first], index=s[s > first])
    q_firsts = later.groupby(later.index.to_period("Q")).min()
    q_firsts = q_firsts[q_firsts.index > first.to_period("Q")]       # 2016-Q2 is covered by `first`
    return pd.DatetimeIndex([first]).append(pd.DatetimeIndex(q_firsts.values))


def score_test(panel, spec, fit_dates, sample_every, log):
    """Walk forward: each fit scores signals until the next fit date."""
    test = panel.date >= pd.Timestamp(W["first_signal"])
    out = panel.loc[test, ["date", "symbol"]].copy()
    members = spec["members"] if spec["kind"] == "ensemble" else [spec]
    for m in members:
        col = m["name"]
        out[col] = np.nan
        bounds = list(fit_dates) + [pd.Timestamp(W["end"]) + pd.Timedelta(days=1)]
        for f0, f1 in zip(bounds[:-1], bounds[1:]):
            mask = (panel.date >= f0) & (panel.date < f1)
            if not mask.any():
                continue
            t1 = time.time()
            pred, _, n = M.fit_predict(panel, m, f0, mask.to_numpy(), sample_every=sample_every)
            out.loc[mask[mask].index, col] = pred          # every scored row is a test row (>= f0)
            log.append({"model": col, "fit_date": f0.date(), "n_train": n, "n_scored": int(mask.sum()),
                        "seconds": round(time.time() - t1, 1)})
            print(f"  fit {col} @ {f0.date()}  n_train={n:,}  scored={int(mask.sum()):,}  {time.time()-t1:.0f}s")
    names = [m["name"] for m in members]
    out["score"] = M.rank_average(out, names) if len(names) > 1 else out[names[0]]
    return out


def main(rerun_reason=None, dryrun=False):
    global W
    W = WINDOWS["dryrun" if dryrun else "test"]
    t0 = time.time()
    frozen = config.HW_DIR / "frozen_config.json"
    cfg = json.loads(frozen.read_text())
    marker = config.OUTPUT_DIR / "final_test_ran.json"
    if marker.exists() and not rerun_reason and not dryrun:
        raise SystemExit("The final test already ran once. Re-running requires --rerun-reason (it is logged).")
    fp = data.database_fingerprint(config.DATABASES["top2000"])
    assert fp["sha256"] == cfg["database"]["sha256"], "database changed since freezing"
    assert cfg["features"]["numeric"] == MODEL_NUMERIC, "feature list changed since freezing"
    assert cfg["backtest"]["fee_bps_per_side"] == config.FEE_BPS

    cache = duckdb.connect(str(config.OUTPUT_DIR / "cache_top2000.duckdb"), read_only=True)
    panel, bench = load_panel(cache)
    print(f"[{time.time()-t0:5.0f}s] panel {len(panel):,} rows; refitting walk-forward ...")
    test_signals = panel.loc[panel.date >= W["first_signal"], "date"].unique()
    fit_dates = quarterly_fit_dates(test_signals)
    if cfg["model"].get("kind") == "composite":
        fit_dates = fit_dates[:1]                                     # nothing to refit
    fit_log = []
    scores = score_test(panel, cfg["model"], fit_dates, cfg["training"]["sample_every"], fit_log)
    meta = panel[["date", "symbol", "sector", "liq_rank", "target", "label_end"] + EXPOSURE_COLS]
    scores = scores.merge(meta, on=["date", "symbol"], validate="one_to_one")

    # ---- backtests -------------------------------------------------------------
    st = cfg["strategy"]
    sched = schedule_for(test_signals, st["cadence"])
    first = pd.Timestamp(W["first_signal"])
    sched = pd.DatetimeIndex([first]).append(sched[sched > first + pd.Timedelta(days=1)])
    sched = sched[sched < pd.Timestamp(W["end"])]
    prices = load_prices_wide(cache, W["prices_from"], W["end"])
    first_entry = prices.index[prices.index.get_loc(first) + 1]
    assert first_entry == pd.Timestamp(W["first_entry"])
    end = pd.Timestamp(W["end"])
    bnav = A.benchmark_nav(bench, first, first_entry, end)

    def run(fee=config.FEE_BPS, recovery=1.0, cash_rate=0.0):
        strat = TopNCohortStrategy(scores[["date", "symbol", "score", "sector", "liq_rank"]], sched,
                                   top_n=st["top_n"], cohort_fraction=st["cohort_fraction"],
                                   name_cap=st["name_cap"], sector_cap=st["sector_cap"], universe=st["universe"],
                                   liq_top=st["liquid_top"])
        return Backtester(prices, strat, None, fee_bps=fee, forced_exit_recovery=recovery,
                          cash_rate_annual=cash_rate).run(first, end)

    runs = {"strategy (10 bps, base)": run()}
    print(f"[{time.time()-t0:5.0f}s] base backtest done; sensitivities ...")
    runs["sensitivity: 0 bps"] = run(fee=0)
    runs["sensitivity: 25 bps"] = run(fee=25)
    runs["sensitivity: 0% recovery on forced exits"] = run(recovery=0.0)
    runs["sensitivity: cash earns 2%/yr (assumed, not T-bill data)"] = run(cash_rate=0.02)

    rows = [{"portfolio": "S&P 500 TR benchmark", **A.performance(bnav)}]
    for name, r in runs.items():
        rows.append({"portfolio": name, **A.performance(r.history.nav, bnav, history=r.history)})
    metrics = pd.DataFrame(rows)
    base = runs["strategy (10 bps, base)"]
    nav = base.history.nav

    # ---- predictive metrics in the test period ----------------------------------
    labelled = scores.loc[scores.target.notna() & (scores.label_end <= end)]
    rep, ic, dec = M.ranking_report(labelled, "score")
    ic_year = ic.groupby(ic.index.year).agg(["mean", "count"]).rename(columns={"mean": "IC", "count": "weeks"})

    # ---- exposures: cost-weighted feature ranks of purchases vs the universe (0.5)
    buys = base.fills.loc[base.fills.side == "BUY", ["signal_date", "symbol", "notional"]]
    bx = buys.merge(scores, left_on=["signal_date", "symbol"], right_on=["date", "symbol"])
    expo = pd.DataFrame({"purchases_cost_weighted": {c: np.average(bx[c].fillna(0.5), weights=bx.notional)
                                                     for c in EXPOSURE_COLS},
                         "eligible_universe": {c: 0.5 for c in EXPOSURE_COLS}})
    sec_buy = bx.groupby("sector").notional.sum() / bx.notional.sum()
    sec_univ = scores.groupby("sector").size() / len(scores)
    sectors = pd.DataFrame({"strategy_purchases": sec_buy, "eligible_universe_rows": sec_univ}).fillna(0)

    # ---- save ------------------------------------------------------------------
    R, O = config.RESULTS_DIR, config.OUTPUT_DIR
    if dryrun:
        R = O = config.OUTPUT_DIR / "dryrun_2013_2015"
        R.mkdir(exist_ok=True)
    R.mkdir(exist_ok=True)
    metrics.round(6).to_csv(R / "final_metrics.csv", index=False)
    years = A.calendar_year_returns({"strategy": nav, "benchmark": bnav})
    years["excess"] = years.strategy - years.benchmark
    years.round(5).to_csv(R / "calendar_year_returns.csv")
    pd.DataFrame({"strategy_nav": nav, "benchmark_nav": bnav, "cash": base.history.cash,
                  "open_lots": base.history.open_lots}).round(2).to_csv(R / "daily_nav.csv")
    ic_year.round(4).to_csv(R / "test_ic_by_year.csv")
    pd.Series(rep).to_csv(R / "test_ranking_metrics.csv", header=["value"])
    dec.rename("mean_excess_vs_universe").to_csv(R / "test_decile_spread.csv")
    expo.round(4).to_csv(R / "exposures.csv")
    sectors.round(4).to_csv(R / "sector_weights.csv")
    pd.DataFrame(fit_log).to_csv(R / "test_refit_log.csv", index=False)
    base.fills.to_csv(O / "final_fills.csv", index=False)
    base.lots.to_csv(O / "final_lots.csv", index=False)
    base.diagnostics.to_csv(O / "final_diagnostics.csv", index=False)
    base.history.to_csv(O / "final_history.csv")
    scores[["date", "symbol", "score"]].to_csv(O / "final_scores.csv", index=False)
    forced = base.lots.loc[base.lots.exit_reason == "no_quote_forced_exit"]
    forced.to_csv(R / "forced_exit_lots.csv", index=False)
    first_lot = base.lots.sort_values("lot_id").iloc[0].to_dict()
    (R / "lot_trace.json").write_text(json.dumps(first_lot, indent=2, default=str))
    make_charts(nav, bnav, ic_year, dec, R)

    if not dryrun:
        marker.write_text(json.dumps({"ran_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                      "frozen_config_sha256": data.file_sha256(frozen),
                                      "rerun_reason": rerun_reason}, indent=2))
    pd.set_option("display.width", 220)
    show = ["portfolio", "final_value", "CAGR", "max_drawdown", "sharpe", "volatility", "beta",
            "tracking_error", "information_ratio", "avg_cash_fraction", "annual_turnover", "total_fees"]
    print(metrics[[c for c in show if c in metrics]].round(4).to_string(index=False))
    print(years.round(4).to_string())
    print("test ranking:", {k: round(v, 4) if isinstance(v, float) else v for k, v in rep.items()})
    print(ic_year.round(4).to_string())
    print(expo.round(3).to_string()); print(sectors.round(3).to_string())
    print(f"forced exits: {len(forced)} lots, cost ${forced.cost.sum():,.0f}, "
          f"exit value ${(forced.units * forced.exit_price).sum():,.0f}")
    print(f"done in {time.time()-t0:.0f}s")


def make_charts(nav, bnav, ic_year, dec, R):
    fig, ax = plt.subplots(figsize=(11, 5))
    (nav / nav.iloc[0] - 1).plot(ax=ax, label="Strategy", color="#1f77b4", lw=1.6)
    (bnav / bnav.iloc[0] - 1).plot(ax=ax, label="S&P 500 TR (^SP500TR)", color="#444444", lw=1.6)
    ax.set(title=f"Cumulative return, {nav.index[1].date()} to {nav.index[-1].date()} (both start with $1M at the same close)",
           ylabel="Cumulative return", xlabel="")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(R / "cumulative_returns.png", dpi=140); plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 3.6))
    A.drawdown(nav).plot(ax=ax, label="Strategy", color="#1f77b4")
    A.drawdown(bnav).plot(ax=ax, label="S&P 500 TR", color="#444444")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set(title="Drawdown from running peak", xlabel=""); ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(R / "drawdowns.png", dpi=140); plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 3.6))
    ex = A.rolling_excess(nav, bnav)
    ax.fill_between(ex.index, ex.values, 0, where=ex.values >= 0, color="#2ca02c", alpha=0.5)
    ax.fill_between(ex.index, ex.values, 0, where=ex.values < 0, color="#d62728", alpha=0.5)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set(title="Rolling 1-year return: strategy minus benchmark", xlabel=""); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(R / "rolling_excess.png", dpi=140); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
    ic_year.IC.plot.bar(ax=ax[0], color=["#2ca02c" if v > 0 else "#d62728" for v in ic_year.IC], rot=0)
    ax[0].axhline(0, color="k", lw=0.8); ax[0].set(title="Test-period mean weekly rank IC by year", xlabel="")
    dec.plot.bar(ax=ax[1], color="#1f77b4", rot=0)
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax[1].set(title="Test: mean 90d return by score decile minus universe", xlabel="decile (10 = best score)")
    fig.tight_layout(); fig.savefig(R / "test_ic_and_deciles.png", dpi=140); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rerun-reason", default=None)
    ap.add_argument("--dry-run-dev", action="store_true",
                    help="run the identical pipeline on 2013-2015 only (no test data)")
    a = ap.parse_args()
    main(a.rerun_reason, a.dry_run_dev)
