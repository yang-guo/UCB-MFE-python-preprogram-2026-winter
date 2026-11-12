"""HW4 pipeline.

    python run_hw4.py dev     # 2010-2015 only: feature screen, walk-forward validation, confirms features and final setup
    python run_hw4.py test    # final test Apr 2016 - Sep 2026: momentum (submitted) + LightGBM and Ridge comparisons

Every result table goes into one file, ``results/results.csv`` (long format:
section, name, metric, value), next to the two charts. The cached panel and full
trade logs go to ``outputs/`` (git-ignored).
"""

import argparse
import json
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from hw4 import data, features as F, model as M  # noqa: E402
from hw4.backtest import Backtester, BuyAndHold, TopNStrategy, performance  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS, OUTPUTS = HERE / "results", HERE / "outputs"
DEV_END, SCREEN_END = pd.Timestamp("2015-12-31"), pd.Timestamp("2013-01-01")
TEST_START, TEST_END = pd.Timestamp("2016-04-01"), pd.Timestamp("2026-09-30")
SIGNAL_EVERY = 10          # sessions between cohorts (~2 weeks); features are sampled every 5
FEE_BPS = 10               # per side: commission + spread/impact assumption
CORR_CUTOFF = 0.4
VALIDATION_YEARS = [2013, 2014, 2015]

LGB_SMALL = dict(num_leaves=7, n_estimators=300, min_child_samples=5000)
FITTED = {   # name -> factory(features, sectors); refit at every fit date
    "lgb_excess": lambda feats, sectors: M.RankLGBM(feats, sectors, target="excess", **LGB_SMALL),
    "lgb_rank": lambda feats, sectors: M.RankLGBM(feats, sectors, target="rank", **LGB_SMALL),
    "lgb_sector": lambda feats, sectors: M.RankLGBM(feats, sectors, use_sector=True, num_leaves=15, n_estimators=300),
    "ridge": lambda feats, sectors: M.RankRidge(feats),
}
RULES = {    # name -> score from signal-date columns only (no fitting)
    "momentum": lambda rows: rows.r_mom_12_1.to_numpy(),          # 12-1 momentum percentile
}
MODELS = ["momentum", "lgb_excess", "lgb_rank", "lgb_sector", "ridge"]
COMPARISONS = ["lgb_excess", "ridge"]      # run next to the submitted strategy in the final test
STRATEGY_GRID = [dict(max_rank=r, top_n=n, min_mom_rank=s) for r in (1000, 2000) for n in (25, 50) for s in (None, 0.5)]

# What `dev` decided (it asserts both), used by `test`:
#   FEATURES - the 12 features kept by the 2010-2012 low-correlation screen (inputs to LightGBM and Ridge);
#   FINAL    - momentum with its best 2013-2015 validation-Sharpe setup. LightGBM and Ridge trade under the
#              same rules; the median-momentum screen was in LightGBM's best setup and never binds for momentum.
FEATURES = ["idio_vol_252", "earnings_yield", "debt_to_assets", "sales_to_price", "gross_profitability",
            "earnings_change", "mom_12_1", "volume_trend", "accruals", "asset_growth", "rev_1m", "beta_252"]
FINAL = dict(model="momentum", comparisons=COMPARISONS, max_rank=2000, top_n=50, min_mom_rank=0.5)

RESULTS_CSV = RESULTS / "results.csv"
SECTIONS = ["test_performance", "test_year_by_year_return", "test_accounting_checks", "test_unquoted_open_lots",
            "test_trade_stats", "test_pnl_by_symbol", "test_ranking_quality", "test_model_refits",
            "test_lightgbm_feature_importance", "validation_backtests_2013_2015", "validation_rank_ic",
            "feature_screen"]


def save_section(section, frame):
    """Store one table in results/results.csv as long rows (section, name, metric, value), replacing that section."""
    assert section in SECTIONS, section
    long = (frame.rename_axis("name").reset_index().melt(id_vars="name", var_name="metric", value_name="value")
            .assign(section=section)[["section", "name", "metric", "value"]])
    long["value"] = [round(v, 6) if isinstance(v, float) else v for v in long.value]
    RESULTS.mkdir(exist_ok=True)
    old = pd.read_csv(RESULTS_CSV) if RESULTS_CSV.is_file() else long.iloc[:0]
    out = pd.concat([old.loc[old.section.ne(section)], long], ignore_index=True)
    out["_order"] = out.section.map(SECTIONS.index)
    out.sort_values("_order", kind="stable").drop(columns="_order").to_csv(RESULTS_CSV, index=False)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_everything(db):
    md = data.load_market_data(db)
    cache = OUTPUTS / "panel.parquet"
    if cache.is_file():
        panel = pd.read_parquet(cache)
    else:
        events, membership = data.load_financial_events(db), data.load_membership(db)
        dates = F.feature_dates(md.calendar, "2010-01-01", TEST_END)
        panel = F.make_labels(F.build_panel(md, events, membership, dates), md)
        panel = F.add_cross_sectional_ranks(panel, F.CANDIDATES)
        OUTPUTS.mkdir(exist_ok=True)
        panel.to_parquet(cache)
    return md, panel


def signal_dates(md, start, end):
    cal = md.calendar
    a = cal.get_loc(pd.Timestamp("2016-03-31"))
    idx = np.arange(a % SIGNAL_EVERY, len(cal) - 1, SIGNAL_EVERY)
    d = cal[idx]
    return d[(d >= start) & (d <= end)]


def score(name, feats, sectors, train, rows, cache=None):
    """Scores for ``rows`` from a model fit on ``train`` (or a fixed rule). Returns (scores, fitted model)."""
    cache = {} if cache is None else cache
    if name in cache:
        return cache[name]
    if name in RULES:
        out = RULES[name](rows), None
    else:
        m = FITTED[name](feats, sectors).fit(train)
        out = m.predict(rows), m
    cache[name] = out
    return out


def run_strategy(md, preds, sig, start, end, **kw):
    strat = TopNStrategy(preds, sig, **kw)
    bt = Backtester(md.calendar, md.symbols, md.adj_close, fee_bps=FEE_BPS)
    return bt.run(strat, start, end), strat


# ------------------------------------------------------------------------------------- dev
def validate(md, dev, feats, sectors, names, grid):
    """Walk-forward predictions for 2013-2015 (fit before each year), then the strategy grid backtests."""
    log("walk-forward model validation (fit before each validation year)")
    folds = M.yearly_folds(dev, VALIDATION_YEARS)
    rows, val_preds = [], {k: [] for k in names}
    for y, fit_date, train, valid in folds:
        all_rows = dev.loc[dev.date.dt.year.eq(y)]          # every row, including censored labels
        cache = {}
        for name in names:
            s, m = score(name, feats, sectors, train, all_rows, cache)
            val_preds[name].append(all_rows.assign(score=s))
            done = all_rows.assign(score=s).loc[all_rows.target.notna()]
            r = M.prediction_metrics(done.loc[done.score.notna()], top_n=50)
            rows.append({"year": y, "fit_date": fit_date.date(), "train_rows": len(train), "model": name, **r})
        log(f"  {y}: fit {fit_date.date()} on {len(train):,} rows (last label end {train.label_end.max().date()})")
    mv = pd.DataFrame(rows)

    log("strategy validation backtests 2013-2015")
    sig = signal_dates(md, folds[0][1], DEV_END)
    start = md.calendar[md.calendar.get_loc(sig[0]) + 1]
    bench = performance(BuyAndHold.run(md.benchmark, start, DEV_END).nav)
    srows = []
    for name in names:
        preds = pd.concat(val_preds[name])
        for cfg in grid:
            if name == "momentum" and cfg["min_mom_rank"] is not None:
                continue                                    # identical to the unscreened run
            res, _ = run_strategy(md, preds, sig, start, DEV_END, **cfg)
            p = performance(res.daily.nav)
            srows.append({"model": name, **cfg, **p.drop(["start", "end"]).to_dict(),
                          "excess_CAGR_vs_SP500TR": p.CAGR - bench.CAGR,
                          "avg_cash_weight": float((res.daily.cash / res.daily.nav).mean())})
        log(f"  backtests done: {name}")
    sv = pd.DataFrame(srows)
    sv.loc[len(sv)] = {"model": "^SP500TR buy&hold", **bench.drop(["start", "end"]).to_dict(), "excess_CAGR_vs_SP500TR": 0}
    return mv, sv


def stage_dev(db):
    RESULTS.mkdir(exist_ok=True)
    md, panel = load_everything(db)
    dev = panel.loc[panel.date.le(DEV_END)]
    sectors = sorted(panel.sector.unique())

    log("feature screen on 2010-2012 matured labels only")
    screen = dev.loc[dev.date.lt(SCREEN_END) & dev.label_end.lt(SCREEN_END) & dev.target_rank.notna()]
    kept, corr, stats = F.select_low_correlation_features(screen, [f"r_{c}" for c in F.CANDIDATES], CORR_CUTOFF)
    feats = [k[2:] for k in kept]
    stats.index = stats.index.str[2:]
    stats["description"] = [{**F.PRICE_CANDIDATES, **F.FUND_CANDIDATES}[c] for c in stats.index]
    stats["kept"] = stats.index.isin(feats)
    save_section("feature_screen", stats)
    corr.index, corr.columns = corr.index.str[2:], corr.columns.str[2:]
    plot_corr(corr.loc[feats, feats], RESULTS / "feature_correlation_selected.png")
    log(f"kept {len(feats)} features: {feats}")

    mv, sv = validate(md, dev, feats, sectors, MODELS, STRATEGY_GRID)
    ic = mv.pivot(index="model", columns="year", values="mean_rank_IC")
    save_section("validation_rank_ic", ic.assign(mean=ic.mean(axis=1)))
    sv["name"] = [m if pd.isna(r) else f"{m} | top {int(r)} | {int(n)} names | screen {'off' if pd.isna(x) else 'on'}"
                  for m, r, n, x in zip(sv.model, sv.max_rank, sv.top_n, sv.min_mom_rank)]
    save_section("validation_backtests_2013_2015",
                 sv.set_index("name")[["final_value", "CAGR", "ann_vol", "sharpe_rf2", "max_drawdown", "excess_CAGR_vs_SP500TR"]])
    print(sv.round(4).to_string())

    mom = sv.loc[sv.model.eq("momentum")].sort_values(["sharpe_rf2", "CAGR"], ascending=False).iloc[0]
    assert feats == FEATURES, f"feature screen changed: {feats}"
    assert (int(mom.max_rank), int(mom.top_n)) == (FINAL["max_rank"], FINAL["top_n"]), "momentum's best setup changed"
    log(f"kept features and final setup confirmed: {FINAL}")


def plot_corr(corr, path):
    fig, ax = plt.subplots(figsize=(8, 6.5))
    im = ax.imshow(corr.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(corr)), corr.index, fontsize=8)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iat[i, j]:.2f}", ha="center", va="center", fontsize=6.5)
    fig.colorbar(im, ax=ax, shrink=0.8, label="mean same-date Spearman correlation")
    ax.set_title("Selected features: average cross-sectional rank correlation, 2010-2012")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ------------------------------------------------------------------------------------ test
def stage_test(db):
    res_dir, out_dir, cfg = RESULTS, OUTPUTS, FINAL
    log(f"final setup: {cfg}")
    md, panel = load_everything(db)
    sectors = sorted(panel.sector.unique())
    feats = FEATURES
    sig = signal_dates(md, TEST_START - pd.Timedelta(days=1), TEST_END)
    assert sig[0] == pd.Timestamp("2016-03-31")

    # Annual refits: first signal date on/after March 31 of each year.
    fit_dates = sorted({sig[sig >= pd.Timestamp(f"{y}-03-31")][0] for y in range(2016, 2027)
                        if (sig >= pd.Timestamp(f"{y}-03-31")).any()})
    # The submitted strategy plus the comparison models, all under the same trading rules.
    names = [cfg["model"]] + cfg["comparisons"]
    preds, fit_log, importances = {n: [] for n in names}, [], []
    for k, fd in enumerate(fit_dates):
        nxt = fit_dates[k + 1] if k + 1 < len(fit_dates) else TEST_END + pd.Timedelta(days=1)
        train = M.training_rows(panel, fd)
        assert train.label_end.max() < fd, "training label not yet observed"
        rows = panel.loc[panel.date.isin(sig[(sig >= fd) & (sig < nxt)])]
        cache = {}
        for n in names:
            s, m = score(n, feats, sectors, train, rows, cache)
            preds[n].append(rows.assign(score=s))
        if "lgb_excess" in cache:
            importances.append(cache["lgb_excess"][1].importance().rename(fd.year))
        fit_log.append({"fit_date": fd.date(), "train_rows": len(train), "last_label_end": train.label_end.max().date(),
                        "predict_from": rows.date.min().date(), "predict_to": rows.date.max().date()})
        log(f"fit {fd.date()}: {len(train):,} rows, last label end {train.label_end.max().date()}")
    preds = {n: pd.concat(v) for n, v in preds.items()}
    save_section("test_model_refits", pd.DataFrame(fit_log).set_index("fit_date"))
    save_section("test_lightgbm_feature_importance", pd.concat(importances, axis=1).rename(columns=str))

    kw = dict(max_rank=cfg["max_rank"], top_n=cfg["top_n"], min_mom_rank=cfg["min_mom_rank"])
    runs = {n: run_strategy(md, preds[n], sig, TEST_START, TEST_END, **kw) for n in names}
    res, strat = runs[cfg["model"]]
    bench = BuyAndHold.run(md.benchmark, TEST_START, TEST_END)
    bp = performance(bench.nav)
    table = pd.DataFrame({f"Strategy ({cfg['model']}, submitted)": performance(res.daily.nav), "^SP500TR buy & hold": bp})
    for n in names[1:]:
        table[f"comparison: {n}"] = performance(runs[n][0].daily.nav)
    table.loc["excess_CAGR_vs_SP500TR"] = [table.at["CAGR", c] - bp.CAGR for c in table.columns]
    save_section("test_performance", table.T)
    print(table.to_string())
    comps = {n: runs[n][0].daily.nav for n in names[1:]}

    # Ranking quality of each score on the test signals with completed labels.
    pm = pd.DataFrame({n: M.prediction_metrics(p.loc[p.target.notna()], top_n=cfg["top_n"]) for n, p in preds.items()})
    save_section("test_ranking_quality", pm.T)
    label = f"12-1 momentum, top {cfg['top_n']} every 2 weeks, 90-day lots (submitted)"
    plot_curves(res.daily.nav, bench.nav, res_dir / "test_cumulative_returns.png", comps, label)

    out_dir.mkdir(parents=True, exist_ok=True)
    res.daily.to_csv(out_dir / "test_daily.csv")
    res.fills.to_csv(out_dir / "test_fills.csv", index=False)
    res.open_lots.to_csv(out_dir / "test_open_lots.csv", index=False)
    pd.DataFrame(strat.log).to_csv(out_dir / "test_plans.csv", index=False)
    checks = accounting_checks(res, md, db, sig)
    save_accounting_checks(checks)
    diagnostics(res, {"momentum": res.daily.nav, "^SP500TR": bench.nav, **comps})
    log("done")


def plot_curves(nav, bench, path, comparisons=None, label="Strategy"):
    pretty = {"lgb_excess": "LightGBM", "ridge": "Ridge"}
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for (n, b), c in zip((comparisons or {}).items(), ["#7f8c8d", "#b9a14a"]):
        ax.plot(b.index, b / 1e6, label=f"{pretty.get(n, n)} (same trading rules)", lw=1.0, color=c, ls="--")
    ax.plot(nav.index, nav / 1e6, label=label, lw=1.8, color="#1f5aa6")
    ax.plot(bench.index, bench / 1e6, label="^SP500TR buy & hold", lw=1.8, color="#c0392b")
    ax.set_ylabel(r"Portfolio value (\$ millions, from \$1M)")
    ax.set_title("Final test: Apr 1, 2016 - Sep 30, 2026")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def accounting_checks(res, md, db, sig):
    """Independent re-derivations of the backtest's cash, NAV, and timing."""
    import duckdb

    f, d = res.fills, res.daily
    buys, sells = f.loc[f.side.eq("BUY")], f.loc[f.side.eq("SELL")]
    cal = md.calendar
    out = {}
    # Cash: initial + every fill's cash change == ending cash.
    out["cash_reconciles"] = bool(np.isclose(res.initial_cash + f.cash_change.sum(), d.cash.iloc[-1]))
    out["min_cash"] = float(d.cash.min())
    # NAV: initial + realized PnL + unrealized PnL on open lots (net of their entry fees) == final NAV.
    ol = res.open_lots
    unreal = float((ol.market_value - ol.units * ol.entry_price - ol.entry_fee).sum()) if len(ol) else 0.0
    out["nav_reconciles"] = bool(np.isclose(res.initial_cash + sells.realized_pnl.sum() + unreal, d.nav.iloc[-1]))
    out["nav_equals_cash_plus_holdings"] = bool(np.allclose(d.nav, d.cash + d.holdings))
    # Timing: each buy fills at the session immediately after its signal close.
    nxt = cal[cal.get_indexer(buys.signal_date) + 1]
    out["buys_fill_next_session"] = bool((pd.DatetimeIndex(buys.date) == nxt).all())
    out["all_signal_dates_on_grid"] = bool(buys.signal_date.isin(sig).all())
    # Exits: never before maturity, and at the first session with a quote on/after it.
    out["no_sale_before_maturity"] = bool((sells.date >= sells.due_date).all())
    cols = md.symbols.get_indexer(sells.symbol)
    first_q = []
    for due, c in zip(sells.due_date, cols):
        j = cal.searchsorted(due)
        while not np.isfinite(md.adj_close[j, c]):
            j += 1
        first_q.append(cal[j])
    out["sales_at_first_quote_on_or_after_maturity"] = bool((pd.DatetimeIndex(first_q) == pd.DatetimeIndex(sells.date)).all())
    out["sessions_late_distribution"] = sells.sessions_late.value_counts().sort_index().to_dict()
    # Fill prices equal the database's adjusted close (random sample, queried directly).
    sample = f.sample(min(25, len(f)), random_state=1)
    with duckdb.connect(str(db), read_only=True) as con:
        q = con.execute("SELECT symbol, date, adj_close FROM prices WHERE (symbol, date) IN (SELECT unnest(?), unnest(?))",
                        [sample.symbol.tolist(), [x.date() for x in sample.date]]).fetchdf()
    q["date"] = pd.to_datetime(q.date)
    m = sample.merge(q, on=["symbol", "date"], how="left")
    out["sampled_fill_prices_match_db"] = bool(np.allclose(m.price, m.adj_close))
    out["fills"] = {"buys": int(len(buys)), "sells": int(len(sells)), "open_lots_at_end": int(len(ol))}
    out["fees_paid"] = float(f.fee.sum())
    out["skipped_orders_no_entry_quote"] = int(len(res.skipped_orders))
    out["incidents"] = res.incidents.to_dict("records") if len(res.incidents) else []
    # Sensitivity: value of open lots whose stock stopped quoting before the end.
    if len(ol):
        stuck = ol.loc[ol.frozen | ol.last_quote.lt(cal[-1])]
        out["open_lots_without_final_quote"] = stuck[["symbol", "last_quote", "market_value"]].to_dict("records")
        out["final_nav_if_unquoted_lots_written_to_zero"] = float(d.nav.iloc[-1] - stuck.market_value.sum())
    # One worked example, end to end.
    ex = sells.iloc[0]
    b = buys.loc[buys.lot_id.eq(ex.lot_id)].iloc[0]
    out["worked_example"] = {
        "symbol": ex.symbol, "signal": str(b.signal_date.date()), "entry": str(b.date.date()), "due": str(ex.due_date.date()),
        "exit": str(ex.date.date()), "entry_price": b.price, "units": b.units, "entry_cost_incl_fee": -b.cash_change,
        "exit_price": ex.price, "exit_proceeds_after_fee": ex.cash_change, "realized_pnl": ex.realized_pnl,
        "check_pnl": b.units * ex.price * (1 - res.fee_rate) + b.cash_change}
    print(json.dumps({k: v for k, v in out.items() if k not in ("incidents", "open_lots_without_final_quote")},
                     indent=2, default=str))
    return out


def save_accounting_checks(checks):
    """Flatten the accounting checks into results.csv: scalar checks, then the unquoted open lots by symbol."""
    flat = {}
    for k, v in checks.items():
        if isinstance(v, dict):
            flat.update({f"{k}.{kk}": vv for kk, vv in v.items()})
        elif not isinstance(v, list):
            flat[k] = v
    stuck = pd.DataFrame(checks.get("open_lots_without_final_quote", []), columns=["symbol", "last_quote", "market_value"])
    flat["open_lots_without_final_quote.count"] = len(stuck)
    flat["open_lots_without_final_quote.value"] = float(stuck.market_value.sum())
    flat["price_incidents"] = len(checks.get("incidents", []))
    save_section("test_accounting_checks", pd.DataFrame({"result": pd.Series(flat, dtype=object)}))
    by_sym = stuck.groupby("symbol").agg(lots=("market_value", "size"), market_value=("market_value", "sum"),
                                         last_quote=("last_quote", "max")).sort_values("market_value", ascending=False)
    by_sym["last_quote"] = pd.to_datetime(by_sym.last_quote).dt.date.astype(str)
    save_section("test_unquoted_open_lots", by_sym)


def diagnostics(res, navs):
    f = res.fills
    sells = f.loc[f.side.eq("SELL")].copy()
    sells["ret"] = sells.price / sells.merge(f.loc[f.side.eq("BUY"), ["lot_id", "price"]], on="lot_id",
                                             suffixes=("", "_b")).price_b.to_numpy() - 1
    top = sells.groupby("symbol").realized_pnl.sum().sort_values()
    save_section("test_pnl_by_symbol", pd.concat([top.head(10), top.tail(10)]).to_frame("realized_pnl"))
    save_section("test_trade_stats", pd.DataFrame({
        "lot_hit_rate": [float((sells.ret > 0).mean())], "mean_lot_return": [float(sells.ret.mean())],
        "median_lot_return": [float(sells.ret.median())], "avg_names_held": [float(res.daily.n_names.mean())],
        "avg_cash_weight": [float((res.daily.cash / res.daily.nav).mean())]}, index=["momentum"]))
    yearly = {}
    for n, nav in navs.items():
        ye = nav.resample("YE").last()
        r = ye.pct_change()
        r.iloc[0] = ye.iloc[0] / 1e6 - 1          # every portfolio starts from $1M
        yearly[n] = r.set_axis(ye.index.year.astype(str))
    save_section("test_year_by_year_return", pd.DataFrame(yearly).T)


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["dev", "test"])
    ap.add_argument("--db", default=None, help="path to equities_top2000.duckdb")
    a = ap.parse_args()
    db = data.find_database(a.db)
    log(f"database: {db}")
    stage_dev(db) if a.stage == "dev" else stage_test(db)
