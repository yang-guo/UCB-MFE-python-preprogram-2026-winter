"""Experiment configuration, research runs, final runs, and summaries.

A config is a plain JSON-serializable dict (so Papermill can pass it and the
registry can hash it).  ``run_research`` can only load research-mode data;
``run_final`` requires a ``FinalAuthorization`` (see ``protocol``).
"""

from __future__ import annotations

import copy
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from . import audit as A  # noqa: E402
from . import features as F  # noqa: E402
from . import integrity as I  # noqa: E402
from . import metrics as M  # noqa: E402
from . import protocol  # noqa: E402
from .backtest import Backtester, OrderManagementSystem, frozen_writeoff_nav  # noqa: E402
from .data import DEFAULT_DB, load_market, load_reference  # noqa: E402
from .panel import PanelConfig, build_panel, feature_columns  # noqa: E402
from .portfolio import (COHORTS_PER_HOLDING, SCHEDULES, SIZERS, CohortBudget, CompositeStrategy,  # noqa: E402
                        NameCap, ScoreBook, SectorCap, TopN)
from .protocol import Mode  # noqa: E402
from .scoring import validation_folds, walk_forward_scores  # noqa: E402

OUTPUT_ROOT = protocol.SUBMISSION_ROOT / "outputs"
EXPERIMENT_ROOT = OUTPUT_ROOT / "experiments"
REGISTRY = EXPERIMENT_ROOT / "registry.csv"

DEFAULT_CONFIG = {
    "name": "baseline",
    "hypothesis": "",
    "features": ["ret_21", "ret_63", "mom_252_21", "vol_63", "drawdown_252"],
    "model": {"kind": "ridge", "params": {"alpha": 10.0}},
    "target": "target_rank",
    "train_start": "2010-01-01",
    "train_window_years": None,
    "refit": "quarterly",
    "validation_years": [2012, 2013, 2014, 2015],
    "panel": PanelConfig().to_dict(),
    "strategy": {"schedule": "weekly", "n": 30, "sizer": "equal", "name_cap": 0.05,
                 "sector_cap": 0.30, "fee_bps": 5.0, "min_order": 100.0},
    "dev_backtest": {"start": "2012-01-01"},
    "price_variant": "repaired",
    # Held stock stops quoting: "settle_at_maturity" (cash out at the last mark at maturity; Ian's
    # choice after the demonstration run showed "freeze" idles capital in acquired companies).
    "on_long_gap": "settle_at_maturity",
}
PRICE_VARIANTS = ("repaired", "as_delivered", "neutralize_all")
DELISTING_POLICIES = ("settle_at_maturity", "freeze")


def delisting_policy(cfg: dict) -> str:
    """Configs written before this key existed (the demonstration run) used 'freeze'."""
    return cfg.get("on_long_gap", "freeze")


def make_config(**overrides) -> dict:
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(cfg.get(k), dict):
            cfg[k] = {**cfg[k], **v}
        else:
            cfg[k] = v
    validate_config(cfg)
    return cfg


def validate_config(cfg: dict) -> None:
    unknown = set(cfg) - set(DEFAULT_CONFIG)
    if unknown:
        raise KeyError(f"Unknown config keys {sorted(unknown)}")
    feature_columns(cfg["features"])  # raises on unknown features
    if cfg["price_variant"] not in PRICE_VARIANTS:
        raise ValueError(f"price_variant must be one of {PRICE_VARIANTS}")
    if delisting_policy(cfg) not in DELISTING_POLICIES:
        raise ValueError(f"on_long_gap must be one of {DELISTING_POLICIES}")
    if cfg["strategy"]["schedule"] not in SCHEDULES or cfg["strategy"]["sizer"] not in SIZERS:
        raise ValueError("Unknown schedule or sizer")
    json.dumps(cfg)  # must be serializable


def config_id(cfg: dict) -> str:
    return f"{cfg['name']}-{protocol.digest(cfg)[:8]}"


# ------------------------------------------------------------------ shared steps
def prepare(mode: Mode, auth=None, database=DEFAULT_DB, panel_cfg: dict | None = None, verbose=True):
    t0 = time.time()
    market = load_market(database, mode, auth)
    ref = load_reference(database, mode, auth)
    audit = I.audit_prices(market.adj, market.close, market.volume, market.calendar, market.symbols)
    end = protocol.last_signal_date(mode)
    panel = build_panel(market, ref, audit, end=end, config=PanelConfig(**(panel_cfg or {})))
    if verbose:
        print(f"[prepare:{Mode(mode).value}] {len(panel):,} panel rows, {panel.eligible.sum():,} eligible, "
              f"{time.time() - t0:.0f}s")
    return market, ref, audit, panel


def backtest_prices(market, audit, variant: str) -> np.ndarray:
    if variant == "repaired":
        return audit.clean
    if variant == "as_delivered":  # bad prints quarantined, nothing repaired
        return I.audit_prices(market.adj, market.close, market.volume, market.calendar, market.symbols,
                              repair_splits=False).clean
    return I.neutralized_prices(market.adj, market.calendar, market.symbols, audit)


def build_strategy(cfg: dict, scores: pd.DataFrame, universe: pd.DataFrame) -> CompositeStrategy:
    s = cfg["strategy"]
    sectors = universe.set_index("symbol").sector.fillna("Unknown").to_dict()
    return CompositeStrategy(
        ScoreBook(scores), SCHEDULES[s["schedule"]](), TopN(int(s["n"])), SIZERS[s["sizer"]](),
        CohortBudget(COHORTS_PER_HOLDING[s["schedule"]]),
        [NameCap(s["name_cap"]), SectorCap(s["sector_cap"], sectors)],
        holding_days=protocol.HOLDING_DAYS, min_order=s["min_order"])


def run_backtest(cfg, market, px, scores, universe, start, end, fee_bps=None, on_long_gap=None):
    on_long_gap = on_long_gap or delisting_policy(cfg)
    strategy = build_strategy(cfg, scores, universe)
    oms = OrderManagementSystem(protocol.INITIAL_CAPITAL, cfg["strategy"]["fee_bps"] if fee_bps is None else fee_bps)
    result = Backtester(market.calendar, market.symbols, px, strategy, oms, on_long_gap=on_long_gap).run(start, end)
    report = A.audit_backtest(result, market.calendar, market.symbols, px)
    A.assert_clean(report)
    return result, report, strategy


def nav_chart(navs: dict, path: Path, title: str):
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, nav in navs.items():
        (nav / nav.iloc[0] - 1).plot(ax=ax, label=name, lw=1.6 if "S&P" not in name else 1.2,
                                     ls="--" if "S&P" in name else "-")
    ax.set(title=title, ylabel="Cumulative return")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.axhline(0, color="black", lw=0.6)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ------------------------------------------------------------------ research
def run_research(cfg: dict, prepared=None, out_root: Path = EXPERIMENT_ROOT, write=True) -> dict:
    """Development-period experiment.  Cannot touch final-test data."""
    validate_config(cfg)
    market, ref, audit, panel = prepared or prepare(Mode.RESEARCH, panel_cfg=cfg["panel"])
    if market.mode is not Mode.RESEARCH:
        raise protocol.HoldoutViolation("run_research received final-mode data")
    X = feature_columns(cfg["features"])
    model = cfg["model"]

    # 1) fixed-fold validation: fit once at each validation year's first signal
    from sklearn.base import clone
    from .models import build_model
    fold_rows, pred_parts = [], []
    for fold in validation_folds(panel, cfg["validation_years"], cfg["train_start"], cfg["target"],
                                 cfg["train_window_years"]):
        tr, va = panel.iloc[fold["train"]], panel.iloc[fold["valid"]]
        est = clone(build_model(model["kind"], model["params"])).fit(tr[X].to_numpy(float), tr[cfg["target"]].to_numpy(float))
        pred = va[["date", "symbol", "target"]].assign(score=est.predict(va[X].to_numpy(float)))
        pred_parts.append(pred.assign(year=fold["year"]))
        fold_rows.append({"year": fold["year"], "fit_date": fold["fit_date"].date(), "train_rows": len(tr),
                          "train_max_label_end": tr.label_end.max().date(), "valid_rows": len(va),
                          **M.prediction_metrics(pred).to_dict()})
    folds = pd.DataFrame(fold_rows)

    # 2) development backtest with the same walk-forward schedule the final test will use
    dev_start = pd.Timestamp(cfg["dev_backtest"]["start"])
    signal0 = market.calendar[market.calendar.searchsorted(dev_start) - 1]
    scores, fit_audit = walk_forward_scores(panel, X, model["kind"], model["params"], cfg["target"],
                                            signal0, protocol.DEVELOPMENT_END, cfg["train_start"], cfg["refit"],
                                            cfg["train_window_years"], Mode.RESEARCH)
    px = backtest_prices(market, audit, cfg["price_variant"])
    result, report, strategy = run_backtest(cfg, market, px, scores, ref.universe, dev_start, protocol.DEVELOPMENT_END)
    bench = M.benchmark_nav(market.benchmark, result.history.index, result.history.index[1], protocol.INITIAL_CAPITAL)
    perf = M.comparison_table(result.history.nav, bench, protocol.ANNUAL_RF)
    rel = M.relative(result.history.nav, bench)
    diag = M.portfolio_diagnostics(result)
    wf_pred = scores.merge(panel[["date", "symbol", "target"]], on=["date", "symbol"])
    wf_metrics = M.prediction_metrics(wf_pred)

    summary = {
        "experiment_id": config_id(cfg), "name": cfg["name"], "hypothesis": cfg["hypothesis"],
        "mode": "research", "holdout_touched": False,
        "data_window": f"{market.calendar[0].date()} .. {market.end.date()} (research cap)",
        "training": f"signals >= {cfg['train_start']}, label_end < fit date, refit {cfg['refit']}",
        "validation_years": cfg["validation_years"],
        "dev_backtest": f"{result.history.index[1].date()} .. {result.history.index[-1].date()}",
        "features": cfg["features"], "model": model, "target": cfg["target"], "strategy": cfg["strategy"],
        "price_variant": cfg["price_variant"], "delisting_policy": delisting_policy(cfg),
        "validation_mean_IC": float(folds.mean_rank_IC.mean()),
        "validation_IC_by_year": dict(zip(folds.year.astype(str), folds.mean_rank_IC.round(4))),
        "validation_years_positive_IC": int((folds.mean_rank_IC > 0).sum()),
        "walk_forward_dev_IC": float(wf_metrics.mean_rank_IC),
        "dev_CAGR": float(perf.loc["CAGR", "strategy"]), "dev_benchmark_CAGR": float(perf.loc["CAGR", "S&P 500 TR"]),
        "dev_Sharpe": float(perf.loc["Sharpe", "strategy"]), "dev_max_drawdown": float(perf.loc["max_drawdown", "strategy"]),
        "dev_final_value": float(perf.loc["final_value", "strategy"]),
        "audit_checks_passed": f"{int(report.passed.sum())}/{len(report)}",
        "warnings": _warnings(diag, result),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    fit_table = fit_audit.drop(columns="_model")
    if all(hasattr(m, "alpha_") for m in fit_audit["_model"]):  # penalty chosen inside each fit's training rows
        fit_table = fit_table.assign(chosen_alpha=[m.alpha_ for m in fit_audit["_model"]])
    artifacts = {"summary": summary, "folds": folds, "performance": perf, "relative": rel, "diagnostics": diag,
                 "audit": report, "fit_audit": fit_table, "result": result,
                 "walk_forward_ic": M.ic_series(wf_pred),
                 "benchmark_nav": bench, "scores": scores, "walk_forward_metrics": wf_metrics,
                 "decisions": pd.DataFrame(strategy.decisions), "fit_models": fit_audit}
    if write:
        out = Path(out_root) / summary["experiment_id"]
        write_artifacts(out, cfg, artifacts)
        summary["output_dir"] = str(out.relative_to(protocol.SUBMISSION_ROOT))
        append_registry(summary, out_root)
    return artifacts


def _warnings(diag, result) -> list[str]:
    w = []
    if diag.frozen_lots_at_end:
        w.append(f"{int(diag.frozen_lots_at_end)} lots frozen at last mark at the end")
    if diag.rejected_orders:
        w.append(f"{int(diag.rejected_orders)} orders rejected for missing quotes")
    if diag.average_cash_fraction > 0.2:
        w.append(f"average cash {diag.average_cash_fraction:.0%} of NAV")
    return w


def write_artifacts(out: Path, cfg: dict, art: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    (out / "summary.json").write_text(json.dumps(art["summary"], indent=2, default=str))
    art["folds"].to_csv(out / "validation_folds.csv", index=False)
    art["performance"].to_csv(out / "dev_performance.csv")
    art["diagnostics"].to_csv(out / "dev_diagnostics.csv", header=["value"])
    art["audit"].to_csv(out / "accounting_audit.csv", index=False)
    art["fit_audit"].to_csv(out / "fit_audit.csv", index=False)
    art["result"].history.to_csv(out / "dev_nav.csv")
    if "walk_forward_ic" in art:
        art["walk_forward_ic"].to_csv(out / "walk_forward_ic.csv")
    nav_chart({"strategy": art["result"].history.nav, "S&P 500 TR": art["benchmark_nav"]},
              out / "dev_cumulative_return.png", f"{art['summary']['experiment_id']} (development)")
    s = art["summary"]
    (out / "summary.md").write_text("\n".join([
        f"# {s['experiment_id']}", "", f"**Hypothesis:** {s['hypothesis'] or '-'}", "",
        f"- Mode: research (holdout untouched); data {s['data_window']}",
        f"- Features: {', '.join(s['features'])}", f"- Model: {s['model']}; target {s['target']}",
        f"- Training: {s['training']}", f"- Strategy: {s['strategy']}", f"- Price variant: {s['price_variant']}",
        f"- Validation mean rank IC: {s['validation_mean_IC']:.4f} by year {s['validation_IC_by_year']}",
        f"- Dev backtest {s['dev_backtest']}: CAGR {s['dev_CAGR']:.2%} vs S&P {s['dev_benchmark_CAGR']:.2%}, "
        f"Sharpe {s['dev_Sharpe']:.2f}, MaxDD {s['dev_max_drawdown']:.1%}",
        f"- Accounting audit: {s['audit_checks_passed']} checks passed",
        f"- Warnings: {s['warnings'] or 'none'}", ""]))


def append_registry(summary: dict, out_root: Path = EXPERIMENT_ROOT) -> None:
    keep = ["experiment_id", "name", "hypothesis", "model", "features", "target", "strategy", "price_variant",
            "validation_mean_IC", "validation_years_positive_IC", "walk_forward_dev_IC", "dev_CAGR",
            "dev_benchmark_CAGR", "dev_Sharpe", "dev_max_drawdown", "audit_checks_passed", "warnings",
            "output_dir", "created_utc"]
    row = pd.DataFrame([{k: (json.dumps(summary.get(k)) if isinstance(summary.get(k), (dict, list))
                             else summary.get(k)) for k in keep}])
    path = Path(out_root) / "registry.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    row.to_csv(path, mode="a", header=not path.is_file(), index=False)


# ------------------------------------------------------------------ final (holdout)
RESULTS_DIR = protocol.SUBMISSION_ROOT / "results"


def run_final(cfg: dict, auth, results_dir: Path = RESULTS_DIR, sensitivities=True) -> dict:
    """Evaluate the frozen strategy on Apr 2016 - Sep 2026.  Requires FinalAuthorization."""
    protocol.require_final(auth)
    if protocol.digest(cfg) != auth.config_sha256:
        raise protocol.HoldoutViolation("Config differs from the authorized frozen config")
    market, ref, audit, panel = prepare(Mode.FINAL, auth=auth, panel_cfg=cfg["panel"])
    X = feature_columns(cfg["features"])
    model = cfg["model"]
    first_signal = market.calendar[market.calendar.get_loc(protocol.TEST_START) - 1]  # 2016-03-31 close
    scores, fit_audit = walk_forward_scores(panel, X, model["kind"], model["params"], cfg["target"], first_signal,
                                            protocol.TEST_END, cfg["train_start"], cfg["refit"],
                                            cfg["train_window_years"], Mode.FINAL)
    if not (fit_audit.train_max_label_end < fit_audit.fit_date).all():
        raise AssertionError("fit audit: immature training label")
    px = backtest_prices(market, audit, cfg["price_variant"])
    result, report, strategy = run_backtest(cfg, market, px, scores, ref.universe, protocol.TEST_START, protocol.TEST_END)
    h = result.history
    bench = M.benchmark_nav(market.benchmark, h.index, protocol.TEST_START, protocol.INITIAL_CAPITAL)
    perf = M.comparison_table(h.nav, bench, protocol.ANNUAL_RF)
    rel = M.relative(h.nav, bench)
    diag = M.portfolio_diagnostics(result)

    # prediction quality on completed test labels (open positions still count in NAV)
    test_pred = scores.loc[scores.date >= protocol.TEST_START].merge(
        panel[["date", "symbol", "target", "label_end"]], on=["date", "symbol"])
    pred_all = M.prediction_metrics(test_pred)
    pred_year = test_pred.assign(year=test_pred.date.dt.year).groupby("year").apply(
        lambda g: M.prediction_metrics(g), include_groups=False)

    # integrity incidents that touched held lots (where data repairs could move PnL)
    lots = pd.concat([result.lots_closed.assign(state="closed"),
                      result.open_lots.assign(state="open", exit_date=pd.NaT)], ignore_index=True)
    ev = audit.events.loc[audit.events.date >= protocol.TEST_START]
    touched = ev.merge(lots[["symbol", "lot_id", "entry_date", "exit_date", "units", "state"]], on="symbol")
    touched = touched.loc[(touched.date > touched.entry_date)
                          & (touched.exit_date.isna() | (touched.date <= touched.exit_date))]

    sens = {}
    if sensitivities:
        def variant(name, **kw):
            c = {**cfg, **kw.pop("cfg", {})}
            p = backtest_prices(market, audit, c["price_variant"])
            r, rep, _ = run_backtest(c, market, p, scores, ref.universe, protocol.TEST_START, protocol.TEST_END, **kw)
            sens[name] = M.performance(r.history.nav, protocol.ANNUAL_RF)
            return r
        policy = delisting_policy(cfg)
        other = "freeze" if policy == "settle_at_maturity" else "settle_at_maturity"
        sens[f"primary (repaired prices, {policy})"] = perf["strategy"]
        sens["frozen lots written to zero (lower bound)"] = M.performance(frozen_writeoff_nav(result), protocol.ANNUAL_RF)
        variant("prices as delivered (no split/level repair)", cfg={"price_variant": "as_delivered"})
        variant("all breaks neutralized", cfg={"price_variant": "neutralize_all"})
        alt = variant(f"delisting policy: {other}", on_long_gap=other)
        alt.history.to_csv(OUTPUT_ROOT / "final_alt_delisting_nav.csv")
        variant("fees 0 bps", fee_bps=0.0)
        variant("fees 25 bps", fee_bps=25.0)
    sensitivity = pd.DataFrame(sens).T

    beat = perf.loc["final_value", "strategy"] - perf.loc["final_value", "S&P 500 TR"]
    summary = {
        "experiment_id": config_id(cfg), "mode": "final", "config_sha256": auth.config_sha256,
        "authorized_at": auth.authorized_at, "prior_final_runs": auth.prior_final_runs,
        "reselection_acknowledged": auth.reselection_acknowledged,
        "test_window": f"{h.index[1].date()} .. {h.index[-1].date()}",
        "delisting_policy": delisting_policy(cfg),
        "final_value": float(perf.loc["final_value", "strategy"]),
        "benchmark_final_value": float(perf.loc["final_value", "S&P 500 TR"]),
        "excess_final_value": float(beat), "beat_benchmark": bool(beat > 0),
        "CAGR": float(perf.loc["CAGR", "strategy"]), "benchmark_CAGR": float(perf.loc["CAGR", "S&P 500 TR"]),
        "Sharpe": float(perf.loc["Sharpe", "strategy"]), "benchmark_Sharpe": float(perf.loc["Sharpe", "S&P 500 TR"]),
        "max_drawdown": float(perf.loc["max_drawdown", "strategy"]),
        "benchmark_max_drawdown": float(perf.loc["max_drawdown", "S&P 500 TR"]),
        "test_mean_rank_IC": float(pred_all.mean_rank_IC),
        "audit_checks_passed": f"{int(report.passed.sum())}/{len(report)}",
        "integrity_events_in_test": audit.events.loc[audit.events.date >= protocol.TEST_START].kind.value_counts().to_dict(),
        "integrity_events_touching_held_lots": touched.kind.value_counts().to_dict(),
        "warnings": _warnings(diag, result),
        "data": market.provenance,
    }
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    perf.to_csv(results_dir / "performance_vs_benchmark.csv")
    rel.to_csv(results_dir / "relative_to_benchmark.csv", header=["value"])
    diag.to_csv(results_dir / "portfolio_diagnostics.csv", header=["value"])
    report.to_csv(results_dir / "accounting_audit.csv", index=False)
    sensitivity.to_csv(results_dir / "sensitivity.csv")
    fit_audit.drop(columns="_model").to_csv(results_dir / "fit_audit.csv", index=False)
    pred_year.to_csv(results_dir / "test_prediction_metrics_by_year.csv")
    touched.to_csv(results_dir / "integrity_events_touching_held_lots.csv", index=False)
    (results_dir / "final_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    nav_chart({"strategy": h.nav, "S&P 500 TR": bench}, results_dir / "cumulative_return.png",
              "Final test: frozen strategy vs $1M in ^SP500TR (Apr 2016 - Sep 2026)")
    # large per-day/per-trade tables go to the git-ignored outputs folder
    big = OUTPUT_ROOT / "final"
    big.mkdir(parents=True, exist_ok=True)
    h.assign(benchmark_nav=bench).to_csv(big / "daily_nav.csv")
    result.fills.to_csv(big / "fills.csv", index=False)
    result.lots_closed.to_csv(big / "lots_closed.csv", index=False)
    result.open_lots.to_csv(big / "open_lots_at_end.csv", index=False)
    result.events.to_csv(big / "backtest_events.csv", index=False)
    scores.to_csv(big / "scores.csv", index=False)
    audit.events.to_csv(big / "integrity_events.csv", index=False)
    return {"summary": summary, "performance": perf, "relative": rel, "diagnostics": diag, "audit": report,
            "sensitivity": sensitivity, "fit_audit": fit_audit, "prediction": pred_all, "prediction_by_year": pred_year,
            "touched": touched, "result": result, "benchmark_nav": bench, "integrity": audit, "panel": panel,
            "market": market, "scores": scores}
