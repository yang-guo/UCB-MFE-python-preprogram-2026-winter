"""Run the pre-registered research study end to end (development data only).

    python scripts/run_study.py experiments/research [--resume] [--random 20]

Implements experiments/research/PLAN.md:

0. random-selection portfolios (noise level; universe vs selection)
1. single-feature rank IC by year (descriptive; not used for selection)
2. alpha candidates, each as a Papermill notebook; sequential selection rule
3. portfolio variants of the selected alpha, each as a Papermill notebook; adoption rule

Writes results/research/ (small tables + STUDY_SUMMARY.md) and stops.  It
never freezes a strategy and cannot load final-test data: everything goes
through research mode.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from quantlab import features as F  # noqa: E402
from quantlab import metrics as M  # noqa: E402
from quantlab import protocol  # noqa: E402
from quantlab.experiment import (EXPERIMENT_ROOT, REGISTRY, backtest_prices, config_id, make_config,  # noqa: E402
                                 prepare, run_backtest)
from quantlab.protocol import Mode  # noqa: E402
from quantlab.report import md_table  # noqa: E402
from run_research import run_plan  # noqa: E402

OUT = ROOT / "results" / "research"
IC_YEARS = [2012, 2013, 2014, 2015]
BOOT_BLOCK, BOOT_DRAWS, BOOT_SEED = 13, 5000, 20261113


# ----------------------------------------------------------------- stage 0
def random_baselines(prepared, n_random: int) -> pd.DataFrame:
    """Top-30 equal-weight portfolios of uniformly random eligible names (same engine, same costs)."""
    market, ref, audit, panel = prepared
    cfg = make_config(name="random_baseline")
    dev_start = pd.Timestamp(cfg["dev_backtest"]["start"])
    signal0 = market.calendar[market.calendar.searchsorted(dev_start) - 1]
    rows = panel.loc[panel.eligible & panel.date.ge(signal0) & panel.date.le(protocol.DEVELOPMENT_END),
                     ["date", "symbol", "sector", "vol_63"]].copy()
    rows["fit_date"] = rows.date
    rows["train_max_label_end"] = pd.NaT
    px = backtest_prices(market, audit, cfg["price_variant"])
    out = []
    for seed in range(n_random):
        t0 = time.time()
        rows["score"] = np.random.default_rng(seed).random(len(rows))
        result, _, _ = run_backtest(cfg, market, px, rows, ref.universe, dev_start, protocol.DEVELOPMENT_END)
        bench = M.benchmark_nav(market.benchmark, result.history.index, result.history.index[1],
                                protocol.INITIAL_CAPITAL)
        perf = M.comparison_table(result.history.nav, bench, protocol.ANNUAL_RF)
        out.append({"seed": seed, **perf["strategy"].to_dict(),
                    "benchmark_CAGR": perf.loc["CAGR", "S&P 500 TR"],
                    "benchmark_max_drawdown": perf.loc["max_drawdown", "S&P 500 TR"],
                    "benchmark_Sharpe": perf.loc["Sharpe", "S&P 500 TR"]})
        print(f"  random seed {seed}: CAGR {out[-1]['CAGR']:.2%} ({time.time() - t0:.0f}s)", flush=True)
    return pd.DataFrame(out)


# ----------------------------------------------------------------- stage 1
def feature_study(panel: pd.DataFrame, signs: dict) -> pd.DataFrame:
    """Mean same-date rank IC of each raw feature vs the next 90-day return, by signal year."""
    dev = panel.loc[panel.eligible & panel.date.ge(protocol.DEVELOPMENT_START)
                    & panel.date.le(protocol.DEVELOPMENT_END) & panel.target.notna()]
    rows = []
    for name, spec in F.REGISTRY.items():
        if name == "availability_imputed":
            continue
        ic = M.ic_series(dev, score=name, target="target")
        by_year = ic.groupby(ic.index.year).mean()
        blocks = ic.groupby(np.arange(len(ic)) // 13).mean()
        rows.append({"feature": name, "group": spec.group, "plan_sign": signs.get(name, ""),
                     **{str(y): by_year.get(y, np.nan) for y in range(2010, 2016)},
                     "mean_IC": ic.mean(),
                     "tstat_13wk_blocks": blocks.mean() / blocks.std(ddof=1) * np.sqrt(len(blocks)),
                     "years_same_sign_as_mean": int((np.sign(by_year) == np.sign(ic.mean())).sum()),
                     "coverage": dev[name].notna().mean()})
    return pd.DataFrame(rows).sort_values("mean_IC", key=np.abs, ascending=False).reset_index(drop=True)


# ----------------------------------------------------------------- stage 2
def load_ic(out_dir: Path) -> pd.Series:
    ic = pd.read_csv(out_dir / "walk_forward_ic.csv", parse_dates=["date"], index_col="date").rank_IC
    return ic.loc[ic.index.year.isin(IC_YEARS)]


def block_bootstrap_mean(diff: np.ndarray, block=BOOT_BLOCK, draws=BOOT_DRAWS, seed=BOOT_SEED) -> np.ndarray:
    """Circular block bootstrap of the mean (overlapping 90-day outcomes make weekly ICs dependent)."""
    rng = np.random.default_rng(seed)
    n = len(diff)
    k = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=(draws, k))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]) % n
    return diff[idx.reshape(draws, -1)[:, :n]].mean(axis=1)


def select_alpha(plan: list[dict], dirs: dict) -> tuple[pd.DataFrame, str | None]:
    order = sorted(plan, key=lambda e: e["complexity"])
    ics = {e["name"]: load_ic(dirs[e["name"]]) for e in order}
    rows, incumbent = [], None
    for e in order:
        name, ic = e["name"], ics[e["name"]]
        by_year = ic.groupby(ic.index.year).mean()
        stable = int((by_year > 0).sum()) >= 3
        row = {"candidate": name, "complexity": e["complexity"], "mean_IC_2012_2015": ic.mean(),
               **{f"IC_{y}": by_year.get(y, np.nan) for y in IC_YEARS}, "years_positive": int((by_year > 0).sum()),
               "stable": stable, "compared_with": incumbent or "", "mean_IC_difference": np.nan,
               "bootstrap_p05_of_difference": np.nan, "bootstrap_share_below_zero": np.nan}
        if not stable:
            row["decision"] = "rejected: not stable"
        elif incumbent is None:
            row["decision"] = "becomes incumbent (first stable candidate)"
            incumbent = name
        else:
            both = pd.concat([ic, ics[incumbent]], axis=1, join="inner").to_numpy()
            diff = both[:, 0] - both[:, 1]
            boot = block_bootstrap_mean(diff)
            p05 = float(np.quantile(boot, 0.05))
            row.update(mean_IC_difference=diff.mean(), bootstrap_p05_of_difference=p05,
                       bootstrap_share_below_zero=float((boot <= 0).mean()))
            if p05 > 0:
                row["decision"] = f"replaces {incumbent}: improvement exceeds noise"
                incumbent = name
            else:
                row["decision"] = f"rejected: not better than {incumbent} beyond noise"
        rows.append(row)
    return pd.DataFrame(rows), incumbent


# ----------------------------------------------------------------- tables
def experiment_row(out_dir: Path) -> dict:
    s = json.loads((out_dir / "summary.json").read_text())
    perf = pd.read_csv(out_dir / "dev_performance.csv", index_col=0)
    diag = pd.read_csv(out_dir / "dev_diagnostics.csv", index_col=0)["value"]
    return {"experiment": s["name"], "validation_mean_IC": s["validation_mean_IC"],
            "walk_forward_IC": s["walk_forward_dev_IC"], "CAGR": s["dev_CAGR"],
            "benchmark_CAGR": s["dev_benchmark_CAGR"], "Sharpe": s["dev_Sharpe"],
            "max_drawdown": s["dev_max_drawdown"],
            "benchmark_max_drawdown": float(perf.loc["max_drawdown", "S&P 500 TR"]),
            "benchmark_Sharpe": float(perf.loc["Sharpe", "S&P 500 TR"]),
            "turnover_x_avg_nav": float(diag.get("turnover_x_avg_nav", np.nan)),
            "audit": s["audit_checks_passed"], "experiment_id": s["experiment_id"]}


def decompose(table: pd.DataFrame, rnd: pd.DataFrame) -> pd.DataFrame:
    t = table.copy()
    t["universe_effect"] = rnd.CAGR.mean() - t.benchmark_CAGR
    t["selection_effect"] = t.CAGR - rnd.CAGR.mean()
    t["selection_in_random_sd"] = t.selection_effect / rnd.CAGR.std(ddof=1)
    return t


def pending(plan, resume):
    """With --resume, skip plan entries whose experiment already finished."""
    todo, dirs = [], {}
    for e in plan:
        cfg = make_config(name=e["name"], hypothesis=e.get("hypothesis", ""), **e.get("overrides", {}))
        out = EXPERIMENT_ROOT / config_id(cfg)
        dirs[e["name"]] = out
        if not (resume and (out / "summary.json").is_file() and (out / "walk_forward_ic.csv").is_file()):
            todo.append(e)
    return todo, dirs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("plan_dir")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--random", type=int, default=20)
    args = ap.parse_args()
    plan_dir = Path(args.plan_dir)
    alpha_plan = json.loads((plan_dir / "plan_alpha.json").read_text())
    portfolio_plan = json.loads((plan_dir / "plan_portfolio.json").read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    prepared = prepare(Mode.RESEARCH)
    panel = prepared[3]

    print("== stage 0: random baselines", flush=True)
    rnd_path = OUT / "random_baselines.csv"
    if args.resume and rnd_path.is_file() and len(pd.read_csv(rnd_path)) == args.random:
        rnd = pd.read_csv(rnd_path)
    else:
        rnd = random_baselines(prepared, args.random)
        rnd.to_csv(rnd_path, index=False)

    print("== stage 1: feature study", flush=True)
    composite = max((e for e in alpha_plan if e["overrides"]["model"]["kind"] == "composite"),
                    key=lambda e: len(e["overrides"]["features"]))["overrides"]
    signs = {f: ("+" if s > 0 else "-") for f, s in zip(composite["features"], composite["model"]["params"]["signs"])}
    feats = feature_study(panel, signs)
    feats.to_csv(OUT / "feature_ic_by_year.csv", index=False)
    del prepared, panel  # free memory before the notebooks load their own copies

    print("== stage 2: alpha candidates", flush=True)
    todo, dirs = pending(alpha_plan, args.resume)
    run_plan(todo)
    selection, chosen = select_alpha(alpha_plan, dirs)
    selection.to_csv(OUT / "alpha_selection.csv", index=False)
    alpha_table = decompose(pd.DataFrame([experiment_row(dirs[e["name"]]) for e in alpha_plan]), rnd)
    alpha_table.to_csv(OUT / "alpha_candidates.csv", index=False)
    print(selection[["candidate", "mean_IC_2012_2015", "years_positive", "decision"]].to_string(index=False), flush=True)

    port_table, adoption, adopted = pd.DataFrame(), pd.DataFrame(), None
    if chosen:
        print(f"== stage 3: portfolio variants of {chosen}", flush=True)
        base = next(e for e in alpha_plan if e["name"] == chosen)
        variants = [{"name": f"{v['name']}__{chosen.split('_')[0]}", "hypothesis": v["hypothesis"],
                     "overrides": {**base["overrides"], **v["overrides"]}} for v in portfolio_plan]
        todo, pdirs = pending(variants, args.resume)
        run_plan(todo)
        base_row = experiment_row(dirs[chosen])
        port_table = decompose(pd.DataFrame([base_row] + [experiment_row(pdirs[v["name"]]) for v in variants]), rnd)
        port_table.to_csv(OUT / "portfolio_variants.csv", index=False)
        noise = rnd.CAGR.std(ddof=1)
        a = port_table.iloc[1:].copy()
        a["CAGR_minus_base"] = a.CAGR - base_row["CAGR"]
        a["random_CAGR_sd"] = noise
        a["beats_base_beyond_noise"] = a.CAGR_minus_base > noise
        a["drawdown_no_worse_than_benchmark"] = a.max_drawdown >= a.benchmark_max_drawdown
        a["drawdown_minus_base"] = a.max_drawdown - base_row["max_drawdown"]
        a["passes_rule"] = a.beats_base_beyond_noise & a.drawdown_no_worse_than_benchmark
        adoption = a[["experiment", "CAGR", "CAGR_minus_base", "random_CAGR_sd", "beats_base_beyond_noise",
                      "max_drawdown", "benchmark_max_drawdown", "drawdown_no_worse_than_benchmark",
                      "drawdown_minus_base", "Sharpe", "passes_rule"]]
        adoption.to_csv(OUT / "portfolio_adoption.csv", index=False)
        ok = a.loc[a.passes_rule].sort_values("CAGR", ascending=False)
        adopted = ok.experiment.iloc[0] if len(ok) else None

    if REGISTRY.is_file():
        pd.read_csv(REGISTRY).drop_duplicates("experiment_id", keep="last").to_csv(OUT / "registry.csv", index=False)

    pct = "{:.2%}".format
    show = ["experiment", "validation_mean_IC", "walk_forward_IC", "CAGR", "benchmark_CAGR", "Sharpe", "max_drawdown",
            "universe_effect", "selection_effect", "selection_in_random_sd", "turnover_x_avg_nav", "audit"]
    lines = [
        "# Research study summary (development data only; final test untouched)", "",
        f"Plan: `{plan_dir.as_posix()}/PLAN.md`.  Development backtest window 2012-01-03 to 2015-12-31; "
        "walk-forward IC on 2012 to 2015 signals.", "",
        "## Stage 0: random 30-name portfolios", "",
        f"- {len(rnd)} random portfolios: mean CAGR {pct(rnd.CAGR.mean())}, standard deviation "
        f"{pct(rnd.CAGR.std(ddof=1))}, range {pct(rnd.CAGR.min())} to {pct(rnd.CAGR.max())}.",
        f"- S&P 500 total return over the same window: CAGR {pct(rnd.benchmark_CAGR.iloc[0])}, "
        f"max drawdown {pct(rnd.benchmark_max_drawdown.iloc[0])}, Sharpe {rnd.benchmark_Sharpe.iloc[0]:.2f}.",
        f"- Random portfolios: mean Sharpe {rnd.Sharpe.mean():.2f}, mean max drawdown {pct(rnd.max_drawdown.mean())}.",
        f"- Universe effect (random mean minus S&P 500): {pct(rnd.CAGR.mean() - rnd.benchmark_CAGR.iloc[0])} per year.",
        "", "## Stage 1: single-feature rank IC by signal year (descriptive)", "",
        md_table(feats, index=False, floatfmt="{:.3f}"), "",
        "## Stage 2: alpha candidates", "", md_table(alpha_table[show], index=False, floatfmt="{:.4f}"), "",
        "Selection rule applied in order of complexity:", "",
        md_table(selection, index=False, floatfmt="{:.4f}"), "",
        f"**Selected alpha by the rule: {chosen or 'none (no stable candidate)'}**", "",
    ]
    if chosen:
        lines += ["## Stage 3: portfolio variants of the selected alpha", "",
                  md_table(port_table[show], index=False, floatfmt="{:.4f}"), "",
                  md_table(adoption, index=False, floatfmt="{:.4f}"), "",
                  f"**Portfolio variant adopted by the rule: {adopted or 'none (base stays: top 30, equal weight)'}**", ""]
    lines += [f"Run time {(time.time() - t_start) / 60:.0f} minutes.  Nothing was frozen.", ""]
    (OUT / "STUDY_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"== done in {(time.time() - t_start) / 60:.0f} min -> {OUT / 'STUDY_SUMMARY.md'}", flush=True)


if __name__ == "__main__":
    main()
