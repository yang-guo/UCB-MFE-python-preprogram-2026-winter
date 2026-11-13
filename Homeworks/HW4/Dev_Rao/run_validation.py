"""Step 2: purged expanding walk-forward validation inside development only.

    uv run python run_validation.py --db top2000      # ~20-40 min

For each validation year Y in {2013, 2014, 2015}: fit on eligible weekly
signals from 2010-01-01 whose label_end < first signal of Y, then predict all of
Y's weekly signals. Only rows with signal date <= 2015-12-31 are ever loaded.
Outputs: outputs/cv_*.csv, results/cv_summary.csv, results/cv_ic_by_fold.png,
and the walk-forward predictions table `pred_val` in outputs/preds_<db>.duckdb (used by
run_strategy_dev.py).
"""

import argparse
import itertools
import json
import time

import duckdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hw4 import config, model as M
from hw4.labels import add_bench_forward
from hw4.features import MODEL_CATEGORICAL, MODEL_NUMERIC

RIDGE_GRID = [{"kind": "ridge", "alpha": a} for a in (1.0, 10.0, 100.0, 1000.0)]
HGB_GRID = [{"kind": "hgb", "max_iter": it, "max_leaf_nodes": lf, "min_samples_leaf": ms,
             "learning_rate": 0.05, "l2_regularization": 1.0}
            for it, lf, ms in itertools.product((200, 400), (15, 31), (500, 2000))]
TARGETS = ("rank", "demean", "resid")
COMPOSITE = {"kind": "composite", "target": "none"}


def spec_name(spec):
    if spec["kind"] == "composite":
        return "composite"
    if spec["kind"] == "ridge":
        return f"ridge_a{spec['alpha']:g}_{spec['target']}"
    return f"hgb_it{spec['max_iter']}_lf{spec['max_leaf_nodes']}_ms{spec['min_samples_leaf']}_{spec['target']}"


def load_dev_panel(cache):
    cols = ["date", "symbol", "sector", "target", "entry_date", "label_end", "liq_rank", "universe_rank",
            "beta_252"] + \
        [c for c in MODEL_NUMERIC if c not in MODEL_CATEGORICAL]
    q = f"SELECT {', '.join(dict.fromkeys(cols))} FROM panel WHERE date >= ? AND date <= ? ORDER BY date, symbol"
    p = cache.execute(q, [config.TRAIN_START, config.DEV_END]).fetchdf()
    assert p.date.max() <= pd.Timestamp(config.DEV_END), "test-period rows loaded in development"
    for c in MODEL_NUMERIC:
        p[c] = p[c].astype("float32")
    bench = cache.execute("SELECT date, adj_close FROM bench").fetchdf().set_index("date").adj_close
    return add_bench_forward(p, bench)


def main(db_key, quick=False):
    t0 = time.time()
    cache = duckdb.connect(str(config.OUTPUT_DIR / f"cache_{db_key}.duckdb"), read_only=True)
    panel = load_dev_panel(cache)
    print(f"dev panel: {len(panel):,} eligible rows, {panel.date.nunique()} dates, "
          f"label coverage {panel.target.notna().mean():.3f}")
    folds = []
    for y in config.VALIDATION_YEARS:
        in_year = panel.date.dt.year.eq(y).to_numpy()
        fit_date = panel.loc[in_year, "date"].min()
        folds.append((y, fit_date, in_year))

    specs = [COMPOSITE] + [dict(s, target=t) for s in RIDGE_GRID + HGB_GRID for t in TARGETS]
    if quick:
        specs = specs[:3] + [dict(RIDGE_GRID[0], target="resid"), dict(HGB_GRID[0], target="resid")]
    rows, preds = [], []
    for spec in specs:
        name = spec_name(spec)
        for y, fit_date, mask in folds:
            t1 = time.time()
            pred, _, n_train = M.fit_predict(panel, spec, fit_date, mask)
            out = panel.loc[mask, ["date", "symbol", "target"]].assign(score=pred)
            rep, ic, dec = M.ranking_report(out, "score")
            rows.append({"model": name, "kind": spec["kind"], "target_variant": spec["target"],
                         "fold": y, "fit_date": fit_date.date(), "n_train": n_train,
                         "fit_seconds": round(time.time() - t1, 1), **rep})
            preds.append(out[["date", "symbol"]].assign(model=name, score=pred.astype("float32")))
            print(f"[{time.time()-t0:6.0f}s] {name:<40} {y}  IC={rep['mean_IC']:+.4f}  "
                  f"top-dec={rep['top_decile_excess']:+.4f}  n_train={n_train:,}  ({time.time()-t1:.0f}s)")
    cv = pd.DataFrame(rows)

    # ---- ensembles (members chosen by mean fold IC) ----
    pv = pd.concat(preds, ignore_index=True)
    mean_ic = cv.groupby("model").mean_IC.mean()
    kinds = cv.groupby("model").kind.first()
    best = {k: mean_ic[kinds.eq(k)].idxmax() for k in ("ridge", "hgb") if (cv.kind == k).any()}
    ensembles = {}
    if len(best) == 2:
        ensembles["ensemble"] = [best["ridge"], best["hgb"]]          # spec: best ridge + best HGB
        best_ml = best["ridge"] if mean_ic[best["ridge"]] >= mean_ic[best["hgb"]] else best["hgb"]
        if "composite" in mean_ic:
            ensembles["ens_composite_ml"] = ["composite", best_ml]     # literature prior + learned model
    for ens_name, members in ensembles.items():
        wide = pv.loc[pv.model.isin(members)].pivot_table(index=["date", "symbol"], columns="model",
                                                          values="score").reset_index()
        wide["score"] = M.rank_average(wide, members)
        wide = wide.merge(panel[["date", "symbol", "target"]], on=["date", "symbol"])
        for y, fit_date, _ in folds:
            rep, _, _ = M.ranking_report(wide.loc[wide.date.dt.year.eq(y)], "score")
            cv.loc[len(cv)] = {"model": ens_name, "kind": "ensemble", "target_variant": "+".join(members),
                               "fold": y, "fit_date": fit_date.date(), **rep}
        pv = pd.concat([pv, wide[["date", "symbol"]].assign(model=ens_name, score=wide.score.astype("float32"))])

    summary = (cv.groupby(["model", "kind", "target_variant"])
               .agg(mean_IC=("mean_IC", "mean"), min_fold_IC=("mean_IC", "min"), std_fold_IC=("mean_IC", "std"),
                    IC_t_NW13=("IC_t_NW13", "mean"), top_decile_excess=("top_decile_excess", "mean"),
                    decile_monotonicity=("decile_monotonicity", "mean"),
                    top25_excess=("top25_excess", "mean"), top25_hit=("top25_hit", "mean"))
               .sort_values("mean_IC", ascending=False).reset_index())
    # Selection rule (fixed before looking): highest mean IC among models whose IC
    # is positive in every fold; ties within 0.002 go to the simpler model.
    ok = summary.loc[summary.min_fold_IC > 0]
    if ok.empty:
        print("WARNING: no model has positive IC in every fold; falling back to max mean IC")
        ok = summary
    chosen = ok.iloc[0]
    close = ok.loc[ok.mean_IC >= chosen.mean_IC - 0.002]
    simplicity = {"composite": 0, "ridge": 1, "hgb": 2, "ensemble": 3}
    chosen = close.assign(s=close.kind.map(simplicity)).sort_values(["s", "mean_IC"], ascending=[True, False]).iloc[0]
    print(summary.round(4).to_string(index=False))
    print(f"\nSELECTED: {chosen.model} (mean IC {chosen.mean_IC:.4f}, min fold {chosen.min_fold_IC:.4f})")

    config.RESULTS_DIR.mkdir(exist_ok=True)
    cv.to_csv(config.OUTPUT_DIR / f"cv_folds_{db_key}.csv", index=False)
    summary.to_csv(config.OUTPUT_DIR / f"cv_summary_{db_key}.csv", index=False)
    if db_key == "top2000" and not quick:
        summary.round(5).to_csv(config.RESULTS_DIR / "cv_summary.csv", index=False)
        cv.round(5).to_csv(config.RESULTS_DIR / "cv_folds.csv", index=False)
        plot_cv(cv, best, config.RESULTS_DIR / "cv_ic_by_fold.png")
    cache.close()
    with duckdb.connect(str(config.OUTPUT_DIR / f"preds_{db_key}.duckdb")) as store:
        store.execute("DROP TABLE IF EXISTS pred_val")
        store.register("pv", pv)
        store.execute("CREATE TABLE pred_val AS SELECT * FROM pv")
    (config.OUTPUT_DIR / f"cv_selection_{db_key}.json").write_text(json.dumps(
        {"selected": chosen.model, "best_ridge": best.get("ridge"), "best_hgb": best.get("hgb"),
         "rule": "max mean fold IC with IC>0 in all folds; within 0.002 prefer simpler"}, indent=2))
    print(f"done in {time.time()-t0:.0f}s")


def plot_cv(cv, best, path):
    show = [m for m in ["composite", *best.values(), "ensemble", "ens_composite_ml"] if m in set(cv.model)]
    d = cv.loc[cv.model.isin(show)].pivot(index="fold", columns="model", values="mean_IC")[show]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    d.plot.bar(ax=ax[0], rot=0)
    ax[0].axhline(0, color="k", lw=0.8)
    ax[0].set(title="Mean weekly rank IC by validation year", ylabel="Spearman IC", xlabel="")
    ax[0].legend(fontsize=8)
    s = cv.groupby("model").mean_IC.mean().sort_values()
    colors = ["#4C72B0" if m.startswith("ridge") else "#DD8452" if m.startswith("hgb") else
              "#8172B3" if m == "composite" else "#55A868" for m in s.index]
    ax[1].barh(range(len(s)), s.values, color=colors)
    ax[1].set_yticks(range(len(s)), s.index, fontsize=6)
    ax[1].set(title="All candidates: mean IC over 3 folds", xlabel="Spearman IC")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", choices=list(config.DATABASES), default="class")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    main(a.db, a.quick)
