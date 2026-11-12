"""HW4 experiment runner.

    .venv/bin/python Homeworks/HW4/Emil_Blaignan/run_hw4.py dev    # tune on 2010-2015 only
    .venv/bin/python Homeworks/HW4/Emil_Blaignan/run_hw4.py test   # final 2016-2026 run, config fixed by `dev`

`dev` writes outputs/chosen_config.json; `test` only reads it. Test-period results never feed back
into any choice.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backtest import execute, metrics  # noqa: E402
import strategy as st  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "outputs"
PANEL = OUT / "panel.parquet"
LAMS = [1e-4, 1e-2, 1.0, 1e2, 1e4]   # relative ridge penalty grid
N_TOP = 100          # holdings per weekly tranche (fixed a priori, not tuned)
DEV_EVAL = ("2013-01-01", "2015-12-31")   # walk-forward validation years inside development
TEST_START, TEST_END = pd.Timestamp("2016-04-01"), pd.Timestamp("2026-09-30")


def setup():
    bench = pd.read_parquet(OUT / "benchmark.parquet").set_index("date").adj_close
    cal = bench.index
    anchor = cal.get_indexer([TEST_START], method="bfill")[0] - 1  # last session before the first test close
    idx = np.arange(anchor % 5, len(cal), 5)                      # weekly cadence anchored to the test start
    sig = cal[idx]
    sig = sig[sig >= "2010-01-01"]
    cols = ["symbol", "date"] + st.FEATS + ["tradable", "target", "entry_date", "label_end"]
    p = pd.read_parquet(PANEL, columns=cols, filters=[("date", ">=", pd.Timestamp("2010-01-01"))])
    p = p[p.tradable & p.date.isin(sig)]
    # winsorization bounds come from development labels only (matured before the buffer ends)
    dev = p[(p.date <= "2015-12-31") & (p.label_end <= "2016-03-31")]
    y = dev.target - dev.groupby("date").target.transform("median")
    clip = tuple(y.quantile([0.001, 0.999]))
    return bench, st.Weeks(p), clip


def score_fn(weeks, clip, kind, P=1000, seed=0, scale=0.5, lam=1e-2):
    if kind == "momentum":
        return lambda d: weeks.panel.momentum_252_21.to_numpy()[slice(*weeks.rows(d))]
    fm = st.FactorModel(weeks, st.make_features(kind, P, seed, scale), clip)
    return fm, lam


def walk(weeks, fm, lam, dates):
    def scorer(fit_date):
        pi, _ = fm.fit(fit_date, lam)
        return lambda d: fm.score(d, pi)
    return st.walk_forward_scores(weeks, scorer, dates)


def momentum_scores(weeks, dates):
    m = weeks.panel.momentum_252_21.to_numpy()
    return {d: np.nan_to_num(m[slice(*weeks.rows(d))], nan=-9) for d in dates}


def dev_stage():
    bench, weeks, clip = setup()
    dates = weeks.dates[(weeks.dates >= DEV_EVAL[0]) & (weeks.dates <= DEV_EVAL[1])]
    rows = []
    mom = momentum_scores(weeks, dates)
    rows.append(dict(model="momentum", P=0, scale=0, lam=0, ic=st.rank_ic(weeks, mom).mean(),
                     spread=st.top_n_spread(weeks, mom, N_TOP).mean()))
    for lam in LAMS:
        fm = st.FactorModel(weeks, st.make_features("linear"), clip)
        s = walk(weeks, fm, lam, dates)
        rows.append(dict(model="linear", P=st.D, scale=0, lam=lam, ic=st.rank_ic(weeks, s).mean(),
                         spread=st.top_n_spread(weeks, s, N_TOP).mean()))
    # NB: the random-feature `scale` is omitted: ReLU has no bias, so ReLU(c*z) = c*ReLU(z) and a
    # positive scale only rescales the features; the relative ridge penalty makes it a no-op.
    for P in [1000, 4000]:
        fm = st.FactorModel(weeks, st.make_features("rf", P, 0, 0.5), clip)
        for lam in LAMS:
            s = walk(weeks, fm, lam, dates)
            rows.append(dict(model="rf", P=P, scale=0.5, lam=lam, ic=st.rank_ic(weeks, s).mean(),
                             spread=st.top_n_spread(weeks, s, N_TOP).mean()))
            print(rows[-1], flush=True)
    grid = pd.DataFrame(rows)
    grid.to_csv(OUT / "dev_grid.csv", index=False)
    print(grid.round(5).to_string(index=False))
    best_rf = grid[grid.model == "rf"].sort_values("ic", ascending=False).iloc[0]
    best_lin = grid[grid.model == "linear"].sort_values("ic", ascending=False).iloc[0]
    cfg = dict(rf=dict(P=int(best_rf.P), scale=float(best_rf.scale), lam=float(best_rf.lam)),
               linear=dict(lam=float(best_lin.lam)), clip=list(clip), n_top=N_TOP)
    # seed sensitivity of the chosen RF config (development only)
    ics = []
    for seed in [0, 1, 2]:
        fm = st.FactorModel(weeks, st.make_features("rf", cfg["rf"]["P"], seed, cfg["rf"]["scale"]), clip)
        ics.append(st.rank_ic(weeks, walk(weeks, fm, cfg["rf"]["lam"], dates)).mean())
    cfg["dev_seed_ics"] = ics
    json.dump(cfg, open(OUT / "chosen_config.json", "w"), indent=2)
    print("chosen:", cfg)


def run_strategy(name, scores, weeks, start, end, cost=0.001, exit_haircut=0.0):
    picks = st.make_picks(weeks, scores, N_TOP)
    picks = picks[(picks.entry_date >= start) & (picks.entry_date <= end)]
    syms = sorted(picks.symbol.unique())
    px = pd.read_parquet(PANEL, columns=["symbol", "date", "adj_close"],
                         filters=[("symbol", "in", syms), ("date", ">=", start), ("date", "<=", end)])
    px = px.pivot(index="date", columns="symbol", values="adj_close")
    cal = pd.read_parquet(OUT / "benchmark.parquet").date
    px = px.reindex(cal[(cal >= start) & (cal <= end)])
    eq, log, stale = execute(picks, px, cost=cost, n_target=N_TOP, exit_haircut=exit_haircut)
    return eq, log, stale


def test_stage():
    cfg = json.load(open(OUT / "chosen_config.json"))
    bench, weeks, _ = setup()
    clip = tuple(cfg["clip"])
    dates = weeks.dates[(weeks.dates >= pd.Timestamp("2016-03-31")) & (weeks.dates <= TEST_END)]
    dates = dates[dates >= dates[dates < TEST_START][-1]]
    models = {"momentum": momentum_scores(weeks, dates)}
    fm = st.FactorModel(weeks, st.make_features("linear"), clip)
    models["linear"] = walk(weeks, fm, cfg["linear"]["lam"], dates)
    r = cfg["rf"]
    fm = st.FactorModel(weeks, st.make_features("rf", r["P"], 0, r["scale"]), clip)
    models["rf"] = walk(weeks, fm, r["lam"], dates)

    b = bench.loc[TEST_START:TEST_END]
    b = 1_000_000 * b / b.iloc[0]
    rows = {"S&P 500 TR": metrics(b)}
    curves = {"S&P 500 TR": b}
    extra = {}
    for name, s in models.items():
        eq, log, stale = run_strategy(name, s, weeks, TEST_START, TEST_END)
        rows[name] = metrics(eq)
        curves[name] = eq
        extra[name] = dict(stale_exits=int(stale), trades=len(log), buys=int((log.side == "BUY").sum()),
                           avg_cash_share=None)
        if name == "rf":
            gross, _, _ = run_strategy(name, s, weeks, TEST_START, TEST_END, cost=0.0)
            half, _, _ = run_strategy(name, s, weeks, TEST_START, TEST_END, exit_haircut=0.5)
            rows["rf (no costs)"] = metrics(gross)
            rows["rf (50% haircut on stale exits)"] = metrics(half)
            log.to_csv(OUT / "trades_rf.csv", index=False)
    res = pd.DataFrame(rows).T
    res.to_csv(OUT / "test_results.csv")
    pd.DataFrame(curves).to_csv(OUT / "test_curves.csv")
    json.dump(extra, open(OUT / "test_extra.json", "w"), indent=2)
    pd.options.display.float_format = "{:,.4f}".format
    print(res.to_string())
    print(extra)
    plot(curves)


def plot(curves):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 5.5))
    style = {"S&P 500 TR": dict(color="black", lw=2), "rf": dict(color="#1f77b4", lw=2),
             "linear": dict(color="#2ca02c", lw=1.2, ls="--"), "momentum": dict(color="#d62728", lw=1.2, ls="--")}
    label = {"rf": "Random-feature Markowitz (chosen)", "linear": "Linear weights", "momentum": "Pure momentum"}
    for k, v in curves.items():
        ax.plot(v.index, v / v.iloc[0] - 1, label=label.get(k, k), **style[k])
    ax.set_title("HW4 final test: cumulative return, Apr 2016 - Sep 2026 (10 bps/side costs)")
    ax.set_ylabel("Cumulative return")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "cumulative_returns.png", dpi=150)


if __name__ == "__main__":
    {"dev": dev_stage, "test": test_stage}[sys.argv[1]]()
