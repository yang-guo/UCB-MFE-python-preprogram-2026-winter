"""Large-liquid core with a modest tilt.

    .venv/bin/python Homeworks/HW4/Emil_Blaignan/core.py dev  <candidate>
    .venv/bin/python Homeworks/HW4/Emil_Blaignan/core.py test <candidate>      # final test, run once

Why: the lecture universe's size ranking (capitalization proxy) is corrupted by share/unit errors, and the
S&P 500 is a cap-weighted large-cap index, so an equal-weighted pick from ~2,000 mostly small names cannot
track it. Size here is the trailing 63-day MEDIAN dollar volume (close x volume), known at the signal close.

Each week:
  1. Core = the K most-traded tradable stocks (largest trailing dollar volume).
  2. Selection: all of the core, or the top N inside a K=150 core by raw 12-1 momentum or by a ridge score
     (rank-normalized characteristics incl. size, ranked inside the core; trained on matured outcomes only).
  3. Weights: equal / proportional to dollar volume (a cap-weight proxy, each weight <= cap) / minimum variance.
Signal at the close, fill at the next close, 90-day hold, 13 overlapping tranches (backtest.py).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

sys.path.insert(0, str(Path(__file__).resolve().parent))
import minvar as mv  # noqa: E402  (covariance + minimum-variance solver, backtest wrappers)

OUT = mv.OUT
FEATS = mv.FEATS + ["adv_log"]
K_TILT, MATURE = 150, pd.Timedelta(days=98)
DEV_EVAL, TEST = mv.DEV_EVAL, mv.TEST

CANDS = {
    "core50_eq":      dict(K=50, N=50, sel="adv", wt="eq"),
    "core100_eq":     dict(K=100, N=100, sel="adv", wt="eq"),
    "core50_advw":    dict(K=50, N=50, sel="adv", wt="advw", cap=0.10),
    "core100_advw":   dict(K=100, N=100, sel="adv", wt="advw", cap=0.10),
    "mom_eq":         dict(K=K_TILT, N=50, sel="mom", wt="eq"),
    "mom_advw":       dict(K=K_TILT, N=50, sel="mom", wt="advw", cap=0.10),
    "mom_mv":         dict(K=K_TILT, N=50, sel="mom", wt="mv", cap=0.10, shrink=0.5),
    "ridge_eq":       dict(K=K_TILT, N=50, sel="ridge", alpha=1e3, wt="eq"),
    "ridge_advw":     dict(K=K_TILT, N=50, sel="ridge", alpha=1e3, wt="advw", cap=0.10),
    "ridge_mv":       dict(K=K_TILT, N=50, sel="ridge", alpha=1e3, wt="mv", cap=0.10, shrink=0.5),
}


def adv_table():
    f = OUT / "adv_weekly.parquet"
    if f.exists():
        return pd.read_parquet(f)
    d = pd.read_parquet(OUT / "panel.parquet", columns=["symbol", "date", "close", "volume"],
                        filters=[("date", ">=", pd.Timestamp("2009-06-01"))])
    dv = (d.close * d.volume).where(lambda x: x > 0)
    D = pd.DataFrame({"symbol": d.symbol, "date": d.date, "dv": dv}).pivot(index="date", columns="symbol", values="dv")
    adv = D.rolling(63, min_periods=40).median()
    w = pd.read_parquet(OUT / "weekly.parquet", columns=["symbol", "date"])
    w["adv"] = adv.to_numpy()[adv.index.get_indexer(w.date), adv.columns.get_indexer(w.symbol)]
    w.to_parquet(f, index=False)
    return w


def load():
    w = pd.read_parquet(OUT / "weekly.parquet").sort_values(["date", "symbol"]).reset_index(drop=True)
    w = w.merge(adv_table(), on=["symbol", "date"], how="left")
    w = w[w.adv.notna()].reset_index(drop=True)
    w["adv_rank"] = w.groupby("date").adv.rank(ascending=False, method="first")
    w["adv_log"] = np.log(w.adv)
    core = w[w.adv_rank <= mv_K()].copy()
    for c in FEATS:  # rank-normalize INSIDE the core so the model separates large liquid names from each other
        core[c + "_x"] = (core.groupby("date")[c].rank(pct=True) * 2 - 1).fillna(0.0)
    dm = core.target - core.groupby("date").target.transform("median")
    dev = (core.date <= "2015-12-31") & (core.label_end <= "2016-03-31")
    lo, hi = dm[dev].quantile([0.001, 0.999])
    core["y"] = dm.clip(lo, hi)
    return w, core


def mv_K():
    return K_TILT


def advw(adv, cap):
    x = adv / adv.sum()
    for _ in range(20):
        x = np.minimum(x, cap)
        x = x / x.sum()
    return x


def build_picks(w, core, R, c, dates):
    Xc = core[[f + "_x" for f in FEATS]].to_numpy(np.float32)
    y, dt = core.y.to_numpy(), core.date.to_numpy()
    dlist = pd.DatetimeIndex(dates)
    q = dlist.to_period("Q")
    refit = set(dlist[np.r_[True, q[1:] != q[:-1]]])
    model, coefs, rows = None, None, []
    for d in dlist:
        if c["sel"] == "ridge" and (d in refit or model is None):
            m = (dt <= (d - MATURE).to_datetime64()) & ~np.isnan(y)
            model = Ridge(alpha=c["alpha"]).fit(Xc[m], y[m])
            coefs = dict(zip(FEATS, model.coef_.round(4)))
        g = w[(w.date == d) & (w.adv_rank <= c["K"])]
        if c["sel"] == "adv":
            s = g
        elif c["sel"] == "mom":
            s = g.assign(score=g.momentum_252_21.fillna(-9)).nlargest(c["N"], "score")
        else:
            gi = np.flatnonzero(dt == d.to_datetime64())
            pred = pd.Series(model.predict(Xc[gi]), index=core.index[gi])
            s = core.loc[pred.nlargest(c["N"]).index]
        syms = s.symbol.to_numpy()
        if c["wt"] == "eq":
            wt = np.full(len(s), 1 / len(s))
        elif c["wt"] == "advw":
            wt = advw(s.adv.to_numpy(), c["cap"])
        elif c["wt"] == "riskw":  # liquidity weight divided by volatility: fewer big bets on volatile names
            wt = advw(s.adv.to_numpy() / s.volatility_63.to_numpy().clip(0.05), c["cap"])
        else:
            wt = mv.min_variance(R, d, syms, c["shrink"], c["cap"])
        rows += [(a, d, e, x) for a, e, x in zip(syms, s.entry_date.to_numpy(), wt)]
    p = pd.DataFrame(rows, columns=["symbol", "signal_date", "entry_date", "weight"])
    return p[p.entry_date.notna()].reset_index(drop=True), coefs


def dev(name):
    c, (w, core), R = CANDS[name], load(), mv.daily_returns()
    dates = w.date.drop_duplicates()
    dates = dates[(dates >= DEV_EVAL[0]) & (dates <= DEV_EVAL[1])]
    picks, coefs = build_picks(w, core, R, c, dates)
    window = (picks.entry_date.min(), DEV_EVAL[1])
    eq, log, stale = mv.backtest(picks, window)
    res = dict(cfg=c, strategy=mv.summary(eq), benchmark=mv.summary(mv.bench_curve(*window)),
               stale_exits=int(stale), avg_max_weight=float(picks.groupby("signal_date").weight.max().mean()),
               last_coefs=coefs)
    json.dump(res, open(OUT / f"dev_core_{name}.json", "w"), indent=1, default=float)
    print(name, json.dumps(res["strategy"], default=float), "maxw", round(res["avg_max_weight"], 3))


def test(name):
    c, (w, core), R = CANDS[name], load(), mv.daily_returns()
    dates = w.date.drop_duplicates()
    dates = dates[(dates >= dates[dates < TEST[0]].iloc[-1]) & (dates <= TEST[1])]
    picks, coefs = build_picks(w, core, R, c, dates)
    eq, log, stale = mv.backtest(picks, TEST)
    out = dict(cfg=c, benchmark=mv.summary(mv.bench_curve(*TEST)), strategy=mv.summary(eq),
               strategy_no_costs=mv.summary(mv.backtest(picks, TEST, cost=0.0)[0]),
               strategy_50pct_haircut=mv.summary(mv.backtest(picks, TEST, haircut=0.5)[0]),
               stale_exits=int(stale), buys=int((log.side == "BUY").sum()), last_coefs=coefs)
    json.dump(out, open(OUT / f"test_core_{name}.json", "w"), indent=1, default=float)
    pd.DataFrame({"S&P 500 TR": mv.bench_curve(*TEST), name: eq}).to_csv(OUT / f"test_core_{name}_curves.csv")
    log.to_csv(OUT / f"test_core_{name}_trades.csv", index=False)
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    {"dev": dev, "test": test}[sys.argv[1]](sys.argv[2])
