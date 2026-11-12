"""Momentum minimum-variance strategy.

    .venv/bin/python Homeworks/HW4/Emil_Blaignan/minvar.py dev  <candidate>
    .venv/bin/python Homeworks/HW4/Emil_Blaignan/minvar.py test <candidate>     # final test, run once

Each week (signal at the close, fill at the next close, 90-day hold, 13 overlapping tranches):
  1. Momentum bucket: keep stocks in the top `bucket` share by 12-1 momentum.
  2. Score (ridge or elastic net): a linear model of the next 90-day return (relative to the cross-section
     median, winsorized) on 15 rank-normalized characteristics, trained only on the momentum bucket and
     only on outcomes that have already matured; refit every quarter. Pick the top N scores.
  3. Weights: minimum variance among the N picks, long-only, fully invested, each weight <= cap, with a
     ridge-shrunk covariance  Sigma + shrink * mean(diag Sigma) * I  from the trailing 252 daily returns.
(Long-only + fully invested makes an L1 penalty a constant, so elastic net's sparsity only acts in step 2.)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import ElasticNet, Ridge

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backtest import execute, metrics  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "outputs"
BASE = ["momentum_21", "momentum_63", "momentum_252_21", "volatility_63", "drawdown_252", "net_margin",
        "operating_roa", "debt_to_assets", "cash_to_assets", "fcf_to_assets", "accruals_to_assets"]
EXTRA = ["beta_mkt", "beta_mom", "resid_mom", "idisc"]
FEATS = BASE + EXTRA
N_TOP, MATURE, COV_DAYS = 100, pd.Timedelta(days=98), 252
DEV_EVAL = (pd.Timestamp("2013-01-01"), pd.Timestamp("2015-12-31"))
TEST = (pd.Timestamp("2016-04-01"), pd.Timestamp("2026-09-30"))

CANDS = {  # sig: mom (raw momentum, no model) / ridge / enet; wt: mv (min variance) / ew (equal weight)
    "mom_mv":          dict(sig="mom", wt="mv", bucket=0.4, shrink=0.5, cap=0.05),
    "ridge_mv":        dict(sig="ridge", alpha=1e4, wt="mv", bucket=0.4, shrink=0.5, cap=0.05),
    "ridge_mv_heavy":  dict(sig="ridge", alpha=1e7, wt="mv", bucket=0.4, shrink=0.5, cap=0.05),
    "enet_mv_light":   dict(sig="enet", alpha=1e-5, l1=0.5, wt="mv", bucket=0.4, shrink=0.5, cap=0.05),
    "enet_mv":         dict(sig="enet", alpha=1e-4, l1=0.5, wt="mv", bucket=0.4, shrink=0.5, cap=0.05),
    "ridge_ew":        dict(sig="ridge", alpha=1e4, wt="ew", bucket=0.4, shrink=0.5, cap=0.05),
    "ridge_mv_shrink": dict(sig="ridge", alpha=1e4, wt="mv", bucket=0.4, shrink=0.1, cap=0.05),
    "ridge_mv_cap3":   dict(sig="ridge", alpha=1e4, wt="mv", bucket=0.4, shrink=0.5, cap=0.03),
    "ridge_mv_b20":    dict(sig="ridge", alpha=1e4, wt="mv", bucket=0.2, shrink=0.5, cap=0.05),
}


def load():
    w = pd.read_parquet(OUT / "weekly.parquet").sort_values(["date", "symbol"]).reset_index(drop=True)
    for c in FEATS:
        w[c + "_x"] = (w.groupby("date")[c].rank(pct=True) * 2 - 1).fillna(0.0)
    w["rank_mom"] = w.groupby("date").momentum_252_21.rank(pct=True).fillna(0.0)
    dm = w.target - w.groupby("date").target.transform("median")
    dev = (w.date <= "2015-12-31") & (w.label_end <= "2016-03-31")
    lo, hi = dm[dev].quantile([0.001, 0.999])  # winsorization bounds: development labels only
    w["y"] = dm.clip(lo, hi)
    return w


def daily_returns():
    cal = pd.read_parquet(OUT / "benchmark.parquet").date
    cal = cal[cal >= "2009-01-01"]
    d = pd.read_parquet(OUT / "panel.parquet", columns=["symbol", "date", "ret_1d"],
                        filters=[("date", ">=", pd.Timestamp("2009-01-01"))])
    return d.pivot(index="date", columns="symbol", values="ret_1d").reindex(cal)


def min_variance(R, d, symbols, shrink, cap):
    """Long-only, fully invested minimum-variance weights with a ridge-shrunk covariance."""
    n = len(symbols)
    pos = R.index.get_loc(d)
    X = R.iloc[max(0, pos - COV_DAYS + 1):pos + 1][list(symbols)]
    S = X.cov(min_periods=100)
    var = np.diag(S).copy()
    fill = np.nanmedian(var) if np.isfinite(var).any() else 4e-4
    var = np.where(np.isfinite(var), var, fill)
    S = S.to_numpy()
    S = np.where(np.isfinite(S), S, 0.0)
    np.fill_diagonal(S, var)
    S = (S + shrink * var.mean() * np.eye(n)) * 1e4
    vals, vecs = np.linalg.eigh(S)
    S = (vecs * np.clip(vals, 1e-6 * vals.max(), None)) @ vecs.T  # repair pairwise-covariance non-PSD
    cap = max(cap, 1.0 / n + 1e-9)
    res = minimize(lambda x: x @ S @ x, np.full(n, 1 / n), jac=lambda x: 2 * S @ x, method="SLSQP",
                   bounds=[(0, cap)] * n, constraints=[{"type": "eq", "fun": lambda x: x.sum() - 1, "jac": lambda x: np.ones(n)}],
                   options=dict(maxiter=300, ftol=1e-12))
    x = np.clip(res.x, 0, cap) if res.success else np.full(n, 1 / n)
    return x / x.sum()


def build_picks(w, R, c, dates):
    X = w[[f + "_x" for f in FEATS]].to_numpy(np.float32)
    y, dt, rk = w.y.to_numpy(), w.date.to_numpy(), w.rank_mom.to_numpy()
    dlist = pd.DatetimeIndex(dates)
    q = dlist.to_period("Q")
    refit = set(dlist[np.r_[True, q[1:] != q[:-1]]])
    model, coefs, rows = None, None, []
    for d in dlist:
        if c["sig"] != "mom" and (d in refit or model is None):
            m = (dt <= (d - MATURE).to_datetime64()) & ~np.isnan(y) & (rk >= 1 - c["bucket"])
            model = (Ridge(alpha=c["alpha"]) if c["sig"] == "ridge"
                     else ElasticNet(alpha=c["alpha"], l1_ratio=c["l1"], max_iter=2000)).fit(X[m], y[m])
            coefs = dict(zip(FEATS, model.coef_.round(4)))
        idx = np.flatnonzero(dt == d.to_datetime64())
        sub = w.iloc[idx]
        elig = idx[rk[idx] >= 1 - c["bucket"]]
        score = (w.momentum_252_21.to_numpy()[elig] if c["sig"] == "mom" else model.predict(X[elig]))
        score = np.nan_to_num(score, nan=-1e9)
        top = elig[np.argsort(-score)[:N_TOP]]
        syms = w.symbol.to_numpy()[top]
        wts = (min_variance(R, d, syms, c["shrink"], c["cap"]) if c["wt"] == "mv" else np.full(len(top), 1 / len(top)))
        rows += [(s, d, e, x) for s, e, x in zip(syms, w.entry_date.to_numpy()[top], wts)]
    p = pd.DataFrame(rows, columns=["symbol", "signal_date", "entry_date", "weight"])
    return p[p.entry_date.notna()].reset_index(drop=True), coefs


def backtest(picks, window, cost=0.001, haircut=0.0, **kw):
    start, end = window
    picks = picks[(picks.entry_date >= start) & (picks.entry_date <= end)]
    px = pd.read_parquet(OUT / "panel.parquet", columns=["symbol", "date", "adj_close"],
                         filters=[("symbol", "in", sorted(picks.symbol.unique())), ("date", ">=", start), ("date", "<=", end)])
    cal = pd.read_parquet(OUT / "benchmark.parquet").date
    px = px.pivot(index="date", columns="symbol", values="adj_close").reindex(cal[(cal >= start) & (cal <= end)])
    return execute(picks, px, cost=cost, n_target=N_TOP, exit_haircut=haircut, **kw)


def bench_curve(start, end):
    b = pd.read_parquet(OUT / "benchmark.parquet").set_index("date").adj_close.loc[start:end]
    return 1_000_000 * b / b.iloc[0]


def summary(eq):
    m = metrics(eq)
    m["ann_vol"] = eq.pct_change().std() * np.sqrt(252)
    return m


def dev(name):
    c, w, R = CANDS[name], load(), daily_returns()
    dates = w.date.drop_duplicates()
    dates = dates[(dates >= DEV_EVAL[0]) & (dates <= DEV_EVAL[1])]
    picks, coefs = build_picks(w, R, c, dates)
    window = (picks.entry_date.min(), DEV_EVAL[1])
    eq, log, stale = backtest(picks, window)
    res = dict(cfg=c, strategy=summary(eq), benchmark=summary(bench_curve(*window)), stale_exits=int(stale),
               avg_max_weight=float(picks.groupby("signal_date").weight.max().mean()), last_coefs=coefs)
    json.dump(res, open(OUT / f"dev_{name}.json", "w"), indent=1, default=float)
    print(name, json.dumps({k: res[k] for k in ("strategy", "avg_max_weight")}, default=float))


def test(name):
    c, w, R = CANDS[name], load(), daily_returns()
    dates = w.date.drop_duplicates()
    dates = dates[(dates >= dates[dates < TEST[0]].iloc[-1]) & (dates <= TEST[1])]
    picks, coefs = build_picks(w, R, c, dates)
    eq, log, stale = backtest(picks, TEST)
    out = dict(cfg=c, benchmark=summary(bench_curve(*TEST)), strategy=summary(eq),
               strategy_no_costs=summary(backtest(picks, TEST, cost=0.0)[0]),
               strategy_50pct_haircut=summary(backtest(picks, TEST, haircut=0.5)[0]),
               stale_exits=int(stale), buys=int((log.side == "BUY").sum()), last_coefs=coefs)
    json.dump(out, open(OUT / f"test_{name}.json", "w"), indent=1, default=float)
    pd.DataFrame({"S&P 500 TR": bench_curve(*TEST), name: eq}).to_csv(OUT / f"test_{name}_curves.csv")
    log.to_csv(OUT / f"test_{name}_trades.csv", index=False)
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    {"dev": dev, "test": test}[sys.argv[1]](sys.argv[2])
