"""Final strategy. A sector relative alpha score, a PCA risk model and an optimizer that controls
tracking error against the S&P 500 total return index.

    python Homeworks/HW4/Emil_Blaignan/benchrel.py dev    # tune on 2011 to 2015 only, saves my choice
    python Homeworks/HW4/Emil_Blaignan/benchrel.py test   # one run on 2016 to 2026 using that choice

Each week at the close I do this
  1. Candidates are the 300 most traded stocks (trailing 63 day median dollar volume).
  2. Every characteristic is winsorized and z scored within sector. A model turns them into a score.
     FM is Fama-MacBeth premia shrunk toward zero, EN is elastic net, HGB is boosted trees and
     MF is a fixed score (momentum + profitability + free cash flow - volatility). Models only train on
     outcomes that have already finished and are refit every quarter.
  3. The score becomes an alpha with  alpha = IC * 90 day volatility * z, using a cautious IC of 0.03.
  4. Risk is 5 PCA factors plus stock specific variance from 252 days of returns. Tracking error uses
     Var(w'r - b) = w'Sigma w - 2 w'c + var(b) where b is the index return and c_i = beta_i * var(b).
  5. The optimizer maximizes alpha'w - (lam / 2) * TE^2 with long only, fully invested, each weight at
     most 5% and portfolio beta within 0.05 of 1.
The fill is the next close, each purchase is held 90 days, 13 weekly tranches overlap, costs are 10 bps
per side and cash earns 2%.
"""
import json
import sys
import warnings
from pathlib import Path

import cvxpy as cp
import duckdb
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import ElasticNet

sys.path.insert(0, str(Path(__file__).resolve().parent))
import helpers as h  # noqa: E402

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[3]
DB = ROOT / "Lectures" / "modeling" / "data" / "equities_top2000.duckdb"
K, CAP, BETA_TOL, IC, N_FACTORS, WINDOW = 300, 0.05, 0.05, 0.03, 5, 252
LAMS = [2.0, 10.0, 50.0]
MODELS = ["fm", "en", "hgb", "mf"]


# data

def load_core():
    """Weekly rows for the 300 most traded stocks with sector z scored characteristics and the target."""
    w = pd.read_parquet(h.OUT / "weekly.parquet").merge(h.adv_table(), on=["symbol", "date"], how="left")
    w = w[w.adv.notna()].copy()
    w["adv_rank"] = w.groupby("date").adv.rank(ascending=False, method="first")
    w = w[w.adv_rank <= K].sort_values(["date", "symbol"]).reset_index(drop=True)
    w["adv_log"] = np.log(w.adv)
    sec = duckdb.connect(str(DB), read_only=True).execute("select symbol, sector from universe").fetchdf()
    w = w.merge(sec, on="symbol", how="left")
    w["sector"] = w.sector.fillna("Unknown")
    by_date = w.groupby("date")
    small_sector = w.groupby(["date", "sector"]).symbol.transform("size") < 8  # too few names, z score over the whole date
    for f in h.FEATS:
        x = w[f].clip(by_date[f].transform(lambda s: s.quantile(0.01)), by_date[f].transform(lambda s: s.quantile(0.99)))
        w["_x"] = x
        g = w.groupby(["date", "sector"])["_x"]
        z_sector = (x - g.transform("mean")) / g.transform("std")
        z_date = (x - w.groupby("date")["_x"].transform("mean")) / w.groupby("date")["_x"].transform("std")
        z = np.where(small_sector, z_date, z_sector)
        w[f + "_z"] = pd.Series(z).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-3, 3).to_numpy()
    # target is the 90 day return minus the sector median that week, clipped at development quantiles
    r = w.target.clip(-1, 3)
    rel = r - r.groupby([w.date, w.sector]).transform("median")
    dev = (w.date <= "2015-12-31") & (w.label_end <= "2016-03-31")
    lo, hi = rel[dev].quantile([0.001, 0.999])
    w["y"] = rel.clip(lo, hi)
    return w.drop(columns="_x")


# alpha models

class Alpha:
    def __init__(self, w):
        self.X = w[[f + "_z" for f in h.FEATS]].to_numpy(np.float64)
        self.y, self.dt, self.label_end = w.y.to_numpy(), w.date.to_numpy(), w.label_end.to_numpy()
        self.rows = {d: np.flatnonzero(self.dt == d.to_datetime64()) for d in w.date.drop_duplicates()}
        # Fama-MacBeth. One cross sectional regression per week. They do not depend on the fit date.
        self.gamma = {}
        for d, ix in self.rows.items():
            ok = ~np.isnan(self.y[ix])
            if ok.sum() > 60:
                A = np.c_[np.ones(ok.sum()), self.X[ix][ok]]
                self.gamma[d] = np.linalg.lstsq(A, self.y[ix][ok], rcond=1e-8)[0][1:]

    def matured(self, fit_date):
        """Rows whose 90 day outcome was already known on the fit date."""
        m = (self.dt <= (fit_date - h.MATURE).to_datetime64()) & ~np.isnan(self.y)
        assert (self.label_end[m] <= fit_date.to_datetime64()).all(), "training row with an unfinished holding period"
        return m

    def fm_premia(self, fit_date, lag=13):
        weeks = sorted(d for d in self.gamma if d <= fit_date - h.MATURE)
        if len(weeks) < 30:
            return None, None
        G = np.stack([self.gamma[d] for d in weeks])
        mean, T = G.mean(0), len(G)
        e = G - mean
        var = (e ** 2).sum(0) / T
        for l in range(1, lag + 1):  # Newey-West, because 90 day targets overlap across about 13 weekly dates
            var = var + 2 * (1 - l / (lag + 1)) * (e[l:] * e[:-l]).sum(0) / T
        t = mean / np.sqrt(np.maximum(var, 1e-12) / T)
        shrunk = mean * t ** 2 / (t ** 2 + 4.0)  # a premium with t of 2 keeps half its size, t near 0 goes to zero
        return shrunk, pd.DataFrame({"mean": mean, "t_stat": t, "shrunk": shrunk}, index=h.FEATS)

    def fit(self, model, fit_date):
        if model == "mf":
            idx = [h.FEATS.index(f) for f in ["momentum_252_21", "operating_roa", "fcf_to_assets", "volatility_63"]]
            sign = np.array([1, 1, 1, -1.0])
            return lambda X: X[:, idx] @ sign
        if model == "fm":
            g, _ = self.fm_premia(fit_date)
            return None if g is None else (lambda X: X @ g)
        m = self.matured(fit_date)
        if m.sum() < 5000:
            return None
        est = (ElasticNet(alpha=5e-4, l1_ratio=0.5, max_iter=2000) if model == "en" else
               HistGradientBoostingRegressor(max_depth=3, max_iter=100, learning_rate=0.05, min_samples_leaf=500,
                                              l2_regularization=1.0, random_state=0))
        est.fit(self.X[m], self.y[m])
        return est.predict

    def scores(self, model, dates):
        q = pd.DatetimeIndex(dates).to_period("Q")
        refit = set(pd.DatetimeIndex(dates)[np.r_[True, q[1:] != q[:-1]]])  # first week of each quarter
        out, predict = {}, None
        for d in dates:
            if d in refit or predict is None:
                predict = self.fit(model, d)
            out[d] = None if predict is None else predict(self.X[self.rows[d]])
        return out


def zscore_within_sector(score, sector):
    s = pd.Series(score)
    z = (s - s.groupby(sector).transform("mean")) / s.groupby(sector).transform("std")
    z_all = (s - s.mean()) / s.std()
    n = s.groupby(sector).transform("size")
    return np.nan_to_num(np.where(n >= 8, z, z_all), nan=0.0)


# risk model and optimizer

_BENCH = None


def bench_returns():
    global _BENCH
    if _BENCH is None:
        _BENCH = pd.read_parquet(h.OUT / "benchmark.parquet").set_index("date").adj_close.pct_change()
    return _BENCH


def risk_model(R, d, symbols):
    pos = R.index.get_loc(d)
    X = R.iloc[pos - WINDOW + 1:pos + 1].reindex(columns=symbols)
    keep = X.notna().sum().to_numpy() >= 200  # need at least 200 days of history
    Xd = (X - X.mean()).fillna(0.0).to_numpy()[:, keep]
    b = bench_returns().reindex(X.index).fillna(0.0).to_numpy()
    bd = b - b.mean()
    T = len(Xd)
    S = Xd.T @ Xd / T
    vals, vecs = np.linalg.eigh(S)
    B = vecs[:, -N_FACTORS:] * np.sqrt(np.maximum(vals[-N_FACTORS:], 0))
    var = np.diag(S)
    D = np.maximum(var - (B ** 2).sum(1), 0.2 * var)  # specific variance, never below 20% of total
    sb2 = bd.var()
    beta = 0.8 * (Xd.T @ bd / T) / sb2 + 0.2          # betas shrunk toward 1
    return keep, B, D, beta * sb2, beta, sb2


def optimize(alpha, risk, lam):
    keep, B, D, c, beta, sb2 = risk
    n = len(D)
    for cap, tol in [(CAP, BETA_TOL), (0.10, 0.10), (1.0, 0.25)]:  # loosen only if the first problem is infeasible
        w = cp.Variable(n)
        te2 = cp.sum_squares(B.T @ w) + cp.sum(cp.multiply(D, cp.square(w))) - 2 * c @ w + sb2
        prob = cp.Problem(cp.Maximize(alpha[keep] @ w - 0.5 * lam * 252 * te2),
                          [w >= 0, cp.sum(w) == 1, w <= cap, cp.abs(beta @ w - 1) <= tol])
        try:
            prob.solve(solver=cp.CLARABEL)
        except Exception:
            continue
        if w.value is not None and prob.status in ("optimal", "optimal_inaccurate"):
            x = np.clip(w.value, 0, None)
            return x / x.sum()
    return np.full(n, 1.0 / n)


def build_picks(frame, alpha_model, model, lam, dates, R):
    """Weekly portfolios. Returns picks (symbol, signal_date, entry_date, weight) and the average rank IC."""
    risk_cache = {}
    scores = alpha_model.scores(model, dates) if model != "none" else {d: None for d in dates}
    rows, ics = [], []
    for d in dates:
        sub = frame.iloc[alpha_model.rows[d]]
        syms = sub.symbol.to_numpy()
        if d not in risk_cache:
            risk_cache[d] = risk_model(R, d, syms)
        risk = risk_cache[d]
        s = scores[d]
        if s is None:
            alpha = np.zeros(len(syms))
        else:
            z = zscore_within_sector(s, sub.sector.to_numpy())
            alpha = IC * sub.volatility_63.to_numpy() * np.sqrt(90 / 252) * z * (365 / 90)
            ok = ~np.isnan(sub.y.to_numpy())
            if ok.sum() > 60:
                ics.append(pd.Series(s[ok]).corr(pd.Series(sub.y.to_numpy()[ok]), method="spearman"))
        weights = optimize(np.nan_to_num(alpha), risk, lam)
        kept = syms[risk[0]]
        m = weights > 1e-4
        weights = weights[m] / weights[m].sum()
        entry = dict(zip(sub.symbol, sub.entry_date))
        rows += [(a, d, entry[a], x) for a, x in zip(kept[m], weights)]
    picks = pd.DataFrame(rows, columns=["symbol", "signal_date", "entry_date", "weight"])
    return picks[picks.entry_date.notna()].reset_index(drop=True), (float(np.mean(ics)) if ics else np.nan)


# evaluation

def evaluate(picks, window, **kw):
    eq, log, stale = h.run(picks, window, **kw)
    bench = h.bench_curve(*window)
    s, b = h.summary(eq), h.summary(bench)
    active = eq.pct_change().dropna() - bench.pct_change().reindex(eq.pct_change().dropna().index)
    te = active.std() * np.sqrt(252)
    s.update(active_cagr=s["cagr"] - b["cagr"], tracking_error=te, info_ratio=active.mean() * 252 / te)
    return s, b, eq, log, stale


def signal_dates(frame, window):
    d = frame.date.drop_duplicates()
    if window[0] == h.TEST[0]:
        d = d[d >= d[d < h.TEST[0]].iloc[-1]]  # start from the last signal before the first test close
    return d[(d >= window[0]) & (d <= window[1])]


def accounting_audit(picks, log, eq):
    """Plain checks on the headline run. Returns a list of text lines."""
    lines = []
    ok = (picks.entry_date > picks.signal_date).all()
    lines.append(f"Every purchase is filled after its signal date. {'Passed' if ok else 'FAILED'}.")
    first = log[(log.side == "BUY") & (log.date == log.date.min())]
    spent = (first.shares * first.price).sum() * (1 + h.COST)
    lines.append(f"First day buys cost {spent:,.0f} including costs. One thirteenth of $1,000,000 is {1e6 / 13:,.0f}.")
    holds = []
    for _, g in log.groupby("symbol"):
        buys, sells = g[g.side == "BUY"].date.tolist(), g[g.side == "SELL"].date.tolist()
        holds += [(s - b).days for b, s in zip(buys, sells)]
    lines.append(f"Holding periods run from {min(holds)} to {max(holds)} calendar days over {len(holds):,} closed positions.")
    lines.append(f"Final value {eq.iloc[-1]:,.0f}. The backtester stops if cash ever goes negative and it never did.")
    lines.append("Training rows only used outcomes finished by each fit date. The code asserts this at every refit.")
    return lines


def plot(curves, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(curves.index, curves.iloc[:, 0] / curves.iloc[0, 0] - 1, color="black", lw=2, label="S&P 500 total return")
    ax.plot(curves.index, curves.iloc[:, 1] / curves.iloc[0, 1] - 1, color="#1f77b4", lw=2, label="My strategy")
    ax.set_title("Cumulative return, April 2016 to September 2026 (10 bps per side, 2% cash)")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)


# stages

def dev():
    frame, R = load_core(), h.daily_returns()
    am = Alpha(frame)
    dates = signal_dates(frame, h.DEV)
    rows = []
    for model, lam in [("none", 10.0)] + [(m, l) for m in MODELS for l in LAMS]:
        picks, ic = build_picks(frame, am, model, lam, dates, R)
        s, b, eq, log, stale = evaluate(picks, (picks.entry_date.min(), h.DEV[1]))
        rows.append(dict(model=model, lam=lam, oos_ic=ic, **{k: s[k] for k in [
            "cagr", "sharpe", "max_drawdown", "ann_vol", "active_cagr", "tracking_error", "info_ratio"]}))
        print(rows[-1], flush=True)
    tab = pd.DataFrame(rows)
    # My rule, written down before looking. Best Sharpe among the alpha models with tracking error of 6% or less.
    eligible = tab[(tab.model != "none") & (tab.tracking_error <= 0.06)]
    pick = eligible.sort_values("sharpe", ascending=False).iloc[0]
    _, premia = am.fm_premia(pd.Timestamp("2015-12-31"))
    h.RESULTS.mkdir(exist_ok=True)
    tab.round(4).to_csv(h.RESULTS / "dev_selection.csv", index=False)
    premia.round(4).to_csv(h.RESULTS / "fm_premia.csv")
    json.dump(dict(model=pick.model, lam=float(pick.lam), benchmark_dev=b), open(h.OUT / "benchrel_choice.json", "w"),
              indent=1, default=float)
    pd.options.display.float_format = "{:,.3f}".format
    print(tab.to_string(index=False))
    print("benchmark on the same window", {k: round(v, 3) for k, v in b.items()})
    print("choice", pick.model, pick.lam)


def test():
    choice = json.load(open(h.OUT / "benchrel_choice.json"))
    frame, R = load_core(), h.daily_returns()
    am = Alpha(frame)
    dates = signal_dates(frame, h.TEST)
    setups = {"My strategy": (choice["model"], choice["lam"]), "Tracker with no alpha": ("none", choice["lam"]),
              **{f"Alternative alpha {m}": (m, choice["lam"]) for m in MODELS if m != choice["model"]}}
    table, extras = {}, {}
    for name, (model, lam) in setups.items():
        picks, ic = build_picks(frame, am, model, lam, dates, R)
        s, b, eq, log, stale = evaluate(picks, h.TEST)
        table[name] = dict(oos_ic=ic, stale_exits=int(stale), **s)
        print(name, {k: round(float(s[k]), 4) for k in ["cagr", "sharpe", "max_drawdown", "tracking_error"]}, flush=True)
        if name == "My strategy":
            table["My strategy, no costs"] = evaluate(picks, h.TEST, cost=0.0)[0]
            table["My strategy, 50% haircut on stale exits"] = evaluate(picks, h.TEST, haircut=0.5)[0]
            curves = pd.DataFrame({"S&P 500 TR": h.bench_curve(*h.TEST), "My strategy": eq})
            extras = dict(years=h.yearly(eq), audit=accounting_audit(picks, log, eq), bench=b)
    table = {"S&P 500 TR": extras["bench"], **table}
    h.RESULTS.mkdir(exist_ok=True)
    pd.DataFrame(table).T.round(4).to_csv(h.RESULTS / "performance_comparison.csv")
    curves.to_csv(h.RESULTS / "cumulative_returns.csv")
    plot(curves, h.RESULTS / "cumulative_returns.png")
    yr = curves.resample("YE").last()
    yr.loc[curves.index[0]] = curves.iloc[0]
    (yr.sort_index().pct_change().dropna() * 100).round(1).rename(index=lambda t: t.year).to_csv(h.RESULTS / "yearly_returns.csv")
    open(h.RESULTS / "accounting_checks.txt", "w").write("\n".join(extras["audit"]) + "\n")
    print("\n".join(extras["audit"]))


if __name__ == "__main__":
    {"dev": dev, "test": test}[sys.argv[1]]()
