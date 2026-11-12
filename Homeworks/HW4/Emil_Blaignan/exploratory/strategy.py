"""Random-feature Markowitz stock scoring (Chernov-Kelly-Malamud-Schwab / Didisheim et al. style).

Construction (CKMS eq. 45), applied to 90-day returns on a weekly signal cadence:
  1. Each week t, rank-normalize the d characteristics across tradable stocks to [-1, 1].
  2. P random ReLU "weights":  f_p(x) = ReLU(varsigma * sqrt(2/d) * x . w_p),  w_p ~ N(0, I).
  3. Managed portfolio p earns  R_p(t) = mean_i  y_i(t) * f_p(x_i(t)),  where y is the stock's
     90-day return minus that week's cross-sectional median (clipped to dev-window quantiles).
  4. Ridge-Markowitz over the P managed portfolios, time = sample:
        pi = (zI + A'A/T)^-1 mu_hat, solved in the T x T dual because P >> T.
  5. The implied stock weight is  score_i = sum_p pi_p f_p(x_i).  We hold the top-N scores
     long-only (the paper's weights are long-short, which HW4 forbids).
Every fit uses only weeks whose 90-day holding period has already ended.
"""
import numpy as np
import pandas as pd

FEATS = ["momentum_21", "momentum_63", "momentum_252_21", "volatility_63", "drawdown_252",
         "net_margin", "operating_roa", "debt_to_assets", "cash_to_assets", "fcf_to_assets",
         "accruals_to_assets"]
D = len(FEATS)
MATURE_DAYS = 98  # 90-day hold + up to 5 sessions of exit delay


class Weeks:
    """Weekly cross-sections with rank-normalized characteristics, indexed by signal date."""

    def __init__(self, panel):
        panel = panel.sort_values(["date", "symbol"]).reset_index(drop=True)
        for f in FEATS:
            panel[f + "_x"] = (panel.groupby("date")[f].rank(pct=True) * 2 - 1).fillna(0.0)
        self.panel = panel
        self.X = panel[[f + "_x" for f in FEATS]].to_numpy(np.float32)
        bounds = np.flatnonzero(np.r_[True, panel.date.values[1:] != panel.date.values[:-1], True])
        self.dates = pd.DatetimeIndex(panel.date.values[bounds[:-1]])
        self.slices = dict(zip(self.dates, zip(bounds[:-1], bounds[1:])))

    def rows(self, d):
        a, b = self.slices[d]
        return a, b


def make_features(kind, P=1000, seed=0, scale=0.5):
    """Feature map x -> F. 'rf' = random ReLU; 'linear' = the characteristics themselves."""
    if kind == "linear":
        return lambda X: X
    w = np.random.default_rng(seed).standard_normal((D, P)).astype(np.float32)
    c = np.float32(scale * np.sqrt(2.0 / D))
    return lambda X: np.maximum(X @ w * c, 0.0)


class FactorModel:
    def __init__(self, weeks, feat, clip):
        self.w, self.feat, self.clip = weeks, feat, clip
        self.R = {}  # signal date -> managed-portfolio returns (P,)
        tgt = weeks.panel.target.to_numpy()
        for d in weeks.dates:
            a, b = weeks.rows(d)
            y = tgt[a:b]
            ok = ~np.isnan(y)
            if ok.sum() < 50:
                continue
            y = np.clip(y[ok] - np.median(y[ok]), *clip)
            F = feat(weeks.X[a:b][ok])
            self.R[d] = (y[:, None] * F).mean(0)

    def fit(self, fit_date, lam):
        """pi from weeks whose holding period ended by fit_date. lam is a fraction of mean energy."""
        ok = [d for d in self.R if d <= fit_date - pd.Timedelta(days=MATURE_DAYS)]
        A = np.stack([self.R[d] for d in ok]).astype(np.float64)
        A /= np.sqrt((A ** 2).mean())
        G = A @ A.T
        z = lam * np.trace(G) / len(G)
        alpha = np.linalg.solve(G + z * np.eye(len(G)), np.ones(len(G)))
        return A.T @ alpha, len(ok)

    def score(self, d, pi):
        a, b = self.w.rows(d)
        return self.feat(self.w.X[a:b]) @ pi.astype(np.float32)


def quarter_refits(dates):
    """First signal date of each calendar quarter."""
    q = dates.to_period("Q")
    return dates[np.r_[True, q[1:] != q[:-1]]]


def walk_forward_scores(weeks, scorer, dates):
    """scorer(fit_date) -> callable(date)->scores. Refit at the first signal date of each quarter."""
    refits = set(quarter_refits(dates))
    out, cur = {}, None
    for d in dates:
        if d in refits or cur is None:
            cur = scorer(d)
        out[d] = cur(d)
    return out


def rank_ic(weeks, scores):
    """Per-date Spearman correlation of score with the (median-demeaned) 90d target."""
    tgt = weeks.panel.target.to_numpy()
    ics = {}
    for d, s in scores.items():
        a, b = weeks.rows(d)
        y = tgt[a:b]
        ok = ~np.isnan(y)
        if ok.sum() < 100:
            continue
        ics[d] = np.corrcoef(pd.Series(s[ok]).rank(), pd.Series(y[ok]).rank())[0, 1]
    return pd.Series(ics)


def top_n_spread(weeks, scores, n=100):
    """Mean 90d target of the top-n names minus the cross-sectional mean, per date."""
    tgt = weeks.panel.target.to_numpy()
    out = {}
    for d, s in scores.items():
        a, b = weeks.rows(d)
        y = tgt[a:b]
        ok = ~np.isnan(y)
        if ok.sum() < 200:
            continue
        top = np.argsort(-s[ok])[:n]
        out[d] = np.median(y[ok][top]) - np.median(y[ok])
    return pd.Series(out)


def make_picks(weeks, scores, n=100):
    """Top-n tradable names per signal date; unfillable names are dropped, never replaced."""
    rows = []
    sym = weeks.panel.symbol.to_numpy()
    ent = weeks.panel.entry_date.to_numpy()
    for d, s in scores.items():
        a, b = weeks.rows(d)
        for j in np.argsort(-s)[:n]:
            rows.append((sym[a + j], d, ent[a + j]))
    p = pd.DataFrame(rows, columns=["symbol", "signal_date", "entry_date"])
    return p[p.entry_date.notna()].reset_index(drop=True)
