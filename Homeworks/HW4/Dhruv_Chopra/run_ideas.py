"""Development tests of my own feature hypotheses. 2010-2015 data only.

    ../../../.venv/bin/python run_ideas.py                    # every group
    ../../../.venv/bin/python run_ideas.py --group dividends  # one group

Groups (features.FEATURE_GROUPS):
  skew           daily versions of my intraday-skew research signals,
                 including skew of the previous 180 returns
  dividends      dividends as a sign of company health
  issuance       share offerings and buybacks
  decomposition  return split into market, industry and stock-specific parts
  health         balance-sheet ratios: operating margin, cash vs debt, liquidity
  long_mom       long-term momentum: 1-year average return, 2- and 3-year reversal
  moments        mean, volatility, skew and kurtosis of the past year's daily returns
  resid_sq       squared residual returns after removing market and sector effects

(Stock splits could not be tested: prices and share counts in the database
are restated for later splits, so a split leaves no trace on its date. See
README.)

The checks are the ones I used in the skewness_signal research
(research/model_selection_report.md):
  * the expected sign is written down *before* running (PRIOR below);
  * single-feature rank IC, its t-stat and top-30 excess return, by year;
  * neighbouring settings (window lengths / thresholds), to find a plateau
    rather than a lucky single cell;
  * placebo features: random noise with the same persistence;
  * "without the best N names" and random halves of the universe;
  * walk-forward Ridge: v1 vs v1 + each group / each feature;
  * development backtests with the same portfolio rules.

Writes results/<group>_*.csv and results/ideas_*.csv.
"""

import argparse

import numpy as np
import pandas as pd

from features import FEATURE_GROUPS, FEATURES, _frame, decomposition_features, idea_features, long_momentum
from model import fit, walk_forward
from pipeline import DEV_BACKTEST, OUTPUTS, RESULTS, compare, run_strategy, setup

DEV_END = pd.Timestamp("2015-12-31")
VALID_YEARS = [2012, 2013, 2014, 2015]   # same validation years as run_development.py
SIGN_YEARS = [2010, 2011]                # used to set a sign only when I had no prior
OVERLAP = 13                             # weekly samples of a 90-day label overlap ~13 weeks
rng = np.random.default_rng(2026)

# Expected direction, written down before running. +1 = buy high values; None = no prior.
PRIOR = {
    "skew": (+1, "long high skew: my prior and Strategy A's direction (report D4)"),
    "kurt": (None, "no prior"),
    "dist_high": (-1, "k-day-high reversal (report 8.1)"),
    "clv": (None, "no prior (my test used a near-median book)"),
    "div_payer": (+1, "paying dividends = healthy"),
    "div_to_assets": (+1, "larger payout = confident management"),
    "div_payout_cfo": (-1, "payout stretched relative to cash flow = fragile"),
    "div_payout_cfo_clip1": (-1, "same, clipped at 1"),
    "div_change_yoy": (+1, "raises good, cuts and omissions bad"),
    "div_cut": (-1, "a cut signals distress (1 = cut)"),
    "net_issuance_yoy": (-1, "issuers underperform (Loughran-Ritter, Pontiff-Woodgate)"),
    "issued_to_assets": (-1, "issuers underperform"),
    "buyback_to_assets": (+1, "buybacks signal undervaluation (Ikenberry et al.)"),
    "net_payout_to_assets": (+1, "total shareholder yield"),
    "beta_mkt": (None, "no prior (low-beta anomaly vs CAPM)"),
    "resid_mom": (+1, "residual momentum (Blitz-Huij-Martens)"),
    "ind_mom": (+1, "industry momentum (Moskowitz-Grinblatt)"),
    "idio_vol": (-1, "idiosyncratic volatility puzzle (Ang et al.)"),
    "r2": (None, "no prior"),
    "operating_margin": (+1, "pricing power / cost control"),
    "cash_to_ltd": (+1, "cash covers long-term debt = safer"),
    "net_cash_to_assets": (+1, "net cash = safer"),
    "current_ratio": (+1, "liquidity (weak prior)"),
    "interest_coverage": (+1, "can service its debt"),
    "st_debt_share": (-1, "refinancing risk"),
    "avg_ret": (+1, "momentum (incl. the last month)"),
    "vol_": (-1, "low-volatility anomaly"),
    "sec_resid_sq": (-1, "high stock-specific variance -> lower returns (Ang et al.)"),
    "mom_504": (-1, "long-term reversal (De Bondt-Thaler)"),
    "mom_756": (-1, "long-term reversal"),
    "mom_1260": (-1, "long-term reversal"),
}


def prior_for(name):
    for key in sorted(PRIOR, key=len, reverse=True):     # longest matching prefix wins
        if name.startswith(key):
            return PRIOR[key]
    raise KeyError(name)


def placebo_window(name):
    """Persistence of a feature in trading days (for the matching placebo)."""
    if name.startswith(("div_", "net_", "issued", "buyback", "operating_margin", "cash_to_ltd",
                        "current_ratio", "interest_coverage", "st_debt")):
        return 63                                       # changes once a quarter
    nums = [int(x) for x in name.split("_") if x.isdigit()]
    return nums[0] if nums else 63


# ---------------------------------------------------------------- helpers
def ic_by_date(df, col, min_names=20):
    """Spearman IC per date, vectorised (ranks, then Pearson within each date)."""
    g = df[["date", col, "target"]].dropna()
    r = g.groupby("date")[[col, "target"]].rank()
    r["date"] = g["date"].to_numpy()
    r = r[r.groupby("date")[col].transform("size") >= min_names]
    d = r[[col, "target"]] - r.groupby("date")[[col, "target"]].transform("mean")
    d["date"] = r["date"]
    d["xy"], d["xx"], d["yy"] = d[col] * d["target"], d[col] ** 2, d["target"] ** 2
    s = d.groupby("date")[["xy", "xx", "yy"]].sum()
    return s.xy / np.sqrt(s.xx * s.yy)


def t_stat(ic):
    return ic.mean() / ic.std() * np.sqrt(len(ic) / OVERLAP)


def top_excess(df, col, n=30):
    """Per-date mean return of the top-n by `col` minus the date's tradeable mean."""
    s = df[["date", "n", col, "target"]].dropna(subset=["target"])
    s = s.assign(excess=s.target - s.groupby("date").target.transform("mean"))
    s = s.dropna(subset=[col])
    s["r"] = s.groupby("date")[col].rank(ascending=False, method="first")
    top = s[s.r <= n]
    return top.groupby("date").excess.mean(), top


def without_best(df, col, k, n=30):
    """Top-n excess after removing the k stocks that contributed most to it."""
    _, top = top_excess(df, col, n)
    contrib = top.groupby("n").excess.sum().sort_values(ascending=False)
    return top_excess(df[~df.n.isin(contrib.index[:k])], col, n)[0].mean()


def random_halves(df, col, draws=200):
    names = df.n.unique()
    out = []
    for _ in range(draws):
        half = rng.choice(names, size=len(names) // 2, replace=False)
        out.append(ic_by_date(df[df.n.isin(half)], col).mean())
    return np.array(out)


# ---------------------------------------------------------------- raw variants per group
def variants(group, data, feats, t_idx, n_idx):
    """Raw (unranked) values at the panel rows for each group's features and
    their neighbouring settings. Spearman IC does not need ranks."""
    at = lambda a: a[t_idx, n_idx].astype(float)
    out = {}
    if group == "skew":
        px = _frame(data.adj_close, data).astype("float64")
        ret = px.pct_change(fill_method=None)
        grid = {"skew_windows": (42, 63, 126, 180, 252), "kurt_windows": (63, 252),
                "high_windows": (10, 20, 40), "clv_windows": (5, 21, 63)}
        for arg, ks in grid.items():
            for k in ks:
                kw = {a: () for a in grid}
                kw[arg] = (k,)
                (name, frame), = idea_features(px, ret, data, **kw).items()
                out[name] = frame.to_numpy(np.float32)[t_idx, n_idx].astype(float)
    elif group == "dividends":
        for f in FEATURE_GROUPS["dividends"]:
            out[f] = at(feats[f])
        out["div_payout_cfo_clip1"] = np.minimum(out["div_payout_cfo"], 1.0)
        for thr in (0.10, 0.25, 0.50):                 # neighbourhood of "what counts as a cut"
            c = out["div_change_yoy"]
            out[f"div_cut_{int(thr * 100)}"] = np.where(np.isfinite(c), (c <= -thr).astype(float), np.nan)
    elif group == "issuance":
        for f in FEATURE_GROUPS["issuance"]:
            out[f] = at(feats[f])
    elif group == "decomposition":
        for f in FEATURE_GROUPS["decomposition"]:
            out[f] = at(feats[f])
        for w in (126, 504):                            # neighbouring regression windows
            for name, a in decomposition_features(data, window=w, ind_window=126).items():
                if name not in out:
                    out[name] = at(a)
                del a
    elif group == "health":
        for f in FEATURE_GROUPS["health"]:
            out[f] = at(feats[f])
        out["interest_coverage_clip10"] = np.minimum(out["interest_coverage"], 10.0)   # neighbour
    elif group == "resid_sq":
        sq = decomposition_features(data, peers="sector", resid_sq_windows=(5, 21, 63, 126))
        for name, a in sq.items():
            if "resid_sq" in name:
                out[name] = at(a)
        del sq
        for f in FEATURE_GROUPS["resid_sq"]:
            out.setdefault(f, at(feats[f]))
    elif group == "moments":
        px = _frame(data.adj_close, data).astype("float64")
        ret = px.pct_change(fill_method=None)
        for k in (126, 252, 504):
            mp = int(0.8 * k)
            for name, frame in [(f"avg_ret_{k}", ret.rolling(k, min_periods=mp).mean()),
                                (f"vol_{k}", ret.rolling(k, min_periods=mp).std()),
                                (f"skew_{k}", ret.rolling(k, min_periods=mp).skew()),
                                (f"kurt_{k}", ret.rolling(k, min_periods=mp).kurt())]:
                out[name] = frame.to_numpy(np.float32)[t_idx, n_idx].astype(float)
    elif group == "long_mom":
        for name, a in long_momentum(data, lags=((24, 12), (36, 12), (60, 12)),
                                     avg_windows=(126, 252, 504)).items():
            out[name] = at(a)
    return out


def placebo_table(tr, n_symbols, windows):
    weeks = np.sort(tr.date.unique())
    w_of = pd.Series(np.arange(len(weeks)), index=weeks)
    w_rows, n_rows = w_of[tr.date].to_numpy(), tr.n.to_numpy()
    valid = tr.date.dt.year.isin(VALID_YEARS).to_numpy()
    rows = {}
    for window in sorted(windows):
        m = max(1, round(window / 5))            # weekly samples spanned by the window
        draws = []
        for _ in range(100):
            noise = rng.standard_normal((len(weeks) + m, n_symbols)).astype(np.float32)
            smooth = pd.DataFrame(noise).rolling(m).mean().to_numpy()[m:]
            frame = tr[["date", "target"]].assign(p=smooth[w_rows, n_rows])[valid]
            draws.append(ic_by_date(frame, "p").mean())
        draws = np.abs(draws)
        rows[window] = {"mean_abs_IC": draws.mean(), "p95_abs_IC": np.quantile(draws, 0.95),
                        "max_abs_IC": draws.max()}
    return pd.DataFrame(rows).T.rename_axis("window_days")


def event_table(tr, col, buckets, fresh=None):
    """Mean 90-day excess return (vs the date's tradeable mean) by bucket."""
    d = tr.assign(excess=tr.target - tr.groupby("date").target.transform("mean"))
    d = d[d.date.dt.year.isin(VALID_YEARS) & d.excess.notna()]
    rows = []
    for label, mask in buckets(d[col]).items():
        for scope, sub in [("all weeks", d[mask]), ("first week after filing", d[mask & fresh(d)])] if fresh else \
                [("all weeks", d[mask])]:
            per_date = sub.groupby("date").excess.mean()
            rows.append({"bucket": label, "scope": scope, "stock_weeks": len(sub),
                         "names": sub.n.nunique(), "mean_excess_90d": sub.excess.mean(),
                         "t_stat_dates": per_date.mean() / per_date.std() * np.sqrt(len(per_date) / OVERLAP)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser()
    all_groups = ["skew", "dividends", "issuance", "decomposition", "health", "long_mom", "moments", "resid_sq"]
    parser.add_argument("--group", action="append", choices=all_groups)
    parser.add_argument("--base", choices=["v1", "skew_180"], default="v1",
                        help="model the groups are added to (v1, or v1 + skew_180)")
    parser.add_argument("--weighting", action="append", choices=["equal", "rank", "inverse_vol"],
                        help="backtest weighting(s); default equal")
    parser.add_argument("--skip-backtests", action="store_true")
    args = parser.parse_args()
    groups = args.group or all_groups
    weightings = args.weighting or ["equal"]
    base_label = "v1" if args.base == "v1" else "v1 + skew_180"
    base_cols = FEATURES if args.base == "v1" else FEATURES + FEATURE_GROUPS["skew_180"]
    tag = "" if args.base == "v1" and groups == all_groups else "_" + "_".join(groups) + f"_on_{args.base}"

    OUTPUTS.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    data, feats, target, label_end, panel = setup()
    dev = panel.loc[panel.date.le(DEV_END)].copy()
    del panel
    t_idx, n_idx = dev.t.to_numpy(), dev.n.to_numpy()
    tradeable = dev.tradeable.to_numpy()

    placebo_windows = set()
    singles = {}
    for group in groups:
        raw = variants(group, data, feats, t_idx, n_idx)
        tr = dev.loc[tradeable, ["date", "t", "n", "target"]].copy()
        for name, values in raw.items():
            tr[name] = values[tradeable]
        tr["filing_age"] = feats["filing_age_days"][t_idx, n_idx][tradeable]
        valid = tr[tr.date.dt.year.isin(VALID_YEARS)]
        early = tr[tr.date.dt.year.isin(SIGN_YEARS)]

        # ---- signed single-feature IC, top-30 excess, coverage, robustness ----
        rows = []
        for name in raw:
            prior, why = prior_for(name)
            early_ic = ic_by_date(early, name).mean()
            sign = prior if prior is not None else int(np.sign(early_ic))
            ic_all = ic_by_date(tr, name)
            ic = ic_all[ic_all.index.year.isin(VALID_YEARS)] * sign
            frame = valid.assign(s=valid[name] * sign)
            halves = random_halves(frame, "s", draws=100)
            placebo_windows.add(placebo_window(name))
            rows.append({"feature": name, "in_model_group": name in sum(FEATURE_GROUPS.values(), []),
                         "prior_sign": prior, "sign_used": sign, "why": why,
                         "coverage": np.isfinite(valid[name]).mean(),
                         "IC_2012_2015_signed": ic.mean(), "IC_t_stat": t_stat(ic),
                         "placebo_window": placebo_window(name),
                         "top30_excess_90d": top_excess(frame, "s")[0].mean(),
                         "without_best_5": without_best(frame, "s", 5),
                         "without_best_10": without_best(frame, "s", 10),
                         "halves_share_IC_pos": (halves > 0).mean(),
                         "IC_2010_2011_raw": early_ic,
                         **{f"IC_{y}_raw": ic_all[ic_all.index.year == y].mean() for y in range(2010, 2016)}})
        single = pd.DataFrame(rows).set_index("feature")
        singles[group] = single
        print(f"\n=== {group}: single features (signed IC uses sign_used) ===\n",
              single.drop(columns=["why", "in_model_group"]).round(4).to_string())
        single.to_csv(RESULTS / f"{group}_single_features.csv")

        # ---- event tables: the economic story, read directly ----
        if group == "dividends":
            def div_buckets(c):
                payer = tr.loc[c.index, "div_payer"].eq(1)      # pays now (TTM > 0)
                return {"omission (-100%)": c.eq(-1),
                        "cut (-10% or worse)": c.gt(-1) & c.le(-0.10),
                        "flat (+/-10%)": c.gt(-0.10) & c.lt(0.10) & payer,
                        "raise (+10% or more)": c.ge(0.10) & c.lt(1),
                        "initiation (+1)": c.eq(1),
                        "never paid": c.eq(0) & ~payer}
            ev = event_table(tr, "div_change_yoy", div_buckets, fresh=lambda d: d.filing_age.le(7))
        elif group == "issuance":
            ev = event_table(tr, "net_issuance_yoy", lambda c: {
                "big buyback (< -5%)": c.lt(-0.05), "buyback (-5%..-1%)": c.ge(-0.05) & c.lt(-0.01),
                "flat (+/-1%)": c.ge(-0.01) & c.le(0.01), "issuance (1%..10%)": c.gt(0.01) & c.le(0.10),
                "big issuance (> 10%)": c.gt(0.10)}, fresh=lambda d: d.filing_age.le(7))
        else:
            ev = None
        if ev is not None:
            print(f"\n{group}: 90-day excess return by event bucket, 2012-2015:\n", ev.round(4).to_string())
            ev.to_csv(RESULTS / f"{group}_events.csv", index=False)
        del tr, valid, early, raw

    # ---- placebo |IC| for the persistence of every feature tested ----
    plac = placebo_table(dev.loc[tradeable, ["date", "n", "target"]], len(data.symbols), placebo_windows)
    print("\nPlacebo |IC| 2012-2015 (100 noise features per persistence window):\n", plac.round(4).to_string())
    plac.to_csv(RESULTS / f"ideas_placebo{tag}.csv")
    for group, single in singles.items():
        p95 = single.placebo_window.map(plac.p95_abs_IC)
        single["beats_placebo_p95"] = single.IC_2012_2015_signed.abs() > p95
        single.to_csv(RESULTS / f"{group}_single_features.csv")
        print(f"{group}: beats placebo 95th pct ->",
              single.index[single.beats_placebo_p95].tolist())

    # ---- walk-forward Ridge: v1 vs v1 + each group / feature ----
    model_groups = {"skew": ["skew", "skew_180"]}
    sets = {"v1 (current)": FEATURES}
    if args.base != "v1":
        sets[base_label] = base_cols
    for group in groups:
        for g in model_groups.get(group, [group]):
            sets[f"{base_label} + {g}"] = base_cols + [f for f in FEATURE_GROUPS[g] if f not in base_cols]
            for f in FEATURE_GROUPS[g]:
                if f not in base_cols:
                    sets.setdefault(f"{base_label} + {f}", base_cols + [f])
    new = [f for g in groups for gg in model_groups.get(g, [g]) for f in FEATURE_GROUPS[gg] if f not in base_cols]
    sets[f"{base_label} + all tested groups"] = base_cols + list(dict.fromkeys(new))
    wf_rows = []
    for label, cols in sets.items():
        wf = walk_forward("ridge", dev, data, VALID_YEARS, features=cols)
        wf = wf[wf.tradeable]
        ic = ic_by_date(wf.rename(columns={"score": "sc"}), "sc")
        ex = top_excess(wf, "score")[0]
        wf_rows.append({"features": label, "rank_IC": ic.mean(), "IC_t_stat": t_stat(ic),
                        "top30_excess_90d": ex.mean(),
                        "without_best_10": without_best(wf, "score", 10),
                        **{f"top30_excess_{y}": ex[ex.index.year == y].mean() for y in VALID_YEARS}})
    wft = pd.DataFrame(wf_rows).set_index("features")
    print("\nWalk-forward Ridge 2012-2015 (tradeable):\n", wft.round(4).to_string())
    wft.to_csv(RESULTS / f"ideas_walk_forward{tag}.csv")

    cols = sets[f"{base_label} + all tested groups"]
    last = fit("ridge", dev, int(dev.loc[dev.date.ge("2015-01-01"), "t"].min()), cols)
    coef = pd.Series(last.model.coef_[:len(cols)], index=cols)
    print("\nRidge coefficients, v1 + all tested groups (last development fit):\n", coef.round(4).to_string())
    coef.rename("coefficient").to_csv(RESULTS / f"ideas_ridge_coefficients_all{tag}.csv")

    if args.skip_backtests:
        return
    # ---- development backtests, same portfolio rules ----
    bt = []
    group_sets = {k: v for k, v in sets.items() if k in ("v1 (current)", base_label)
                  or k.endswith("all tested groups") or k.removeprefix(f"{base_label} + ") in FEATURE_GROUPS}
    # Equal weight without redistribution reproduces the original v1 exactly;
    # other weightings use the cap redistribution (fully invested, fair comparison).
    for weighting in weightings:
        for label, cols in group_sets.items():
            result, _, _, nav, bench = run_strategy(data, feats, dev, *DEV_BACKTEST, features=cols,
                                                    weighting=weighting, redistribute=weighting != "equal")
            name = f"{label} [{weighting}]"
            m = compare(nav, bench, name)
            diag = result.daily.iloc[1:]
            bt.append(m.iloc[:, 0].rename(name).to_frame().T.assign(
                avg_cash_fraction=(diag.cash / diag.nav).mean(), fees=result.fills.fee.sum()))
            bench_row = m.iloc[:, 1]
            print(f"{name}: CAGR {m.iloc[:, 0].CAGR:.2%}, Sharpe {m.iloc[:, 0].Sharpe:.2f}, "
                  f"max drawdown {m.iloc[:, 0].max_drawdown:.2%}, cash {(diag.cash / diag.nav).mean():.1%}",
                  flush=True)
    bt.append(bench_row.rename("S&P 500 TR").to_frame().T)
    btt = pd.concat(bt)
    print("\nDevelopment backtests", DEV_BACKTEST, ":\n", btt.to_string())
    btt.to_csv(RESULTS / f"ideas_dev_backtests{tag}.csv")


if __name__ == "__main__":
    main()
