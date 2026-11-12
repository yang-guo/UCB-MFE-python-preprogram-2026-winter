"""Regime barbell: index-like large-liquid core <-> minimum-variance ballast.

    .venv/bin/python Homeworks/HW4/Emil_Blaignan/barbell.py sleeves dev|test    # build + cache the sleeves
    .venv/bin/python Homeworks/HW4/Emil_Blaignan/barbell.py dev  <candidate>     # validate on 2011-2015 only
    .venv/bin/python Homeworks/HW4/Emil_Blaignan/barbell.py test <candidate>     # final test, run once

Sleeves (each is a full set of weekly long-only picks, weights sum to 1):
  A50 / A100 / AEQ50  the 50 / 100 most-traded tradable stocks (trailing 63-day median dollar volume), weighted by
                      dollar volume (a cap-weight proxy, each weight <= 5%) or equally. Index-like, broad, steady.
  B                   minimum-variance portfolio of the 100 most-traded stocks (cap 3%, ridge-shrunk covariance).
  C                   ridge-selected momentum/quality tilt inside the 150 most-traded stocks, risk-weighted.
Mix: each tranche is a weighted blend of sleeves. A "regime" mix uses one set of weights when the S&P 500 TR is above
its 200-day average at the signal close and another when it is below (rotation, never cash).
Signal at the close, fill at the next close, 90-day hold, 13 overlapping tranches; cash earns 2%, costs 10 bps/side.
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import blend  # noqa: E402
import core  # noqa: E402
import minvar as mv  # noqa: E402

OUT = mv.OUT
DEV, TEST = blend.DEV, mv.TEST
SLEEVES = {
    "A50":   dict(K=50, N=50, sel="adv", wt="advw", cap=0.05),
    "A100":  dict(K=100, N=100, sel="adv", wt="advw", cap=0.05),
    "AEQ50": dict(K=50, N=50, sel="adv", wt="eq"),
    "B":     dict(K=100, N=100, sel="adv", wt="mv", cap=0.03, shrink=0.5),
    "C":     blend.tilt("riskw", 0.05),
}
# mix = [(sleeve, weight when uptrend, weight when downtrend)]
CANDS = {
    "cap50":          [("A50", 1, 1)],
    "cap100":         [("A100", 1, 1)],
    "eq50":           [("AEQ50", 1, 1)],
    "mv100":          [("B", 1, 1)],
    "static_70_30":   [("A50", .7, .7), ("B", .3, .3)],
    "static_50_50":   [("A50", .5, .5), ("B", .5, .5)],
    "regime_80_20":   [("A50", .8, .2), ("B", .2, .8)],
    "regime_100_0":   [("A50", 1, 0), ("B", 0, 1)],
    "regime_a100":    [("A100", .8, .2), ("B", .2, .8)],
    "regime_tilt":    [("A50", .56, .14), ("B", .14, .56), ("C", .3, .3)],
}


def build_sleeves(window_tag):
    dates_rng = DEV if window_tag == "dev" else TEST
    w, core_df = core.load()
    R = mv.daily_returns()
    d = w.date.drop_duplicates()
    if window_tag == "test":
        d = d[d >= d[d < TEST[0]].iloc[-1]]
    d = d[(d >= dates_rng[0]) & (d <= dates_rng[1])]
    out = {k: core.build_picks(w, core_df, R, c, d)[0] for k, c in SLEEVES.items()}
    pickle.dump(out, open(OUT / f"sleeves_{window_tag}.pkl", "wb"))
    print({k: len(v) for k, v in out.items()})


def uptrend(signal_dates):
    b = pd.read_parquet(OUT / "benchmark.parquet").set_index("date").adj_close
    ok = b >= b.rolling(200, min_periods=200).mean()
    return ok.reindex(signal_dates).fillna(True)


def mix(sleeves, spec):
    sig = sorted(set().union(*[set(s.signal_date) for s in sleeves.values()]))
    up = uptrend(pd.DatetimeIndex(sig))
    parts = []
    for name, w_up, w_dn in spec:
        p = sleeves[name].copy()
        m = np.where(up.reindex(p.signal_date).to_numpy(), w_up, w_dn)
        p["weight"] = p.weight.to_numpy() * m
        parts.append(p[p.weight > 0])
    return pd.concat(parts, ignore_index=True)


def run(name, window_tag):
    sleeves = pickle.load(open(OUT / f"sleeves_{window_tag}.pkl", "rb"))
    picks = mix(sleeves, CANDS[name])
    window = (picks.entry_date.min(), DEV[1]) if window_tag == "dev" else TEST
    eq, log, stale = blend.run(picks, None, window)
    return picks, eq, log, stale, window


def dev(name):
    picks, eq, log, stale, window = run(name, "dev")
    res = dict(spec=CANDS[name], strategy=mv.summary(eq), benchmark=mv.summary(mv.bench_curve(*window)),
               years=blend.yearly(eq), stale_exits=int(stale))
    json.dump(res, open(OUT / f"dev_barbell_{name}.json", "w"), indent=1, default=float)
    print(name, json.dumps(res["strategy"], default=float))


def test(name):
    picks, eq, log, stale, window = run(name, "test")
    out = dict(spec=CANDS[name], benchmark=mv.summary(mv.bench_curve(*TEST)), strategy=mv.summary(eq),
               years=blend.yearly(eq), stale_exits=int(stale), buys=int((log.side == "BUY").sum()),
               strategy_no_costs=mv.summary(blend.run(picks, None, TEST, cost=0.0)[0]),
               strategy_50pct_haircut=mv.summary(blend.run(picks, None, TEST, haircut=0.5)[0]))
    json.dump(out, open(OUT / f"test_barbell_{name}.json", "w"), indent=1, default=float)
    pd.DataFrame({"S&P 500 TR": mv.bench_curve(*TEST), name: eq}).to_csv(OUT / f"test_barbell_{name}_curves.csv")
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    if sys.argv[1] == "sleeves":
        build_sleeves(sys.argv[2])
    else:
        {"dev": dev, "test": test}[sys.argv[1]](sys.argv[2])
