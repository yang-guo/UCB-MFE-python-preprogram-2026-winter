"""Large-liquid core strategy with risk controls: risk-adjusted weights, a low-vol ballast sleeve, a trend filter.

    .venv/bin/python Homeworks/HW4/Emil_Blaignan/blend.py dev  <candidate>     # validate on 2011-2015 only
    .venv/bin/python Homeworks/HW4/Emil_Blaignan/blend.py test <candidate>     # final test, run once

Goal: steady growth with lower volatility (high Sharpe, shallow drawdowns), not the highest raw return.

Building blocks (signals use information through the signal close; fills at the next close):
  tilt sleeve   core.py's ridge-selected top N of the K=150 most-traded stocks, weighted by dollar volume
                (`advw`) or dollar volume / volatility (`riskw`), each weight <= cap.
  ballast       share `b` of every tranche goes to a minimum-variance portfolio of the 100 most-traded stocks
                (long-only, cap 5%, ridge-shrunk covariance) instead of the tilt sleeve.
  trend filter  the tranche is scaled by `defensive` (0.5) when the S&P 500 TR is below its 200-day average
                at the signal close; the rest stays in cash earning the 2% cash rate.
Cash earns 2% (the assignment's risk-free assumption) in every run, so all candidates are on the same footing.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import core  # noqa: E402
import minvar as mv  # noqa: E402

OUT = mv.OUT
CASH = 0.02
DEV = (pd.Timestamp("2011-01-01"), pd.Timestamp("2015-12-31"))
TEST = mv.TEST
BALLAST = dict(K=100, N=100, sel="adv", wt="mv", cap=0.05, shrink=0.5)


def tilt(wt, cap, N=50):
    return dict(K=core.K_TILT, N=N, sel="ridge", alpha=1e3, wt=wt, cap=cap)


CANDS = {
    "ref_advw10":        dict(tilt=tilt("advw", 0.10)),
    "riskw5":            dict(tilt=tilt("riskw", 0.05)),
    "ballast_only":      dict(tilt=tilt("riskw", 0.05), b=1.0),
    "riskw5_b30":        dict(tilt=tilt("riskw", 0.05), b=0.3),
    "riskw5_b50":        dict(tilt=tilt("riskw", 0.05), b=0.5),
    "riskw5_b70":        dict(tilt=tilt("riskw", 0.05), b=0.7),
    "ref_trend":         dict(tilt=tilt("advw", 0.10), trend=True),
    "riskw5_b30_trend":  dict(tilt=tilt("riskw", 0.05), b=0.3, trend=True),
    "riskw5_b50_trend":  dict(tilt=tilt("riskw", 0.05), b=0.5, trend=True),
    "advw5_b50_trend":   dict(tilt=tilt("advw", 0.05), b=0.5, trend=True),
    # round 2: variations on the pure minimum-variance core (still validated on 2011-2015 only)
    "mv100_cap3":        dict(tilt=dict(K=100, N=100, sel="adv", wt="mv", cap=0.03, shrink=0.5)),
    "mv200_cap3":        dict(tilt=dict(K=200, N=200, sel="adv", wt="mv", cap=0.03, shrink=0.5)),
    "mv100_shrink1":     dict(tilt=dict(K=100, N=100, sel="adv", wt="mv", cap=0.05, shrink=1.0)),
    "mv100_shrink01":    dict(tilt=dict(K=100, N=100, sel="adv", wt="mv", cap=0.05, shrink=0.1)),
    "ridge_mv60":        dict(tilt=dict(K=core.K_TILT, N=60, sel="ridge", alpha=1e3, wt="mv", cap=0.05, shrink=0.5)),
}


def trend_multiplier(picks, defensive=0.5):
    b = pd.read_parquet(OUT / "benchmark.parquet").set_index("date").adj_close
    ok = b >= b.rolling(200, min_periods=200).mean()
    pairs = picks.drop_duplicates("signal_date").set_index("entry_date").signal_date
    m = np.where(ok.reindex(pairs.values).fillna(True).to_numpy(), 1.0, defensive)
    return pd.Series(m, index=pairs.index)


def build(c, w, core_df, R, dates):
    picks, coefs = core.build_picks(w, core_df, R, c["tilt"], dates)
    b = c.get("b", 0.0)
    if b:
        ballast, _ = core.build_picks(w, core_df, R, BALLAST, dates)
        picks = pd.concat([picks.assign(weight=picks.weight * (1 - b)), ballast.assign(weight=ballast.weight * b)],
                          ignore_index=True)
        picks = picks[picks.weight > 0].reset_index(drop=True)
    expo = trend_multiplier(picks) if c.get("trend") else None
    return picks, expo


def run(picks, expo, window, **kw):
    return mv.backtest(picks, window, exposure=expo, cash_rate=CASH, **kw)


def yearly(eq):
    y = eq.resample("YE").last()
    y.loc[eq.index[0]] = eq.iloc[0]
    r = (y.sort_index().pct_change().dropna() * 100).round(1)
    return {int(k.year): float(v) for k, v in r.items()}


def dev(name):
    c, (w, core_df), R = CANDS[name], core.load(), mv.daily_returns()
    dates = w.date.drop_duplicates()
    dates = dates[(dates >= DEV[0]) & (dates <= DEV[1])]
    picks, expo = build(c, w, core_df, R, dates)
    window = (picks.entry_date.min(), DEV[1])
    eq, log, stale = run(picks, expo, window)
    bench = mv.bench_curve(*window)
    res = dict(cfg=c, strategy=mv.summary(eq), benchmark=mv.summary(bench), years=yearly(eq),
               avg_max_weight=float(picks.groupby("signal_date").weight.max().mean()), stale_exits=int(stale))
    json.dump(res, open(OUT / f"dev_blend_{name}.json", "w"), indent=1, default=float)
    print(name, json.dumps(res["strategy"], default=float))


def test(name):
    c, (w, core_df), R = CANDS[name], core.load(), mv.daily_returns()
    dates = w.date.drop_duplicates()
    dates = dates[(dates >= dates[dates < TEST[0]].iloc[-1]) & (dates <= TEST[1])]
    picks, expo = build(c, w, core_df, R, dates)
    eq, log, stale = run(picks, expo, TEST)
    out = dict(cfg=c, benchmark=mv.summary(mv.bench_curve(*TEST)), strategy=mv.summary(eq), years=yearly(eq),
               strategy_no_costs=mv.summary(run(picks, expo, TEST, cost=0.0)[0]),
               strategy_50pct_haircut=mv.summary(run(picks, expo, TEST, haircut=0.5)[0]),
               stale_exits=int(stale), buys=int((log.side == "BUY").sum()))
    json.dump(out, open(OUT / f"test_blend_{name}.json", "w"), indent=1, default=float)
    pd.DataFrame({"S&P 500 TR": mv.bench_curve(*TEST), name: eq}).to_csv(OUT / f"test_blend_{name}_curves.csv")
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    {"dev": dev, "test": test}[sys.argv[1]](sys.argv[2])
