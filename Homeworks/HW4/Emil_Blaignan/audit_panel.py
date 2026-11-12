"""Audit the preprocessed panel: things worth knowing before building a strategy.

    .venv/bin/python Homeworks/HW4/Emil_Blaignan/audit_panel.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent / "outputs"
p = pd.read_parquet(OUT / "panel.parquet")
b = pd.read_parquet(OUT / "benchmark.parquet")
pd.options.display.width = 200
pd.options.display.float_format = "{:,.4f}".format


def hdr(t):
    print(f"\n=== {t}")


hdr("shape / coverage")
print(f"rows {len(p):,}  symbols {p.symbol.nunique():,}  dates {p.date.nunique():,}  {p.date.min().date()} -> {p.date.max().date()}")
print(f"ever a universe member: {p[p.is_universe_member].symbol.nunique():,}")

hdr("eligible names per year (member + 63 complete returns), mean per day")
e = p[p.eligible].groupby("date").size()
print(e.groupby(e.index.year).agg(["mean", "min", "max"]).round(0).T.to_string())

hdr("daily return outliers (calendar-consecutive returns only)")
r = p.ret_1d.dropna()
print(f"n={len(r):,}  |ret|>50%: {(r.abs() > .5).sum():,}  >100%: {(r > 1).sum():,}  <-50%: {(r < -.5).sum():,}  max {r.max():,.1f}  min {r.min():.3f}")
m = p[p.is_universe_member].ret_1d.dropna()
print(f"members only: |ret|>50% {(m.abs() > .5).sum():,}  >100% {(m > 1).sum():,}  max {m.max():,.1f}")
print(p.loc[p.ret_1d.abs().nlargest(8).index, ["symbol", "date", "adj_close", "close", "ret_1d"]].to_string(index=False))

hdr("90d target distribution (labelled rows, tradable = what the model sees)")
t = p[p.tradable & p.target.notna()].target
print(t.describe(percentiles=[.001, .01, .05, .5, .95, .99, .999]).to_string())
print("targets > +500%:", (t > 5).sum(), "  < -95%:", (t < -.95).sum())

hdr("label censoring (tradable rows whose entry should be labelable: date <= 2026-06-30)")
m2 = p[p.tradable & (p.date <= "2026-06-30")]
print(f"no entry price: {m2.entry_price.isna().mean():.2%}   entry ok but no exit: {(m2.entry_price.notna() & m2.target.isna()).mean():.2%}")
cens = m2[m2.entry_price.notna() & m2.target.isna()]
print("censored exits by year:", cens.groupby(cens.date.dt.year).size().to_dict())

hdr("stocks whose price history ends before the last session (possible delist/merger/stale)")
last = p.groupby("symbol").date.max()
ended = last[last < "2026-09-01"]
mem_last = p[p.is_universe_member].groupby("symbol").date.max()
print(f"{len(ended):,} symbols end before Sep 2026; of which ever members: {ended.index.isin(mem_last.index).sum():,}")
print("end year of those:", ended.groupby(ended.dt.year).size().to_dict())

hdr("price quality")
print("zero/NaN volume share of rows:", f"{(p.volume.fillna(0) <= 0).mean():.2%}", " members:", f"{(p[p.is_universe_member].volume.fillna(0) <= 0).mean():.2%}")
ratio = (p.adj_close / p.close).replace([np.inf, -np.inf], np.nan)
print("adj_close/close ratio: median", round(ratio.median(), 3), " share >1.5 or <0.5:", f"{((ratio > 1.5) | (ratio < .5)).mean():.2%}")

hdr("benchmark")
bb = b.set_index("date").adj_close
br = bb.pct_change().dropna()
tb = bb.loc["2016-04-01":]
yrs = (tb.index[-1] - tb.index[0]).days / 365.25
dd = (tb / tb.cummax() - 1).min()
sh = ((br.loc["2016-04-02":] - 0.02 / 252).mean() / br.loc["2016-04-02":].std()) * np.sqrt(252)
print(f"S&P500TR test window: {tb.index[0].date()}..{tb.index[-1].date()} CAGR {(tb.iloc[-1] / tb.iloc[0]) ** (1 / yrs) - 1:.2%}  maxDD {dd:.1%}  Sharpe {sh:.2f}  final ${1e6 * tb.iloc[-1] / tb.iloc[0]:,.0f}")

hdr("what drives the equal-weight top-500 universe vs the index (members, by year, mean 90d target)")
mm = p[p.is_universe_member & p.target.notna()]
print(mm.groupby(mm.date.dt.year).target.agg(mean="mean", median="median", n="size").T.round(3).to_string())

hdr("feature coverage on tradable rows (share non-missing), by period")
FEATS = ["momentum_21", "momentum_63", "momentum_252_21", "volatility_63", "drawdown_252",
         "net_margin", "operating_roa", "debt_to_assets", "cash_to_assets", "fcf_to_assets",
         "accruals_to_assets", "filing_age_days"]
tr = p[p.tradable]
per = pd.cut(tr.date, pd.to_datetime(["2000-01-01", "2010-01-01", "2016-01-01", "2026-10-01"]), right=False,
             labels=["pre-2010", "dev 2010-15", "test 2016-26"])
print(tr[FEATS].notna().groupby(per, observed=True).mean().T.round(3).to_string())
print("sector 'Unknown' share:", f"{(tr.sector == 'Unknown').mean():.2%}")
