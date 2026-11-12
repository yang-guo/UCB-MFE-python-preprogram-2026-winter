"""Builds the weekly dataset the final model uses.

    python Homeworks/HW4/Emil_Blaignan/build_weekly.py     (writes outputs/weekly.parquet)

One row per tradable stock on each weekly signal date. It has the lecture features plus these, which only
use trailing information.
  beta_mkt    252 day beta to the S&P 500 total return index
  beta_mom    252 day loading on a momentum spread (top minus bottom momentum quintile, ranked the day before)
  resid_mom   12 minus 1 momentum after removing the market and momentum spread, per unit of risk
  idisc       share of down days minus share of up days over 252 days, times the sign of momentum
The weekly dates are every fifth session, lined up so one lands on the session before the first test close.
"""
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent / "outputs"
W, MINP = 252, 126
FEATS = ["momentum_21", "momentum_63", "momentum_252_21", "volatility_63", "drawdown_252",
         "net_margin", "operating_roa", "debt_to_assets", "cash_to_assets", "fcf_to_assets",
         "accruals_to_assets"]
TEST_START = pd.Timestamp("2016-04-01")

bench = pd.read_parquet(OUT / "benchmark.parquet").set_index("date").adj_close
cal_all = bench.index
anchor = cal_all.get_indexer([TEST_START], method="bfill")[0] - 1
sig = cal_all[np.arange(anchor % 5, len(cal_all), 5)]
sig = sig[sig >= "2010-01-01"]

cal = cal_all[cal_all >= "2008-01-01"]
d = pd.read_parquet(OUT / "panel.parquet", columns=["symbol", "date", "ret_1d", "momentum_252_21", "tradable"],
                    filters=[("date", ">=", pd.Timestamp("2008-01-01"))])
wide = lambda col: d.pivot(index="date", columns="symbol", values=col).reindex(cal).astype("float32")
R, MOM = wide("ret_1d"), wide("momentum_252_21")
T = d.pivot(index="date", columns="symbol", values="tradable").reindex(cal).fillna(False).astype(bool)
M = bench.pct_change().reindex(cal).astype("float32")
ok = R.notna()

# momentum spread factor, ranked on the previous close so it is tradable ex ante
rank = MOM.where(T).rank(axis=1, pct=True)
top, bot = (rank >= 0.8).shift(1, fill_value=False), (rank <= 0.2).shift(1, fill_value=False)
F = (R.where(top).mean(axis=1) - R.where(bot).mean(axis=1)).astype("float32")

Mm = ok.mul(M, axis=0).where(ok)
Fm = ok.mul(F, axis=0).where(ok)
rm = lambda X: X.rolling(W, min_periods=MINP).mean()
mR, mM, mF = rm(R), rm(Mm), rm(Fm)
Cmm, Cff = rm(Mm * Mm) - mM ** 2, rm(Fm * Fm) - mF ** 2
Cmf, Crm, Crf = rm(Mm * Fm) - mM * mF, rm(R * Mm) - mR * mM, rm(R * Fm) - mR * mF
det = (Cmm * Cff - Cmf ** 2).where(lambda x: x > 1e-12)
beta_mkt = (Crm * Cff - Crf * Cmf) / det
beta_mom = (Crf * Cmm - Crm * Cmf) / det
resvar = ((rm(R * R) - mR ** 2) - (beta_mkt * Crm + beta_mom * Crf)).clip(lower=1e-8)
ssum = lambda X: X.fillna(0).rolling(W).sum() - X.fillna(0).rolling(21).sum()  # t-251 .. t-21
resid_mom = (ssum(R) - beta_mkt * ssum(Mm) - beta_mom * ssum(Fm)) / (np.sqrt(W - 21) * np.sqrt(resvar))
fp, fn = rm((R > 0).astype("float32").where(ok)), rm((R < 0).astype("float32").where(ok))
idisc = np.sign(MOM) * (fn - fp)

state = pd.DataFrame({
    "mkt_ret_252": bench / bench.shift(252) - 1, "mkt_ret_21": bench / bench.shift(21) - 1,
    "mkt_vol_63": bench.pct_change().rolling(63).std() * np.sqrt(252),
    "mkt_dd_252": bench / bench.rolling(252).max() - 1,
    "mom_ret_126": F.rolling(126).sum().reindex(cal_all), "mom_vol_63": F.rolling(63).std().reindex(cal_all) * np.sqrt(252),
})

cols = ["symbol", "date"] + FEATS + ["target", "entry_date", "label_end"]
w = pd.read_parquet(OUT / "panel.parquet", columns=cols + ["tradable"], filters=[("date", ">=", pd.Timestamp("2010-01-01"))])
w = w[w.tradable & w.date.isin(sig)].drop(columns="tradable").reset_index(drop=True)
di, si = cal.get_indexer(w.date), R.columns.get_indexer(w.symbol)
for name, fr in [("beta_mkt", beta_mkt), ("beta_mom", beta_mom), ("resid_mom", resid_mom), ("idisc", idisc)]:
    w[name] = fr.to_numpy()[di, si]
w = w.join(state, on="date")
w.to_parquet(OUT / "weekly.parquet", index=False)
print(w.shape, "weeks", w.date.nunique(), "\nnon-missing share:\n", w.notna().mean().round(3).to_string())
print(w[["beta_mkt", "beta_mom", "resid_mom", "idisc"]].describe().round(3).to_string())
