"""Small shared pieces for the final strategy (windows, return data, metrics, backtest wrapper)."""
from pathlib import Path

import numpy as np
import pandas as pd

from backtest import execute, metrics

OUT = Path(__file__).resolve().parent / "outputs"
RESULTS = Path(__file__).resolve().parent / "results"

# Development runs 2010 to 2015. I validate on 2011 to 2015 and train from 2010.
DEV = (pd.Timestamp("2011-01-01"), pd.Timestamp("2015-12-31"))
TEST = (pd.Timestamp("2016-04-01"), pd.Timestamp("2026-09-30"))

MATURE = pd.Timedelta(days=102)  # the longest label I found ended 101 days after its signal date
COST = 0.001                    # 10 bps per side
CASH_RATE = 0.02                # idle cash earns the 2% risk free rate

# Characteristics the models see. The first 11 come from the lecture pipeline.
# The next 4 are the beta, momentum spread, residual momentum and path smoothness features.
FEATS = ["momentum_21", "momentum_63", "momentum_252_21", "volatility_63", "drawdown_252",
         "net_margin", "operating_roa", "debt_to_assets", "cash_to_assets", "fcf_to_assets",
         "accruals_to_assets", "beta_mkt", "beta_mom", "resid_mom", "idisc", "adv_log"]


def adv_table():
    """Trailing 63 day median dollar volume (close x volume) at each weekly signal date."""
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


def daily_returns():
    """Daily returns (dates by symbols) from 2009, used for the risk model."""
    cal = pd.read_parquet(OUT / "benchmark.parquet").date
    cal = cal[cal >= "2009-01-01"]
    d = pd.read_parquet(OUT / "panel.parquet", columns=["symbol", "date", "ret_1d"],
                        filters=[("date", ">=", pd.Timestamp("2009-01-01"))])
    return d.pivot(index="date", columns="symbol", values="ret_1d").reindex(cal)


def bench_curve(start, end):
    """$1,000,000 in the S&P 500 total return index at the first close, held to the end."""
    b = pd.read_parquet(OUT / "benchmark.parquet").set_index("date").adj_close.loc[start:end]
    return 1_000_000 * b / b.iloc[0]


def summary(eq):
    m = metrics(eq)
    m["ann_vol"] = eq.pct_change().std() * np.sqrt(252)
    return m


def yearly(eq):
    y = eq.resample("YE").last()
    y.loc[eq.index[0]] = eq.iloc[0]
    r = (y.sort_index().pct_change().dropna() * 100).round(1)
    return {int(k.year): float(v) for k, v in r.items()}


def run(picks, window, cost=COST, haircut=0.0):
    """Backtest weekly picks (symbol, entry_date, weight) over a window. Returns equity, trade log, stale exits."""
    start, end = window
    picks = picks[(picks.entry_date >= start) & (picks.entry_date <= end)]
    px = pd.read_parquet(OUT / "panel.parquet", columns=["symbol", "date", "adj_close"],
                         filters=[("symbol", "in", sorted(picks.symbol.unique())), ("date", ">=", start), ("date", "<=", end)])
    cal = pd.read_parquet(OUT / "benchmark.parquet").date
    px = px.pivot(index="date", columns="symbol", values="adj_close").reindex(cal[(cal >= start) & (cal <= end)])
    return execute(picks, px, cost=cost, exit_haircut=haircut, cash_rate=CASH_RATE)
