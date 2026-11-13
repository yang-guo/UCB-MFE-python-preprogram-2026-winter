"""Performance metrics for a daily NAV series vs the benchmark.

Conventions (match the data guide / hw4.md): CAGR uses actual elapsed days /
365.25; volatility and Sharpe use 252 sessions; Sharpe uses a 2% annual
risk-free rate converted to a daily rate (1.02^(1/252) - 1).
"""

import numpy as np
import pandas as pd

from . import config


def benchmark_nav(bench, start_signal, first_entry, end, capital=config.INITIAL_CAPITAL):
    """$1M in ^SP500TR bought at the first entry close (no fee), held to `end`.
    Includes the same initial-capital row on the signal date as the strategy."""
    b = bench.loc[start_signal:end]
    nav = capital * b / b.loc[first_entry]
    nav.loc[:first_entry] = capital
    return nav


def drawdown(nav):
    return nav / nav.cummax() - 1


def performance(nav, bench_nav=None, annual_rf=config.ANNUAL_RF, history=None):
    nav = nav.dropna()
    r = nav.pct_change().dropna()
    rf = (1 + annual_rf) ** (1 / 252) - 1
    years = (nav.index[-1] - nav.index[0]).days / 365.25
    out = {
        "start": nav.index[0].date(), "end": nav.index[-1].date(),
        "final_value": nav.iloc[-1],
        "total_return": nav.iloc[-1] / nav.iloc[0] - 1,
        "CAGR": (nav.iloc[-1] / nav.iloc[0]) ** (1 / years) - 1,
        "volatility": r.std() * np.sqrt(252),
        "sharpe": (r - rf).mean() / (r - rf).std() * np.sqrt(252),
        "max_drawdown": drawdown(nav).min(),
    }
    out["calmar"] = out["CAGR"] / abs(out["max_drawdown"])
    if bench_nav is not None:
        rb = bench_nav.reindex(nav.index).pct_change().dropna()
        rr, rb = r.align(rb, join="inner")
        active = rr - rb
        out["beta"] = np.cov(rr, rb)[0, 1] / rb.var()
        out["correlation"] = rr.corr(rb)
        out["tracking_error"] = active.std() * np.sqrt(252)
        out["information_ratio"] = active.mean() * 252 / out["tracking_error"]
        out["alpha_annual"] = (rr.mean() - rf - out["beta"] * (rb.mean() - rf)) * 252
    if history is not None:
        h = history.loc[nav.index[0]:nav.index[-1]]
        out["avg_cash_fraction"] = (h.cash / h.nav).iloc[1:].mean()
        out["annual_turnover"] = h.gross_bought.sum() / h.nav.mean() / years
        out["total_fees"] = h.fees.sum()
        out["avg_names"] = h.n_names.iloc[1:].mean()
        out["stale_mark_days"] = int((h.stale_marks > 0).sum())
        out["forced_exits"] = int(h.forced_exits.sum())
        out["rejected_orders"] = int(h.rejected.sum())
        out["reduced_orders"] = int(h.reduced.sum())
    return out


def calendar_year_returns(navs: dict):
    """Calendar-year returns; partial first/last years are labelled as such."""
    out = {}
    for name, nav in navs.items():
        yearly = nav.groupby(nav.index.year).last()
        start = pd.Series([nav.iloc[0]], index=[nav.index[0].year - 1])
        out[name] = pd.concat([start, yearly]).pct_change().dropna()
    df = pd.DataFrame(out)
    df.index.name = "year"
    return df


def rolling_excess(nav, bench_nav, window=252):
    a = nav.pct_change(window)
    b = bench_nav.reindex(nav.index).pct_change(window)
    return (a - b).dropna()
