"""Performance metrics and accounting audits.

``performance`` follows ``BacktestAnalytics.performance`` in
``Lectures/modeling/backtest.py`` (calendar-time CAGR with 365.25-day years,
252-session Sharpe with a 2% annual hurdle converted to a daily rate).
The audit functions are my own independent re-computations.
"""

import numpy as np
import pandas as pd


def performance(nav, annual_rf=0.02):
    nav = nav.sort_index()
    returns = nav.pct_change().dropna()
    daily_rf = (1 + annual_rf) ** (1 / 252) - 1
    excess = returns - daily_rf
    vol = returns.std(ddof=1)
    years = (nav.index[-1] - nav.index[0]).days / 365.25
    return pd.Series({
        "start": nav.index[0].date(), "end": nav.index[-1].date(),
        "final_value": nav.iloc[-1],
        "total_return": nav.iloc[-1] / nav.iloc[0] - 1,
        "CAGR": (nav.iloc[-1] / nav.iloc[0]) ** (1 / years) - 1,
        "max_drawdown": (nav / nav.cummax() - 1).min(),
        "annualized_volatility": vol * np.sqrt(252),
        "Sharpe (rf=2%)": np.sqrt(252) * excess.mean() / vol,
    })


def benchmark_nav(benchmark, history, initial_cash):
    """$1M in ^SP500TR at the first test close, on the same rows as the strategy.

    The strategy history starts with an initial-capital row on the session
    before the test; the benchmark holds cash ($1M) on that row and is invested
    at the first test close, so both start from the same $1M base.
    """
    days = history.index[1:]
    b = initial_cash * benchmark.loc[days] / benchmark.loc[days[0]]
    return pd.concat([pd.Series([initial_cash], index=history.index[:1]), b]).rename("benchmark_nav")


def comparison_table(result, benchmark, annual_rf=0.02):
    h = result.history
    bench = benchmark_nav(benchmark, h, result.initial_cash)
    return pd.DataFrame({
        "Strategy (unresolved lots at last mark)": performance(h.nav, annual_rf),
        "Strategy (unresolved lots at zero)": performance(h.nav_low, annual_rf),
        "S&P 500 TR (^SP500TR)": performance(bench, annual_rf),
    })


# ----------------------------------------------------------------------------- audits
def audit_cash(result):
    """Rebuild daily cash from the fill ledger alone and compare."""
    h = result.history
    flows = result.fills.groupby("date").cash_change.sum().reindex(h.index, fill_value=0.0)
    rebuilt = result.initial_cash + flows.cumsum()
    return float((rebuilt - h.cash).abs().max())


def audit_nav(result, adj, max_stale=5):
    """Rebuild NAV each day from fills + raw quote matrix, independently of the loop.

    Positions are reconstructed from BUY/SELL fills.  Each holding is marked at
    the most recent fill price or *accepted* quote: we approximate "accepted"
    by re-using the backtester's marks only through the fills, so this check
    is exact on days without suspect quotes and reports the largest gap.
    """
    h = result.history.iloc[1:]
    fills = result.fills
    units = fills.assign(signed=np.where(fills.side.eq("BUY"), fills.units, -fills.units)) \
        .pivot_table(index="date", columns="symbol", values="signed", aggfunc="sum") \
        .reindex(h.index, fill_value=0.0).fillna(0.0).cumsum()
    units = units.where(units.abs() > 1e-9, 0.0)
    marks = adj.reindex(index=h.index, columns=units.columns).ffill()
    holdings = (units * marks).sum(axis=1)
    rebuilt = h.cash + holdings
    return (rebuilt - h.nav)


def audit_timing(result, calendar, holding_days=90):
    """Check every fill obeys the decision/entry/exit timing rules."""
    f = result.fills
    cal = pd.DatetimeIndex(calendar)
    buys, sells = f[f.side.eq("BUY")], f[f.side.eq("SELL")]
    next_session = cal[cal.get_indexer(buys.signal_date) + 1]
    entry = buys.set_index("lot_id").date
    s = sells.set_index("lot_id")
    first_on_or_after_due = cal[cal.searchsorted(s.due_date)]
    return pd.Series({
        "buys": len(buys), "sells": len(sells),
        "buys filled at the session after their signal": bool((buys.date.to_numpy() == next_session.to_numpy()).all()),
        "due date = entry + 90 calendar days": bool(((s.due_date - entry.reindex(s.index)).dt.days.eq(holding_days)).all()),
        "no sale before maturity": bool((s.date >= s.due_date).all()),
        "sales on first session on/after maturity": int((s.date.to_numpy() == first_on_or_after_due.to_numpy()).sum()),
        "sales delayed (missing/suspect quote at maturity)": int((s.date.to_numpy() > first_on_or_after_due.to_numpy()).sum()),
        "each lot sold at most once": bool(~s.index.duplicated().any()),
        "min cash": float(result.history.cash.min()),
    })
