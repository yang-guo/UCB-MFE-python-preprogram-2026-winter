"""Portfolio and prediction metrics.

``performance`` follows Lecture 8's ``BacktestAnalytics.performance``: daily
simple returns, CAGR over actual elapsed days / 365.25, max drawdown from the
NAV path, Sharpe = sqrt(252) * mean(daily excess) / std(daily return) with the
2% annual hurdle converted to its daily equivalent (1.02^(1/252) - 1).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def performance(nav: pd.Series, annual_rf: float = 0.02) -> pd.Series:
    nav = nav.sort_index()
    if len(nav) < 2 or not np.isfinite(nav).all() or (nav <= 0).any():
        raise ValueError("Need >= 2 positive finite NAVs")
    r = nav.pct_change().dropna()
    daily_rf = (1 + annual_rf) ** (1 / 252) - 1
    vol = r.std(ddof=1)
    years = (nav.index[-1] - nav.index[0]).days / 365.25
    dd = nav / nav.cummax() - 1
    return pd.Series({
        "start": nav.index[0].date(), "end": nav.index[-1].date(),
        "final_value": nav.iloc[-1], "total_return": nav.iloc[-1] / nav.iloc[0] - 1,
        "CAGR": (nav.iloc[-1] / nav.iloc[0]) ** (1 / years) - 1,
        "max_drawdown": dd.min(), "max_drawdown_date": dd.idxmin().date(),
        "annualized_volatility": vol * np.sqrt(252),
        "Sharpe": np.sqrt(252) * (r - daily_rf).mean() / vol if vol > 1e-12 else np.nan,
    })


def relative(nav: pd.Series, bench: pd.Series) -> pd.Series:
    """Active-return statistics against the benchmark NAV on the same dates."""
    a, b = nav.pct_change().dropna(), bench.reindex(nav.index).pct_change().dropna()
    active = a - b
    beta = np.cov(a, b)[0, 1] / b.var() if b.var() > 0 else np.nan
    te = active.std() * np.sqrt(252)
    return pd.Series({"beta": beta, "tracking_error": te,
                      "information_ratio": active.mean() * 252 / te if te > 0 else np.nan,
                      "correlation": a.corr(b)})


def comparison_table(nav: pd.Series, bench_nav: pd.Series, annual_rf=0.02, names=("strategy", "S&P 500 TR")):
    t = pd.DataFrame({names[0]: performance(nav, annual_rf), names[1]: performance(bench_nav, annual_rf)})
    return t


def benchmark_nav(benchmark: pd.Series, dates: pd.DatetimeIndex, first_close, capital: float) -> pd.Series:
    """$capital invested at ``first_close``; flat (cash) on any earlier date in ``dates``."""
    b = benchmark.reindex(dates)
    first_close = pd.Timestamp(first_close)
    out = pd.Series(capital, index=dates, dtype=float)
    later = dates >= first_close
    out.loc[later] = capital * b.loc[later] / benchmark.loc[first_close]
    return out


def ic_series(frame: pd.DataFrame, score="score", target="target", min_names=20) -> pd.Series:
    """Same-date Spearman rank IC, one value per signal date."""
    f = frame[["date", score, target]].dropna()
    return f.groupby("date").apply(
        lambda x: x[score].corr(x[target], method="spearman") if len(x) >= min_names else np.nan,
        include_groups=False).dropna().rename("rank_IC")


def prediction_metrics(frame: pd.DataFrame, score="score", target="target", min_names=20) -> pd.Series:
    """Same-date rank IC and top/bottom-decile spread.

    Overlapping 90-day targets mean dates are not independent; the t-stat
    below uses non-overlapping 13-week blocks of the IC series.
    """
    f = frame[["date", score, target]].dropna()
    g = f.groupby("date")
    ic = g.apply(lambda x: x[score].corr(x[target], method="spearman") if len(x) >= min_names else np.nan,
                 include_groups=False).dropna()

    def spread(x):
        q = x[score].rank(pct=True)
        return x.loc[q > 0.9, target].mean() - x.loc[q <= 0.1, target].mean()
    sp = g.apply(spread, include_groups=False).dropna()
    blocks = ic.groupby(np.arange(len(ic)) // 13).mean()
    t = blocks.mean() / blocks.std(ddof=1) * np.sqrt(len(blocks)) if len(blocks) > 2 else np.nan
    return pd.Series({"dates": len(ic), "mean_rank_IC": ic.mean(), "IC_std": ic.std(),
                      "IC_hit_rate": (ic > 0).mean(), "IC_tstat_blocks": t,
                      "top_minus_bottom_decile": sp.mean()})


def portfolio_diagnostics(result) -> pd.Series:
    h = result.history.iloc[1:]
    closed = result.lots_closed
    return pd.Series({
        "average_cash_fraction": (h.cash / h.nav).mean(),
        "fees": h.fees.sum(),
        "turnover_x_avg_nav": (h.bought.sum() + h.sold.sum()) / 2 / h.nav.mean(),
        "lots_opened": int((result.fills.side == "BUY").sum()),
        "lots_closed": len(closed),
        "lot_win_rate": (closed.net_return > 0).mean() if len(closed) else np.nan,
        "mean_lot_net_return": closed.net_return.mean() if len(closed) else np.nan,
        "open_lots_at_end": len(result.open_lots),
        "frozen_lots_at_end": int(result.open_lots.frozen.sum()) if len(result.open_lots) else 0,
        "max_frozen_value_fraction": (h.frozen_value / h.nav).max(),
        "rejected_orders": int((result.events.event == "rejected_no_quote").sum()),
        "avg_names_held_proxy": h.open_lots.mean(),
    })
