"""My backtester for HW4. Long only, starts with $1,000,000 and never borrows.

How it works
  * Picks are decided at the close of the signal date and filled at the next session's close.
  * Each purchase is held 90 calendar days, then sold at that day's close. If the stock has no quote that
    day I wait up to 5 sessions. If it never trades again I sell at its last close and count it as a stale exit.
  * Each weekly tranche invests equity / 13 of capital, split by the weight column in picks (or equally if
    there is no weight column). A stock that cannot be filled stays in cash.
  * Costs are charged per side on the traded amount. Idle cash earns cash_rate.
  * Missing prices are never filled from the future. The last close is carried for valuation only.
"""
import numpy as np
import pandas as pd


def execute(picks, px, capital=1_000_000.0, tranches=13, n_target=100, cost=0.001,
            hold_days=90, max_exit_delay=5, exit_haircut=0.0, exposure=None, cash_rate=0.0):
    """picks: DataFrame[symbol, entry_date]; px: DataFrame(index=sessions, columns=symbols) of closes.
    exposure: optional Series (entry_date -> multiplier in [0, 1]) scaling that day's tranche; the
    rest of the tranche stays in cash. It must be computed from information known at the signal."""
    dates = px.index
    D = len(dates)
    col = {s: i for i, s in enumerate(px.columns)}
    P = px.to_numpy(float)
    Pf = px.ffill().to_numpy(float)  # carried marks

    by_day = {}
    weights = picks["weight"] if "weight" in picks.columns else [None] * len(picks)
    for s, d, wgt in zip(picks.symbol, picks.entry_date, weights):
        t = dates.get_loc(d) if d in dates else None
        if t is not None and s in col:
            by_day.setdefault(t, []).append((col[s], wgt))

    cash = float(capital)
    positions = []  # [col, shares, exit_idx, stale]
    equity = np.empty(D)
    trades, stale_exits = [], 0
    for t in range(D):
        if t > 0 and cash_rate:  # idle cash earns cash_rate (annual, simple, by calendar days)
            cash *= 1 + cash_rate * (dates[t] - dates[t - 1]).days / 365.0
        still_open = []
        for pos in positions:
            c, shares, exit_idx, stale = pos
            if exit_idx == t:
                price = Pf[t, c] * ((1 - exit_haircut) if stale else 1.0)
                cash += shares * price * (1 - cost)
                stale_exits += int(stale)
                trades.append((dates[t], px.columns[c], "SELL", shares, price))
            else:
                still_open.append(pos)
        positions = still_open

        if t in by_day:
            equity_prev = equity[t - 1] if t > 0 else capital
            mult = 1.0 if exposure is None else float(exposure.get(dates[t], 1.0))
            budget = min(equity_prev / tranches * min(max(mult, 0.0), 1.0), cash)
            for c, wgt in by_day[t]:
                # `weight` (share of the tranche) if given, else an equal 1/n_target split
                per_name = budget * (1.0 / n_target if wgt is None else wgt)
                price = P[t, c]
                if np.isnan(price) or per_name <= 0:
                    continue  # cannot fill: stays in cash, no invented fill
                shares = per_name / (1 + cost) / price
                cash -= per_name
                maturity = dates[t] + pd.Timedelta(days=hold_days)
                m = dates.searchsorted(maturity)
                exit_idx, stale = None, False
                if m < D:
                    window = range(m, min(m + max_exit_delay, D - 1) + 1)
                    exit_idx = next((j for j in window if not np.isnan(P[j, c])), None)
                    if exit_idx is None and m + max_exit_delay < D:
                        exit_idx, stale = m, True
                positions.append([c, shares, exit_idx, stale])
                trades.append((dates[t], px.columns[c], "BUY", shares, price))
            assert cash >= -1e-6, "cash went negative"

        equity[t] = cash + sum(p[1] * Pf[t, p[0]] for p in positions)
    out = pd.Series(equity, index=dates, name="equity")
    log = pd.DataFrame(trades, columns=["date", "symbol", "side", "shares", "price"])
    return out, log, stale_exits


def metrics(value, rf=0.02):
    """Final value, CAGR, max drawdown, Sharpe from daily returns (2% annual risk-free)."""
    value = value.dropna()
    r = value.pct_change().dropna()
    years = (value.index[-1] - value.index[0]).days / 365.25
    return {
        "final_value": value.iloc[-1],
        "cagr": (value.iloc[-1] / value.iloc[0]) ** (1 / years) - 1,
        "max_drawdown": (value / value.cummax() - 1).min(),
        "sharpe": (r.mean() - rf / 252) / r.std() * np.sqrt(252),
    }
