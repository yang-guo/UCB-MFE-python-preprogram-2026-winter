"""HW4 backtester and performance metrics.

Rules: $1M, long only, no borrowing. We decide at the close of day t and buy at the next close.
Each purchase is a lot, sold 90 days later at the close (or at the next quote).

My choices:
- costs = 10 bps of what we trade, on each buy and each sale
- cash earns nothing
- no quote: we keep the last price ("freeze"), or we put the stock at 0 after 5 days ("write_off")
- a price that jumps more than x5 is a data error, we ignore it
- when we sell, we put the money back the next day in the latest top stocks

How it works:
A. a loop day by day decides what we buy and sell, with the cash
B. the portfolio value: p_ret = (ret * W.shift()).sum(axis=1) - costs, value = 1M x cumprod(1 + p_ret)
Metrics like in the course: CAGR on days / 365.25, Sharpe on 252 days with 2% risk-free.
"""

import numpy as np
import pandas as pd


def run_backtest(prices, scores, start, end, top_n=20, cohort_fraction=1 / 3, cost_bps=10,
                 missing_policy="freeze", reinvest=True, capital=1_000_000):
    """prices: days x stocks (adjusted close, empty if no quote). scores: date, symbol, score."""
    c = cost_bps / 10_000
    first = prices.index.searchsorted(pd.Timestamp(start))
    last = prices.index.searchsorted(pd.Timestamp(end), side="right") - 1
    assert first > 0, "prices must include the signal day before start"
    days = prices.index[first - 1:last + 1]  # day 0 = the signal day, we only start buying the day after
    px = prices.loc[days].to_numpy(float)
    px_known = prices.loc[:days[-1]].ffill().loc[days].to_numpy(float)  # last price we know, to spot the jumps
    col = {s: j for j, s in enumerate(prices.columns)}
    scores_by_day = dict(list(scores.groupby("date")))

    cash, lots, orders, latest = float(capital), [], [], None
    last_price, last_seen, bad_now = {}, {}, set()
    history, trades, events, positions, marks = [], [], [], [], []

    def event(day, symbol, text, lot=None):
        events.append({"date": day, "symbol": symbol, "lot_id": lot, "event": text})

    for i, day in enumerate(days):
        # 1. today's prices for what we hold and what we want to buy (a x5 jump = no price)
        held = {lot["symbol"] for lot in lots}
        reference = {o["symbol"]: o["ref_price"] for o in orders} | {s: last_price[s] for s in held}
        today = {}
        for s, ref in reference.items():
            p = px[i, col[s]]
            if not (np.isfinite(p) and p > 0):
                continue
            if not 1 / 5 < p / ref < 5:
                if s not in bad_now:
                    event(day, s, f"bad print ignored ({ref:.4f} -> {p:.4f})")
                    bad_now.add(s)
                continue
            bad_now.discard(s)
            today[s] = p
            if s in held:
                last_price[s], last_seen[s] = p, i
        bought = sold = costs = sale_cash = 0.0

        # 2. we sell the lots that are 90 days old, if the stock trades today
        for lot in list(lots):
            s = lot["symbol"]
            if day < lot["due"]:
                continue
            if s not in today:
                if not lot["delay_logged"]:
                    event(day, s, "exit delayed: no quote on due date", lot["id"])
                    lot["delay_logged"] = True
                continue
            value = lot["units"] * today[s]
            cash += value - value * c
            sale_cash += value - value * c
            sold, costs = sold + value, costs + value * c
            trades.append({"date": day, "side": "SELL", "lot_id": lot["id"], "symbol": s, "price": today[s],
                           "units": lot["units"], "value": value, "cost": value * c, "cash_change": value - value * c,
                           "cash_after": cash, "signal_date": lot["signal"], "due_date": lot["due"],
                           "pnl": value - value * c - lot["units"] * lot["price"] - lot["cost"]})
            lots.remove(lot)

        # 3. no quote: we keep the last price, or with write_off the lot goes to 0 after 5 days
        stale, written_off = 0, 0.0
        for lot in list(lots):
            s = lot["symbol"]
            gap = i - last_seen[s]
            if gap == 0:
                continue
            stale += 1
            if gap > 5 and missing_policy == "write_off":
                event(day, s, f"written off at 0 (last mark {last_price[s]:.4f})", lot["id"])
                written_off += lot["units"] * last_price[s]
                lots.remove(lot)
            elif gap == 6:
                event(day, s, "no quote for more than 5 sessions: mark frozen", lot["id"])

        # 4. we buy what we decided yesterday, the budget pays the price + the cost
        for o in orders:
            s = o["symbol"]
            if s not in today:
                event(day, s, "order skipped: no quote")
                continue
            units = o["budget"] / (today[s] * (1 + c))
            value = units * today[s]
            assert value + value * c <= cash + 1e-6, "never spend the same cash twice"
            cash -= value + value * c
            bought, costs = bought + value, costs + value * c
            lot = {"id": len(trades), "symbol": s, "units": units, "price": today[s], "cost": value * c,
                   "entry": day, "due": day + pd.Timedelta(days=90), "signal": o["signal"], "delay_logged": False}
            lots.append(lot)
            trades.append({"date": day, "side": "BUY", "lot_id": lot["id"], "symbol": s, "price": today[s],
                           "units": units, "value": value, "cost": value * c, "cash_change": -(value + value * c),
                           "cash_after": cash, "signal_date": o["signal"], "due_date": lot["due"], "pnl": np.nan})
            last_price[s], last_seen[s] = today[s], i
        orders = []
        cash = 0.0 if -1e-6 < cash < 0 else cash  # tiny rounding errors
        assert cash >= 0

        # 5. value = cash + units x last price; we also keep the units and prices for part B
        units = {}
        for lot in lots:
            units[lot["symbol"]] = units.get(lot["symbol"], 0.0) + lot["units"]
        positions.append(units)
        marks.append({s: last_price[s] for s in held | set(units)})
        holdings = sum(u * last_price[s] for s, u in units.items())
        nav = cash + holdings
        history.append({"date": day, "cash": cash, "holdings": holdings, "nav": nav, "open_lots": len(lots),
                        "stale_lots": stale, "bought": bought, "sold": sold, "costs": costs,
                        "written_off": written_off})

        # 6. we decide tomorrow's buys with the latest scores (end of month: NAV/3, plus the money from sales)
        if day in scores_by_day:
            latest = scores_by_day[day]
        if latest is not None and i < len(days) - 1:
            budget = min(cash, nav * cohort_fraction) if day in scores_by_day else 0.0
            if reinvest:
                budget = max(budget, min(cash, sale_cash))
            top = latest.sort_values(["score", "symbol"], ascending=[False, True]).drop_duplicates("symbol").head(top_n)
            if len(top) and budget > 0:
                orders = [{"symbol": s, "budget": budget / len(top), "signal": day, "ref_price": px_known[i, col[s]]}
                          for s in top.symbol]

    # B. portfolio value: yesterday's weights x today's returns
    history = pd.DataFrame(history, index=days)
    prices_held = pd.DataFrame(marks, index=days)              # price of each stock we hold, day by day
    values = pd.DataFrame(positions, index=days).fillna(0.0) * prices_held
    W = values.div(history.nav, axis=0)                        # weights move with the prices, no rebalancing
    ret = prices_held.pct_change(fill_method=None)              # 0 when the price is frozen
    p_ret = (ret * W.shift()).sum(axis=1) - (history.costs + history.written_off) / history.nav.shift()
    p_ret.iloc[0] = 0.0
    value = capital * (1 + p_ret).cumprod()
    assert np.allclose(value, history.nav, rtol=1e-9)  # must give the same value as the cash accounting
    history["nav"] = value
    history["p_ret"] = p_ret

    open_lots = pd.DataFrame([{"lot_id": lot["id"], "symbol": lot["symbol"], "entry": lot["entry"],
                               "mark": last_price[lot["symbol"]], "last_quote": days[last_seen[lot["symbol"]]],
                               "value": lot["units"] * last_price[lot["symbol"]]} for lot in lots])
    return {"history": history.drop(columns="date"), "trades": pd.DataFrame(trades),
            "events": pd.DataFrame(events, columns=["date", "symbol", "lot_id", "event"]), "open_lots": open_lots}


def benchmark_nav(benchmark, history, capital=1_000_000):
    """$1M in the S&P 500 TR, bought at the same close as our first buy."""
    b = benchmark.reindex(history.index)
    nav = capital * b / b.iloc[1]
    nav.iloc[0] = capital
    return nav


def performance(nav, annual_rf=0.02):
    """Final value, CAGR, max drawdown, volatility and Sharpe from the daily values."""
    nav = nav.dropna()
    returns = nav.pct_change().dropna()
    years = (nav.index[-1] - nav.index[0]).days / 365.25
    excess = returns - ((1 + annual_rf) ** (1 / 252) - 1)
    return pd.Series({
        "final_value": nav.iloc[-1],
        "total_return": nav.iloc[-1] / nav.iloc[0] - 1,
        "CAGR": (nav.iloc[-1] / nav.iloc[0]) ** (1 / years) - 1,
        "max_drawdown": (nav / nav.cummax() - 1).min(),
        "volatility": returns.std() * np.sqrt(252),
        "Sharpe": excess.mean() / returns.std() * np.sqrt(252),
    })


def trading_stats(result):
    """How much we trade and pay in costs per year, and how much cash we keep on average."""
    h = result["history"].iloc[1:]
    years = (h.index[-1] - result["history"].index[0]).days / 365.25
    return pd.Series({
        "annual_turnover": (h.bought.sum() + h.sold.sum()) / h.nav.mean() / years,
        "annual_cost_drag": h.costs.sum() / h.nav.mean() / years,
        "average_cash_share": (h.cash / h.nav).mean(),
        "stale_lot_days": int(h.stale_lots.sum()),
    })
