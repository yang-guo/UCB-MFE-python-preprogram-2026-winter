"""Long-only backtester for the monthly top-10 Ridge strategy.

A signal is the close on date t. The buy fills at the next session's close, and
the dollar budget is fixed from the close on t: one third of NAV, split equally
across the top scores, and never larger than the cash on t. Sales on the fill
date do not increase that budget. Each lot is sold at the first close on or
after 90 calendar days from the fill. Fees are taken out of the buy budget and
out of sale proceeds. Cash earns zero. Adjusted prices are the marks.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Lot:
    lot_id: int
    symbol: str
    units: float
    signal_date: pd.Timestamp
    entry_date: pd.Timestamp
    entry_price: float
    due_date: pd.Timestamp


def run_backtest(
    prices,
    scores,
    start,
    end,
    top_n=10,
    cohort_fraction=1 / 3,
    fee_rate=0.001,
    capital=1_000_000.0,
    holding_days=90,
    max_stale_sessions=5,
):
    """Run the strategy from start through end, inclusive.

    prices: DataFrame of adjusted closes, dates in the index, symbols in columns.
    scores: columns date, symbol, score. One finite score per symbol on a signal date.
    """
    if not 0 < cohort_fraction <= 1 or top_n < 1 or fee_rate < 0 or capital <= 0:
        raise ValueError("bad backtest settings")
    px = prices.sort_index().astype(float)
    calendar = pd.DatetimeIndex(px.index)
    if calendar.has_duplicates or not calendar.is_monotonic_increasing:
        raise ValueError("price dates must be unique and sorted")

    score_table = scores.copy()
    score_table["date"] = pd.to_datetime(score_table["date"])
    if score_table.duplicated(["date", "symbol"]).any() or not np.isfinite(score_table["score"]).all():
        raise ValueError("scores must be finite with one row per date and symbol")
    by_date = {date: group.set_index("symbol")["score"] for date, group in score_table.groupby("date")}

    start, end = pd.Timestamp(start), pd.Timestamp(end)
    window = calendar[(calendar >= start) & (calendar <= end)]
    if len(window) == 0:
        raise ValueError("no sessions in the backtest window")

    cash = float(capital)
    lots = []
    next_id = 1
    fills = []
    history = []
    last_price = {}
    stale = {}

    # The day before the window is only a mark, so the first trade's fee shows up in a return.
    first_pos = calendar.get_loc(window[0])
    if first_pos > 0:
        history.append(_mark(calendar[first_pos - 1], cash, lots, px, last_price, stale, max_stale_sessions))

    for pos in range(first_pos, calendar.get_loc(window[-1]) + 1):
        date = calendar[pos]
        signal_date = calendar[pos - 1] if pos else None
        held_symbols = {lot.symbol for lot in lots}

        planned = {}
        if signal_date is not None and signal_date.to_period("M") != date.to_period("M"):
            prior_marks = _prices_for(
                signal_date, px, held_symbols, dict(last_price), dict(stale), max_stale_sessions
            )
            prior_nav = cash + sum(lot.units * prior_marks[lot.symbol] for lot in lots)
            budget = min(cash, prior_nav * cohort_fraction)
            planned = _budgets(by_date.get(signal_date, pd.Series(dtype=float)), budget, top_n)

        for lot in list(lots):
            if date < lot.due_date:
                continue
            price = px.at[date, lot.symbol] if lot.symbol in px.columns else np.nan
            if not np.isfinite(price) or price <= 0:
                continue
            notional = lot.units * price
            fee = notional * fee_rate
            cash += notional - fee
            fills.append(_fill("sell", date, lot, price, notional, fee, cash_change=notional - fee))
            lots.remove(lot)

        for symbol, budget in planned.items():
            if symbol not in px.columns:
                continue
            price = px.at[date, symbol]
            if not np.isfinite(price) or price <= 0 or budget <= 0:
                continue
            units = budget / (price * (1 + fee_rate))
            notional = units * price
            fee = notional * fee_rate
            if notional + fee > cash + 1e-6:
                units = cash / (price * (1 + fee_rate))
                notional = units * price
                fee = notional * fee_rate
            if units <= 0:
                continue
            lot = Lot(next_id, symbol, units, signal_date, date, price, date + pd.Timedelta(days=holding_days))
            next_id += 1
            cash -= notional + fee
            lots.append(lot)
            fills.append(_fill("buy", date, lot, price, notional, fee, cash_change=-(notional + fee)))

        history.append(_mark(date, cash, lots, px, last_price, stale, max_stale_sessions))

    return {
        "history": pd.DataFrame(history),
        "fills": pd.DataFrame(fills),
        "open_lots": lots,
        "cash": cash,
    }


def _budgets(scores, budget, top_n):
    if scores.empty or budget <= 0:
        return {}
    ranked = scores.rename("score").reset_index().sort_values(["score", "symbol"], ascending=[False, True]).head(top_n)
    each = budget / len(ranked)
    return dict(zip(ranked["symbol"], [each] * len(ranked)))


def _prices_for(date, px, symbols, last_price, stale, max_stale_sessions):
    marks = {}
    for symbol in symbols:
        observed = px.at[date, symbol] if symbol in px.columns else np.nan
        if np.isfinite(observed) and observed > 0:
            last_price[symbol] = float(observed)
            stale[symbol] = 0
            marks[symbol] = float(observed)
            continue
        # A missing print is marked at the last close. The lot is not sold until a real close exists.
        seen = stale.get(symbol, 0) + 1
        stale[symbol] = seen
        if symbol not in last_price:
            raise ValueError(f"no usable price for {symbol} on {date.date()}")
        marks[symbol] = last_price[symbol]
    return marks


def _mark(date, cash, lots, px, last_price, stale, max_stale_sessions):
    marks = _prices_for(date, px, {lot.symbol for lot in lots}, last_price, stale, max_stale_sessions)
    market_value = sum(lot.units * marks[lot.symbol] for lot in lots)
    return {"date": date, "cash": cash, "market_value": market_value, "nav": cash + market_value, "n_lots": len(lots)}


def _fill(side, date, lot, price, notional, fee, cash_change):
    return {
        "date": date,
        "side": side,
        "symbol": lot.symbol,
        "lot_id": lot.lot_id,
        "signal_date": lot.signal_date,
        "price": price,
        "units": lot.units,
        "notional": notional,
        "fee": fee,
        "cash_change": cash_change,
        "due_date": lot.due_date,
    }
