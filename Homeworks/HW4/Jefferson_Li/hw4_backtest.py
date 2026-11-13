"""Monthly ranking and 90-day lot accounting."""

import numpy as np
import pandas as pd

PARTICIPATION = 0.05


def run_backtest(scores, prices, calendar, start, end, *,
                 initial_cash=1_000_000, order_budget=250_000, fee_rate=0.001,
                 cash_events=None):
    sessions = pd.DatetimeIndex(calendar)
    sessions = sessions[(sessions >= pd.Timestamp(start)) & (sessions <= pd.Timestamp(end))]
    if sessions.empty or not sessions.is_monotonic_increasing or sessions.has_duplicates:
        raise ValueError("Expected sorted, unique market sessions.")
    if not np.isfinite([initial_cash, order_budget, fee_rate]).all() or min(initial_cash, order_budget) <= 0 or fee_rate < 0:
        raise ValueError("Invalid budget or fee.")

    quotes = prices.set_index(["date", "symbol"])[["adj_close", "close", "volume"]]
    if not quotes.index.is_unique:
        raise ValueError("Duplicate price rows.")
    if scores.duplicated(["date", "symbol"]).any() or not np.isfinite(scores["prediction"]).all():
        raise ValueError("Expected one finite score per stock and date.")
    winners = (scores.sort_values(["date", "prediction", "symbol"], ascending=[True, False, True])
               .drop_duplicates("date").set_index("date")["symbol"])
    signal_days = set(sessions[~sessions.to_period("M").duplicated()])
    cash, lots, pending = float(initial_cash), {}, None
    history, fills, skipped = [], [], []
    cash_events = cash_events or {}

    for index, day in enumerate(sessions):
        used = {}

        def quote(symbol):
            key = (day, symbol)
            if key not in quotes.index:
                return None, 0.0
            adjusted, close, volume = quotes.loc[key]
            if not np.isfinite(adjusted) or adjusted <= 0:
                return None, 0.0
            turnover = close * volume
            capacity = PARTICIPATION * turnover if np.isfinite(turnover) and turnover > 0 else 0.0
            return float(adjusted), max(0.0, capacity - used.get(symbol, 0.0))

        def fill(lot, side, price, units, fee):
            nonlocal cash
            notional = units * price
            change = -notional - fee if side == "BUY" else notional - fee
            cash += change
            fills.append(dict(lot_id=lot["lot_id"], side=side, date=day, symbol=lot["symbol"],
                              signal_date=lot["signal_date"], price=price, units=units,
                              notional=notional, fee=fee, cash_change=change, cash_after=cash))
            if side in {"BUY", "SELL"}:
                used[lot["symbol"]] = used.get(lot["symbol"], 0.0) + notional

        for lot_id, lot in list(lots.items()):
            symbol = lot["symbol"]
            event = cash_events.get((symbol, day))
            if event is not None:
                price, side = event
                fill(lot, side, price, lot["units"], 0.0)
                del lots[lot_id]
                continue

            price, capacity = quote(symbol)
            if price is None:
                lot["stale_sessions"] += 1
                if lot["stale_sessions"] > 5:
                    raise ValueError(f"No recent mark for {symbol} on {day.date()}.")
                continue

            lot["mark"], lot["stale_sessions"] = price, 0
            # Sell from maturity onward, sharing the day's capacity across lots.
            if day >= lot["maturity"] and capacity > 0:
                units = min(lot["units"], capacity / price)
                fill(lot, "SELL", price, units, units * price * fee_rate)
                lot["units"] -= units
                if lot["units"] <= 1e-10:
                    del lots[lot_id]

        if pending is not None:
            signal, symbol, budget = pending
            price, capacity = quote(symbol)
            notional = min(budget / (1 + fee_rate), capacity)
            if price is None or notional <= 0:
                skipped.append(dict(signal_date=signal, entry_date=day, symbol=symbol))
            else:
                units = notional / price
                lot_id = len(fills) + 1
                lot = dict(lot_id=lot_id, symbol=symbol, signal_date=signal, entry_date=day,
                           maturity=day + pd.Timedelta(days=90), units=units, mark=price,
                           stale_sessions=0)
                fill(lot, "BUY", price, units, notional * fee_rate)
                lots[lot_id] = lot
            pending = None

        value = sum(lot["units"] * lot["mark"] for lot in lots.values())
        history.append(dict(date=day, cash=cash, position_value=value, equity=cash + value,
                            open_lots=len(lots), stale_lots=sum(l["stale_sessions"] > 0 for l in lots.values())))
        if day in signal_days and day in winners.index and index + 1 < len(sessions) and cash > 0:
            pending = day, winners.at[day], min(order_budget, cash)

    return (
        pd.DataFrame(history).set_index("date"),
        pd.DataFrame(fills, columns=[
            "lot_id", "side", "date", "symbol", "signal_date", "price", "units",
            "notional", "fee", "cash_change", "cash_after",
        ]),
        pd.DataFrame(lots.values(), columns=[
            "lot_id", "symbol", "signal_date", "entry_date", "maturity", "units", "mark", "stale_sessions",
        ]),
        pd.DataFrame(skipped, columns=["signal_date", "entry_date", "symbol"]),
    )


def audit(history, fills, open_lots, prices, calendar, *,
          initial_cash=1_000_000, order_budget=250_000, fee_rate=0.001, cash_events=None):
    """Rebuild cash and marked positions from the fills."""
    quotes = prices.set_index(["date", "symbol"])
    marks = prices.pivot(index="date", columns="symbol", values="adj_close").reindex(history.index).ffill(limit=5)
    market = fills.loc[fills["side"].isin(["BUY", "SELL"])]
    turnover = (quotes["close"] * quotes["volume"]).reindex(market.groupby(["date", "symbol"]).size().index)
    assert market.groupby(["date", "symbol"])["notional"].sum().le(PARTICIPATION * turnover + 1e-7).all()
    value = pd.Series(0.0, index=history.index)
    count = pd.Series(0, index=history.index)
    remaining = {}
    for lot_id, trades in fills.groupby("lot_id", sort=False):
        buy = trades.iloc[0]
        assert buy["side"] == "BUY" and trades["side"].eq("BUY").sum() == 1
        assert buy["date"] == calendar[calendar.searchsorted(buy["signal_date"], side="right")]
        assert -buy["cash_change"] <= min(order_budget, history.at[buy["signal_date"], "cash"]) + 1e-7
        changes = pd.Series(0.0, index=history.index)
        for trade in trades.itertuples():
            key = (trade.date, trade.symbol)
            assert trade.symbol == buy["symbol"]
            if trade.side == "BUY":
                change = -trade.notional - trade.fee
            else:
                change = trade.notional - trade.fee
            if trade.side in {"BUY", "SELL"}:
                assert np.isclose(trade.price, quotes.at[key, "adj_close"])
                assert np.isclose(trade.fee, trade.notional * fee_rate)
                if trade.side == "SELL":
                    assert trade.date >= buy["date"] + pd.Timedelta(days=90)
            else:
                assert (trade.price, trade.side) == (cash_events or {})[(trade.symbol, trade.date)]
                assert trade.fee == 0
            assert np.isclose(trade.notional, trade.units * trade.price)
            assert np.isclose(trade.cash_change, change)
            changes.at[trade.date] += trade.units if trade.side == "BUY" else -trade.units
        units = changes.cumsum()
        assert units.min() >= -1e-9
        held = units.gt(1e-10)
        value.loc[held] += units.loc[held] * marks.loc[held, buy["symbol"]]
        count.loc[held] += 1
        if held.iloc[-1]:
            remaining[lot_id] = units.iloc[-1]

    cash = initial_cash + fills.groupby("date")["cash_change"].sum().reindex(history.index, fill_value=0).cumsum()
    np.testing.assert_allclose(initial_cash + fills["cash_change"].cumsum(), fills["cash_after"], atol=1e-7)
    np.testing.assert_allclose(history["cash"], cash, atol=1e-7)
    np.testing.assert_allclose(history["position_value"], value, atol=1e-7)
    np.testing.assert_allclose(history["equity"], cash + value, atol=1e-7)
    np.testing.assert_array_equal(history["open_lots"], count)
    assert history["cash"].min() >= -1e-7
    assert set(open_lots["lot_id"]) == set(remaining)
    for lot in open_lots.itertuples():
        assert np.isclose(lot.units, remaining[lot.lot_id])
