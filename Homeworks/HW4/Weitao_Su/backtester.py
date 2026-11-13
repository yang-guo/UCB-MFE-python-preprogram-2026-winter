"""Track cash and each stock purchase day by day."""
import numpy as np
import pandas as pd


def simulate(quotes, benchmark, scores, start, end, settings, actions=None):
    actions = actions or []
    calendar = benchmark.index
    dates = calendar[(calendar >= pd.Timestamp(start)) & (calendar <= pd.Timestamp(end))]
    previous = calendar[calendar.get_loc(dates[0]) - 1]
    next_session = pd.Series(calendar[1:], index=calendar[:-1])
    assert (scores.date.map(next_session) == scores.entry_date).all()
    if "fit_date" in scores:
        assert scores.fit_date.le(scores.date).all()
    by_date = dict(tuple(scores.groupby("date")))
    cash = settings["initial_cash"]
    fee_rate = settings["fee_bps"] / 10000
    lots, trades, orders, events, claims = [], [], [], [], []
    problem = None
    next_id = 1

    def plan(date, nav):
        if date not in by_date:
            return []
        budget = min(cash, nav / 3) / settings["top_n"]
        exposure = {}
        for lot in lots:
            symbol = lot["symbol"]
            exposure[symbol] = exposure.get(symbol, 0) + lot["units"] * lot["mark"]
        pending = []
        ranked = by_date[date].sort_values(["score", "symbol"], ascending=[False, True])
        for row in ranked.itertuples():
            amount = min(budget, max(0, nav * .05 - exposure.get(row.symbol, 0)))
            if amount > 1e-8:
                order = {"signal_date": date, "entry_date": row.entry_date,
                         "symbol": row.symbol, "budget": amount}
                pending.append(order)
                orders.append(order.copy())
                exposure[row.symbol] = exposure.get(row.symbol, 0) + amount
            if len(pending) == settings["top_n"]:
                break
        assert sum(order["budget"] for order in pending) <= cash + 1e-7
        return pending

    daily = [{"date": previous, "nav": cash, "cash": cash, "open_value": 0.,
              "benchmark_nav": cash, "initial": True}]
    pending = plan(previous, cash)
    for date in dates:
        prices = quotes.loc[date]
        # Apply deals only after their terms have become available.
        conversions = {}
        for lot in lots:
            for action_id, deal in enumerate(actions):
                if (lot["symbol"] == deal["symbol"] and "cash_price" not in lot
                        and lot["entry_date"] <= deal["effective_date"]
                        and action_id not in lot.get("applied_actions", [])
                        and date >= deal["available_date"]):
                    conversions[lot["lot_id"]] = (action_id, deal)
        # Unknown missing prices still stop the run before changing the ledger.
        missing = []
        for lot in lots:
            item = conversions.get(lot["lot_id"])
            deal = item[1] if item else None
            if "cash_price" in lot or (deal and deal.get("kind", "cash") == "cash"):
                continue
            symbol = deal["new_symbol"] if deal else lot["symbol"]
            if not np.isfinite(prices.get(symbol, np.nan)) or prices.get(symbol, 0) <= 0:
                missing.append(symbol)
        if missing:
            problem = f"No price for {missing[0]} on {date.date()}"
            break
        for lot in lots:
            if lot["lot_id"] in conversions:
                action_id, deal = conversions[lot["lot_id"]]
                lot.setdefault("applied_actions", []).append(action_id)
                if deal.get("cvr_per_unit", 0) > 0:
                    claims.append({"date": date, "lot_id": lot["lot_id"], "symbol": lot["symbol"],
                                   "units": lot["units"] * deal["cvr_per_unit"],
                                   "maximum_per_unit": deal["cvr_max"], "assumed_value": 0.,
                                   "source": deal["source"]})
                if deal.get("kind", "cash") == "cash":
                    lot["cash_price"] = deal["cash_price"]
                    value = lot["units"] * deal["cash_price"]
                else:
                    lot["reserved_cash"] = lot.get("reserved_cash", 0) + lot["units"] * deal.get("cash_per_unit", 0)
                    lot["symbol"] = deal["new_symbol"]
                    lot["units"] *= deal["unit_ratio"]
                    value = lot["units"] * prices[lot["symbol"]] + lot["reserved_cash"]
                events.append({"date": date, "lot_id": lot["lot_id"], "symbol": lot["symbol"],
                               "kind": deal.get("kind", "cash"), "value": value, "source": deal["source"]})
        for lot in list(lots):
            if date >= lot["maturity"]:
                is_cash = "cash_price" in lot
                price = lot["cash_price"] if is_cash else prices[lot["symbol"]]
                gross = lot["units"] * price
                fee = 0. if is_cash else gross * fee_rate
                cash_component = lot.get("reserved_cash", 0.)
                cash += gross - fee + cash_component
                trades.append({"date": date, "side": "CASH_RELEASE" if is_cash else "SELL", "lot_id": lot["lot_id"],
                               "symbol": lot["symbol"], "units": lot["units"], "price": price,
                               "fee": fee, "cash_change": gross - fee + cash_component,
                               "cash_component": cash_component,
                               "entry_date": lot["entry_date"], "maturity": lot["maturity"]})
                lots.remove(lot)
        for order in pending:
            assert order["entry_date"] == date
            price = prices.get(order["symbol"], np.nan)
            if not np.isfinite(price) or price <= 0:
                continue  # Skip this purchase and keep the cash.
            amount = order["budget"]
            assert amount <= cash + 1e-7
            units = amount / (price * (1 + fee_rate))
            fee = units * price * fee_rate
            cash -= units * price + fee
            lot = {"lot_id": next_id, "symbol": order["symbol"], "entry_date": date,
                   "maturity": date + pd.Timedelta(days=90), "units": units,
                   "cost": amount, "mark": price}
            next_id += 1
            lots.append(lot)
            trades.append({"date": date, "side": "BUY", "lot_id": lot["lot_id"],
                           "symbol": lot["symbol"], "units": units, "price": price,
                           "fee": fee, "cash_change": -amount,
                           "entry_date": date, "maturity": lot["maturity"]})
        for lot in lots:
            lot["mark"] = lot["cash_price"] if "cash_price" in lot else prices[lot["symbol"]]
        open_value = sum(lot["units"] * lot["mark"] + lot.get("reserved_cash", 0.) for lot in lots)
        nav = cash + open_value
        assert cash >= -1e-7 and all(lot["units"] > 0 for lot in lots)
        daily.append({"date": date, "nav": nav, "cash": cash, "open_value": open_value,
                      "benchmark_nav": settings["initial_cash"] * benchmark.loc[date] / benchmark.loc[dates[0]],
                      "initial": False})
        pending = plan(date, nav) if date < dates[-1] else []

    daily = pd.DataFrame(daily)
    trades = pd.DataFrame(trades, columns=["date", "side", "lot_id", "symbol", "units",
                                         "price", "fee", "cash_change", "entry_date", "maturity", "cash_component"])
    trades["cash_component"] = trades.cash_component.fillna(0.)
    cash_changes = trades.groupby("date").cash_change.sum().reindex(daily.date, fill_value=0).astype(float)
    assert np.allclose(daily.cash, settings["initial_cash"] + cash_changes.cumsum(), atol=1e-6, rtol=0)
    assert np.allclose(daily.nav, daily.cash + daily.open_value)
    return {"daily": daily, "trades": trades, "orders": pd.DataFrame(orders),
            "open_lots": pd.DataFrame(lots), "events": pd.DataFrame(events),
            "claims": pd.DataFrame(claims), "problem": problem}
