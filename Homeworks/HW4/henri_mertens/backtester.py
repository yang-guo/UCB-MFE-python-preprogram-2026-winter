"""An independent close-to-close, lot-based HW4 backtester.

Orders contain dollar budgets decided at the previous session's close.
Fractional adjusted-price units represent total-return exposure.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


class UnresolvedPriceGap(RuntimeError):
    """Stop instead of fabricating a mark or liquidation proceeds."""

    def __init__(self, message, partial=None):
        super().__init__(message)
        self.partial = partial


@dataclass
class Lot:
    symbol: str
    units: float
    entry_date: pd.Timestamp
    due_date: pd.Timestamp
    last_price: float
    stale_sessions: int = 0
    receivable: bool = False


def performance(history, initial_capital, start, end):
    nav = history.nav.to_numpy(dtype=float)
    returns = nav / np.r_[initial_capital, nav[:-1]] - 1
    path = np.r_[initial_capital, nav]
    drawdown = path / np.maximum.accumulate(path) - 1
    years = (pd.Timestamp(end) - pd.Timestamp(start)).days / 365.25
    risk_free_daily = 1.02 ** (1 / 252) - 1
    volatility = returns.std(ddof=1)
    return {
        "final_value": float(nav[-1]),
        "cagr": float((nav[-1] / initial_capital) ** (1 / years) - 1),
        "maximum_drawdown": float(drawdown.min()),
        "sharpe": float((returns.mean() - risk_free_daily)
                        / volatility * np.sqrt(252)) if volatility > 0 else None,
    }


def simulate(prices, calendar, predictions, start, end, config, actions=None):
    """Plan at t, fill at t+1; never access targets or future score rows.

    Missing entries are cancelled without a hindsight-selected replacement.
    Missing held-stock marks/exits carry at most the stated session limit;
    a longer incident makes the result unresolved and stops the simulation.
    """
    calendar = pd.DatetimeIndex(calendar)
    sessions = calendar[(calendar >= start) & (calendar <= end)]
    if len(sessions) < 2:
        raise ValueError("Need at least two market sessions")
    if not prices.index.is_unique or not prices.columns.is_unique:
        raise ValueError("Duplicate quote keys")
    allowed = {"date", "symbol", "prediction"}
    if set(predictions.columns) != allowed:
        raise ValueError("Allocator accepts only dates, symbols and predictions")
    if predictions.duplicated(["date", "symbol"]).any():
        raise ValueError("Duplicate prediction keys")
    initial = config["initial_capital"]
    cash = float(initial)
    fee = config["cost_bps_each_side"] / 10000
    lots, fills, history, incidents, plans = [], [], [], [], []
    daily_scores = {date: group for date, group in predictions.groupby("date")}
    actions = {} if actions is None else actions
    for date in sessions:
        position = calendar.get_loc(date)
        signal_date = calendar[position - 1] if position > 0 else None
        # This snapshot belongs to the preceding close, before today's quotes.
        planned_nav = cash + sum(lot.units * lot.last_price for lot in lots)
        orders = []
        first_session = position == 0 or calendar[position - 1].month != date.month
        if first_session and signal_date in daily_scores:
            scores = daily_scores[signal_date].copy()
            scores = scores.loc[np.isfinite(scores.prediction)].sort_values(
                ["prediction", "symbol"], ascending=[False, True]
            ).head(config["number_of_stocks"])
            total_budget = min(cash, config["monthly_nav_budget"] * planned_nav)
            per_name = total_budget / len(scores) if len(scores) else 0.0
            for row in scores.itertuples():
                existing = sum(lot.units * lot.last_price for lot in lots
                               if lot.symbol == row.symbol)
                capacity = max(0, config["maximum_name_weight"] * planned_nav - existing)
                budget = min(per_name, capacity)
                if budget > 1e-8:
                    orders.append((row.symbol, budget))
                    plans.append({"signal_date": signal_date, "entry_date": date,
                                  "symbol": row.symbol, "budget": budget})
            assert sum(budget for _, budget in orders) <= cash + 1e-6
        quotes = prices.loc[date] if date in prices.index else pd.Series(dtype=float)
        remaining = []
        for lot in lots:
            action = actions.get(lot.symbol)
            if action and date >= pd.Timestamp(action["effective_date"]):
                old_symbol = lot.symbol
                if action["kind"] == "cash":
                    lot.last_price = action["adjusted_cash_per_unit"]
                    lot.receivable = True
                else:
                    if "adjusted_unit_ratio" not in action:
                        raise UnresolvedPriceGap(
                            f"{old_symbol} converts to {action['successor']} on "
                            f"{date.date()}, but verified successor prices are unavailable.",
                            partial={"history": pd.DataFrame(history),
                                     "fills": pd.DataFrame(fills),
                                     "incidents": pd.DataFrame(incidents),
                                     "orders": pd.DataFrame(plans),
                                     "open_lots": pd.DataFrame([vars(item) for item in lots])},
                        )
                    lot.symbol = action["successor"]
                    lot.units *= action["adjusted_unit_ratio"]
                    lot.last_price = action["successor_initial_mark"]
                lot.stale_sessions = 0
                incidents.append({"date": date, "symbol": old_symbol,
                                  "kind": "verified_" + action["kind"] + "_conversion"})
                # Cash claims must not be converted repeatedly on later days.
                if lot.receivable:
                    lot.symbol = "CASH_CLAIM:" + old_symbol
            if lot.receivable:
                if date >= lot.due_date:
                    gross = lot.units * lot.last_price
                    cash += gross
                    fills.append({"date": date, "side": "cash_claim_release",
                                  "symbol": lot.symbol, "units": lot.units,
                                  "price": lot.last_price, "gross": gross, "fee": 0.0,
                                  "entry_date": lot.entry_date,
                                  "scheduled_exit": lot.due_date})
                else:
                    remaining.append(lot)
                continue
            quote = quotes.get(lot.symbol, np.nan)
            if np.isfinite(quote) and quote > 0:
                lot.last_price, lot.stale_sessions = float(quote), 0
            else:
                lot.stale_sessions += 1
                incidents.append({"date": date, "symbol": lot.symbol,
                                  "kind": "missing_held_quote",
                                  "stale_sessions": lot.stale_sessions})
                if lot.stale_sessions > config["max_stale_sessions"]:
                    raise UnresolvedPriceGap(
                        f"{lot.symbol} missing {lot.stale_sessions} sessions "
                        f"on {date.date()}; no verified exit or recovery.",
                        partial={
                            "history": pd.DataFrame(history),
                            "fills": pd.DataFrame(fills),
                            "incidents": pd.DataFrame(incidents),
                            "orders": pd.DataFrame(plans),
                            "open_lots": pd.DataFrame([vars(item) for item in lots]),
                        },
                    )
            if date >= lot.due_date and lot.stale_sessions == 0:
                gross = lot.units * lot.last_price
                cash += gross * (1 - fee)
                fills.append({"date": date, "side": "sell", "symbol": lot.symbol,
                              "units": lot.units, "price": lot.last_price,
                              "gross": gross, "fee": gross * fee,
                              "entry_date": lot.entry_date,
                              "scheduled_exit": lot.due_date})
            else:
                remaining.append(lot)
        lots = remaining
        for symbol, budget in orders:
            action = actions.get(symbol)
            if action and date >= pd.Timestamp(action["effective_date"]):
                incidents.append({"date": date, "symbol": symbol,
                                  "kind": "cancelled_nontrading_predecessor"})
                continue
            quote = quotes.get(symbol, np.nan)
            if not (np.isfinite(quote) and quote > 0):
                incidents.append({"date": date, "symbol": symbol,
                                  "kind": "cancelled_missing_entry"})
                continue
            if budget > cash + 1e-6:
                raise AssertionError("An order spends cash twice")
            units = budget / (float(quote) * (1 + fee))
            gross = units * float(quote)
            cash -= gross * (1 + fee)
            due = date + pd.Timedelta(days=config["holding_days"])
            lots.append(Lot(symbol, units, date, due, float(quote)))
            fills.append({"date": date, "side": "buy", "symbol": symbol,
                          "units": units, "price": float(quote), "gross": gross,
                          "fee": gross * fee, "entry_date": date,
                          "scheduled_exit": due})
        marked = sum(lot.units * lot.last_price for lot in lots)
        if cash < -1e-6:
            raise AssertionError("Negative cash / leverage")
        history.append({"date": date, "cash": cash, "positions_value": marked,
                        "nav": cash + marked, "open_lots": len(lots),
                        "stale_lots": sum(lot.stale_sessions > 0 for lot in lots)})
    return {
        "history": pd.DataFrame(history), "fills": pd.DataFrame(fills),
        "incidents": pd.DataFrame(incidents), "orders": pd.DataFrame(plans),
        "open_lots": pd.DataFrame([vars(lot) for lot in lots]),
    }
