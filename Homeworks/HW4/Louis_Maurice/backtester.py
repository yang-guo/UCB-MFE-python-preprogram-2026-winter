"""Daily long-only backtester for HW4."""

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd


@dataclass
class Lot:
    symbol: str
    units: float
    signal_date: pd.Timestamp
    entry_date: pd.Timestamp
    entry_price: float
    entry_fee: float
    due_date: pd.Timestamp


class Backtester:
    """quotes: adjusted closes (sessions x symbols). scores: {signal_date: Series}."""

    def __init__(self, quotes, scores, initial_cash=1_000_000, fee_bps=5,
                 top_n=20, cohort_fraction=1 / 3, holding_days=90):
        self.quotes = quotes.sort_index()
        self.scores = scores
        self.initial_cash = float(initial_cash)
        self.fee_rate = fee_bps / 10_000
        self.top_n = top_n
        self.cohort_fraction = cohort_fraction
        self.holding_days = holding_days

    def plan_budgets(self, signal_date, cash, nav):
        scores = self.scores.get(signal_date)
        if scores is None or scores.empty:
            return {}
        top = scores.sort_index().sort_values(ascending=False, kind="stable").head(self.top_n)
        budget = min(cash, nav * self.cohort_fraction)
        return {symbol: budget / len(top) for symbol in top.index}

    def run(self, start, end):
        calendar = self.quotes.index
        days = calendar[(calendar >= pd.Timestamp(start)) & (calendar <= pd.Timestamp(end))]
        cash = self.initial_cash
        nav = cash
        lots, fills, missed = [], [], []
        last_mark, last_seen = {}, {}
        history = [{"date": calendar[calendar.get_loc(days[0]) - 1], "cash": cash,
                    "holdings_value": 0., "nav": cash, "open_lots": 0, "stale_lots": 0}]

        for day in days:
            i = calendar.get_loc(day)
            signal_date = calendar[i - 1]

            budgets = {}
            if day.to_period("M") != signal_date.to_period("M"):
                budgets = self.plan_budgets(signal_date, cash, nav)

            observed = self.quotes.loc[day].dropna()
            for symbol, price in observed.items():
                last_mark[symbol], last_seen[symbol] = price, i

            for lot in list(lots):
                if day >= lot.due_date and lot.symbol in observed:
                    price = observed[lot.symbol]
                    notional = lot.units * price
                    fee = notional * self.fee_rate
                    cash += notional - fee
                    lots.remove(lot)
                    fills.append({"date": day, "symbol": lot.symbol, "side": "SELL", "price": price,
                                  "units": lot.units, "notional": notional, "fee": fee,
                                  "cash_change": notional - fee, "signal_date": lot.signal_date,
                                  "due_date": lot.due_date})

            for symbol, budget in budgets.items():
                if symbol not in observed:
                    missed.append({"date": day, "symbol": symbol})
                    continue
                price = observed[symbol]
                units = budget / (price * (1 + self.fee_rate))
                notional = units * price
                fee = notional * self.fee_rate
                cash -= notional + fee
                lot = Lot(symbol, units, signal_date, day, price, fee,
                          day + pd.Timedelta(days=self.holding_days))
                lots.append(lot)
                fills.append({"date": day, "symbol": symbol, "side": "BUY", "price": price,
                              "units": units, "notional": notional, "fee": fee,
                              "cash_change": -(notional + fee), "signal_date": signal_date,
                              "due_date": lot.due_date})
            if cash < -1e-6:
                raise ValueError(f"Negative cash on {day.date()}")

            holdings = sum(lot.units * last_mark[lot.symbol] for lot in lots)
            stale = sum(last_seen[lot.symbol] < i for lot in lots)
            nav = cash + holdings
            history.append({"date": day, "cash": cash, "holdings_value": holdings, "nav": nav,
                            "open_lots": len(lots), "stale_lots": stale})

        open_lots = pd.DataFrame([asdict(lot) for lot in lots])
        if len(open_lots):
            open_lots["last_quote_date"] = [calendar[last_seen[s]] for s in open_lots.symbol]
            open_lots["last_mark"] = [last_mark[s] for s in open_lots.symbol]
        return (pd.DataFrame(history).set_index("date"), pd.DataFrame(fills),
                open_lots, pd.DataFrame(missed))


def performance(nav, annual_rf=0.02):
    returns = nav.pct_change().dropna()
    daily_rf = (1 + annual_rf) ** (1 / 252) - 1
    years = (nav.index[-1] - nav.index[0]).days / 365.25
    return pd.Series({
        "final_value": nav.iloc[-1],
        "CAGR": (nav.iloc[-1] / nav.iloc[0]) ** (1 / years) - 1,
        "max_drawdown": (nav / nav.cummax() - 1).min(),
        "Sharpe": np.sqrt(252) * (returns - daily_rf).mean() / returns.std(),
    })
