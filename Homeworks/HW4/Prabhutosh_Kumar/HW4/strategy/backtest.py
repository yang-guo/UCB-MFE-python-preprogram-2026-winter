"""Independent lot-based backtester. Orders are dollar budgets fixed yesterday."""
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd


@dataclass
class Lot:
    lot_id: int
    symbol: str
    entry: pd.Timestamp
    maturity: pd.Timestamp
    units: float
    mark: float
    stale: int = 0


def run(prices, scores, start, end, config):
    dates = prices.loc[start:end].index
    if len(dates) == 0:
        raise ValueError("No market sessions in evaluation period")
    cash, lots, orders = float(config["capital"]), [], []
    history, trades, incidents = [], [], []
    next_id = 0
    # The first test close can use a signal from the preceding session.
    previous = prices.index[prices.index < dates[0]]

    def plan(signal, equity):
        if signal not in scores.index:
            return []
        ranked = scores.loc[signal].dropna().sort_values(ascending=False, kind="stable").head(config["top_n"])
        if ranked.empty:
            return []
        budget = min(cash, equity * config["tranche_fraction"])
        allocations = []
        for symbol in ranked.index:
            exposure = sum(l.units * l.mark for l in lots if l.symbol == symbol)
            dollars = min(budget / len(ranked), max(0., equity * config["max_stock_weight"] - exposure))
            if dollars > 1e-8:
                allocations.append((symbol, dollars, signal))
        assert sum(x[1] for x in allocations) <= cash + 1e-6
        return allocations

    if len(previous):
        orders = plan(previous[-1], cash)
    for i, date in enumerate(dates):
        quotes = prices.loc[date]
        remaining = []
        # Sales at today's close cannot enlarge budgets fixed at yesterday's close.
        for lot in lots:
            price = quotes.get(lot.symbol, np.nan)
            valid = np.isfinite(price) and price > 0
            if valid:
                lot.mark, lot.stale = price, 0
            else:
                lot.stale += 1
                incidents.append({"date": date, "lot_id": lot.lot_id, "symbol": lot.symbol,
                                  "stale_sessions": lot.stale, "marked_value": lot.units * lot.mark})
                if config["strict_prices"] and lot.stale > config["stale_limit"]:
                    raise ValueError(f"Unresolved held-stock price: {lot.symbol} on {date}; no reliable NAV")
            if date >= lot.maturity and valid:
                gross = lot.units * price
                fee = gross * config["fee"]
                cash += gross - fee
                trades.append(dict(date=date, signal_date=pd.NaT, lot_id=lot.lot_id,
                                   symbol=lot.symbol, side="sell", price=price, units=lot.units,
                                   fee=fee, cash_flow=gross-fee, entry=lot.entry, maturity=lot.maturity))
            else:
                remaining.append(lot)
        lots = remaining
        for symbol, budget, signal in orders:
            price = quotes.get(symbol, np.nan)
            if not np.isfinite(price) or price <= 0:
                incidents.append({"date": date, "symbol": symbol, "kind": "skipped_entry"})
                continue
            assert signal < date
            assert budget <= cash + 1e-6
            units = budget / (price * (1 + config["fee"]))
            fee = budget - units * price
            cash -= budget
            next_id += 1
            maturity = date + pd.Timedelta(days=config["holding_days"])
            lots.append(Lot(next_id, symbol, date, maturity, units, price))
            trades.append(dict(date=date, signal_date=signal, lot_id=next_id, symbol=symbol,
                               side="buy", price=price, units=units, fee=fee,
                               cash_flow=-budget, entry=date, maturity=maturity))
        holdings = sum(l.units * l.mark for l in lots)
        stale_value = sum(l.units * l.mark for l in lots if l.stale > config["stale_limit"])
        assert cash >= -1e-6
        history.append(dict(date=date, cash=cash, holdings=holdings, value=cash+holdings,
                            open_lots=len(lots), unresolved_value=stale_value))
        orders = plan(date, cash + holdings) if i + 1 < len(dates) else []
    ledger = pd.DataFrame(trades)
    daily = pd.DataFrame(history).set_index("date")
    if not ledger.empty:
        assert np.isclose(config["capital"] + ledger.cash_flow.sum(), cash, atol=1e-5)
    assert np.isclose(daily.value.iloc[-1], cash + sum(l.units*l.mark for l in lots))
    return daily, ledger, pd.DataFrame([asdict(l) for l in lots]), pd.DataFrame(incidents)


def metrics(values, capital):
    """Include first-close entry fees against an explicit initial-cash baseline."""
    returns = values.pct_change(fill_method=None)
    returns.iloc[0] = values.iloc[0] / capital - 1
    peaks = values.cummax().clip(lower=capital)
    years = (values.index[-1] - values.index[0]).days / 365.25
    excess = returns - ((1.02)**(1/252) - 1)
    return {"final_value": float(values.iloc[-1]),
            "CAGR": float((values.iloc[-1]/capital)**(1/years)-1) if years else 0.,
            "max_drawdown": float((values/peaks-1).min()),
            "Sharpe": float(excess.mean()/excess.std()*np.sqrt(252)) if excess.std() > 0 else None}
