from dataclasses import dataclass

import duckdb
import numpy as np
import pandas as pd

def load_prices(database, start, end, symbols, calendar):
    """Adj close for backtest window"""
    with duckdb.connect(str(database), read_only=True) as con:
        px = con.execute("""SELECT date, symbol, adj_close FROM prices
                            WHERE date BETWEEN ? AND ? AND symbol IN (SELECT unnest(?))""",
                         [start, end, list(symbols)]).fetchdf()
    px["date"] = pd.to_datetime(px.date).astype("datetime64[ns]")
    px = px.loc[np.isfinite(px.adj_close) & px.adj_close.gt(0) & px.date.isin(calendar)]
    days = calendar[(calendar >= pd.Timestamp(start)) & (calendar <= pd.Timestamp(end))]
    return px.pivot(index="date", columns="symbol", values="adj_close").reindex(days)


NON_COMMON_SUFFIX = set("GHILMNOPRTUVWZ")   # Nasdaq 5th-letter codes: preferred, notes, warrants, rights, units

def is_common_stock(symbol, exchange):
    """Drop preferreds, notes, warrants, rights and SPAC units, keep only common stock"""
    if "-P" in symbol:
        return False
    if exchange == "NASDAQ" and len(symbol) == 5 and symbol[-1] in NON_COMMON_SUFFIX:
        return False
    return True

@dataclass
class Lot:
    lot_id: int
    symbol: str
    units: float
    entry_date: pd.Timestamp
    entry_price: float
    cost: float            # cash paid including fee
    exit_date: pd.Timestamp

class Portfolio:
    """OMS"""

    def __init__(self, initial_cash=1_000_000, fee_bps=10):
        self.initial_cash = float(initial_cash)
        self.cash = float(initial_cash)
        self.fee_rate = fee_bps / 10_000
        self.lots = {}
        self.fills = []
        self._next_id = 1

    def buy(self, symbol, budget, price, date, holding_days=90):
        """Spend budget on symbol at price"""
        if budget <= 0 or price <= 0:
            raise ValueError("budget and price must be < 0")
        if budget > self.cash + 1e-6:
            raise ValueError(f"not enough cash: need {budget:,.2f}, have {self.cash:,.2f}")
        notional = budget / (1 + self.fee_rate)
        fee = notional * self.fee_rate
        units = notional / price
        self.cash -= notional + fee
        lot = Lot(self._next_id, symbol, units, date, price, notional + fee,
                  date + pd.Timedelta(days=holding_days))
        self.lots[lot.lot_id] = lot
        self._next_id += 1
        self.fills.append({"date": date, "side": "buy", "lot_id": lot.lot_id, "symbol": symbol,
                           "units": units, "price": price, "notional": notional, "fee": fee,
                           "cash_change": -(notional + fee), "pnl": np.nan})
        return lot

    def sell(self, lot_id, price, date):
        """Close one lot at price"""
        lot = self.lots.pop(lot_id)
        notional = lot.units * price
        fee = notional * self.fee_rate
        self.cash += notional - fee
        pnl = notional - fee - lot.cost
        self.fills.append({"date": date, "side": "sell", "lot_id": lot_id, "symbol": lot.symbol,
                           "units": lot.units, "price": price, "notional": notional, "fee": fee,
                           "cash_change": notional - fee, "pnl": pnl})
        return pnl

    def held_symbols(self):
        return {lot.symbol for lot in self.lots.values()}

    def holdings_value(self, marks):
        return sum(lot.units * marks[lot.symbol] for lot in self.lots.values())

class TopNStrategy:
    """Each signal date: buy the top-N scored stocks, equal weight"""

    def __init__(self, scores, top_n=30, cohort_fraction=1 / 3, min_price=5.0, min_order=1_000):
        self.top_n = top_n
        self.cohort_fraction = cohort_fraction
        self.min_price = min_price
        self.min_order = min_order
        self.by_date = {d: g.set_index("symbol") for d, g in scores.groupby("date")}

    def is_signal_date(self, date):
        return date in self.by_date

    def orders(self, signal_date, cash, nav, held):
        """Return budget, using only signal-date information"""
        today = self.by_date.get(signal_date)
        if today is None:
            return {}
        candidates = today.loc[today.close.ge(self.min_price) & ~today.index.isin(held)]
        picks = (candidates.reset_index()
                 .sort_values(["score", "symbol"], ascending=[False, True])
                 .head(self.top_n).symbol)
        budget = min(cash, nav * self.cohort_fraction)
        if len(picks) == 0 or budget / len(picks) < self.min_order:  
            return {}
        return {s: budget / len(picks) for s in picks}

def run_backtest(quotes, strategy, portfolio, holding_days=90):
    """Advance one session each time, for each session:
      1. sell lots whose exit date arrived (needs a quote today)
      2. fill orders decided at the previous close (skip a stock with no quote)
      3. mark to market with the latest known prices, record the day
      4. if t is a signal date, decide orders for session t+1 using only data through t
    """
    days = quotes.index
    marks, pending, log, rows, waiting = {}, {}, [], [], set()
    for i, day in enumerate(days):
        today = quotes.loc[day].dropna()
        for symbol, price in today.items():
            if symbol in marks and price > 3 * marks[symbol] and symbol in portfolio.held_symbols():
                log.append({"date": day, "symbol": symbol, "event": "suspicious_jump_x3"})
            marks[symbol] = price

        # 1. maturities
        for lot in list(portfolio.lots.values()):
            if day >= lot.exit_date:
                if lot.symbol in today.index:
                    portfolio.sell(lot.lot_id, today[lot.symbol], day)
                    if day > lot.exit_date + pd.Timedelta(days=7):
                        log.append({"date": day, "symbol": lot.symbol, "event": "late_exit"})
                elif lot.lot_id not in waiting:          # log once per lot
                    waiting.add(lot.lot_id)
                    log.append({"date": day, "symbol": lot.symbol, "event": "exit_waiting"})

        # 2. fills of yesterday's decisions
        for symbol, budget in pending.items():
            if symbol not in today.index:
                log.append({"date": day, "symbol": symbol, "event": "skipped_no_quote"})
                continue
            portfolio.buy(symbol, min(budget, portfolio.cash), today[symbol], day, holding_days)
        pending = {}

        # 3. record
        held_value = portfolio.holdings_value(marks)
        stale = sum(1 for s in portfolio.held_symbols() if s not in today.index)
        rows.append({"date": day, "cash": portfolio.cash, "holdings": held_value,
                     "nav": portfolio.cash + held_value, "open_lots": len(portfolio.lots),
                     "stale_marks": stale})

        # 4. decide for tomorrow (nothing to fill after the last session)
        if i + 1 < len(days) and strategy.is_signal_date(day):
            pending = strategy.orders(day, portfolio.cash, portfolio.cash + held_value,
                                      portfolio.held_symbols())

    history = pd.DataFrame(rows).set_index("date")
    fills = pd.DataFrame(portfolio.fills)
    open_lots = pd.DataFrame([vars(l) for l in portfolio.lots.values()])
    return history, fills, open_lots, pd.DataFrame(log, columns=["date", "symbol", "event"])


def performance(nav, annual_rf=0.02):
    """Final value, CAGR, max drawdown, Sharpe (daily returns, 252 sessions/yr)"""
    nav = nav.dropna()
    daily = nav.pct_change().dropna()
    years = (nav.index[-1] - nav.index[0]).days / 365.25
    excess = daily - annual_rf / 252
    return pd.Series({
        "final_value": nav.iloc[-1],
        "total_return": nav.iloc[-1] / nav.iloc[0] - 1,
        "CAGR": (nav.iloc[-1] / nav.iloc[0]) ** (1 / years) - 1,
        "max_drawdown": (nav / nav.cummax() - 1).min(),
        "annual_vol": daily.std() * np.sqrt(252),
        "sharpe": excess.mean() / daily.std() * np.sqrt(252),
    })

def benchmark_nav(database, start, end, initial_cash=1_000_000):
    """$1M in ^SP500TR bought at the first test close, held to the end"""
    with duckdb.connect(str(database), read_only=True) as con:
        b = con.execute("SELECT date, adj_close FROM benchmark WHERE date BETWEEN ? AND ? ORDER BY date",
                        [start, end]).fetchdf()
    b = b.set_index(pd.to_datetime(b.date)).adj_close
    return initial_cash * b / b.iloc[0]