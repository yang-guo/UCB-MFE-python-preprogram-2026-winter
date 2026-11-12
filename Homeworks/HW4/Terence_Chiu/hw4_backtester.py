"""My HW4 backtester: daily accounting for cash, open lots, trades and portfolio value."""
from dataclasses import dataclass

import pandas as pd


@dataclass
class Lot:
    """One purchase, held until its maturity date."""

    symbol: str
    units: float
    entry_date: object
    entry_price: float
    maturity_date: object


class Backtester:
    def __init__(self, prices, entries, initial_cash, fee_rate=0.0005, annual_rf=0.02):
        """
        prices: DataFrame. Index = trading days (dates, or ints for the hand check),
            columns = symbols, values = adj_close. NaN means no quote that day.
        entries: list of (entry_date, maturity_date, symbols_to_buy), using prices' index labels.
        initial_cash: starting cash.
        fee_rate: cost per side as a fraction of trade value (0.0005 = 5 bps).
        annual_rf: annual interest earned on idle cash.
        """
        self.prices = prices
        self.entries = {entry_date: (maturity_date, symbols) for entry_date, maturity_date, symbols in entries}
        self.initial_cash = initial_cash
        self.fee_rate = fee_rate
        self.annual_rf = annual_rf

        self.cash = initial_cash
        self.lots = []        # open Lot objects
        self.last_price = {}  # symbol -> last observed price
        self.trades = []      # one dict per buy or sell
        self.history = []     # one dict per day
        self.skipped_buys = 0    # picked symbols with no quote on their entry day
        self.delayed_sells = 0   # lot-days a matured lot waited for a quote

    def run_backtest(self):
        """Process every day in prices.index; return (history, trades) as DataFrames.

        Order each day:
            1. accrue_interest (skip on the first day)
            2. update_prices
            3. sell_matured
            4. buy_entries
            5. record
        """
        for day in self.prices.index:
            if day != self.prices.index[0]:
                self.accrue_interest()
            self.update_prices(day)
            self.sell_matured(day)
            self.buy_entries(day)
            self.record(day)

        return pd.DataFrame(self.history), pd.DataFrame(self.trades)

    def accrue_interest(self):
        """Grow cash by one trading day of interest at annual_rf (252 trading days a year)."""
        self.cash = self.cash * (1 + self.annual_rf / 252)

    def update_prices(self, day):
        """Store today's quote for every symbol that has one."""
        today = self.prices.loc[day].dropna().to_dict()
        self.last_price.update(today)

    def sell_matured(self, day):
        """Sell each lot whose maturity has arrived and that has a quote today.

        Cash in: units * price * (1 - fee_rate). Matured lots with no quote stay held
        (no invented fills) and are sold at their next quoted close.
        """
        kept_lots = self.lots.copy()

        for lot in self.lots:
            if lot.maturity_date <= day:
                price = self.prices.at[day, lot.symbol]
                if pd.notna(price):
                    fee = lot.units * price * self.fee_rate
                    self.cash += lot.units * price - fee
                    kept_lots.remove(lot)
                    self.trades.append({"day": day, "side": "sell", "symbol": lot.symbol,
                                        "units": lot.units, "price": price, "fee": fee})
                else:
                    self.delayed_sells += 1

        self.lots = kept_lots

    def buy_entries(self, day):
        """If day is an entry date, buy its symbols that have a quote today.

        Equal budget per symbol from current cash; units = budget / (price * (1 + fee_rate)).
        Never let cash go below zero.
        """
        if day not in self.entries:
            return

        maturity_date, symbols = self.entries[day]
        n = sum(1 for symbol in symbols if pd.notna(self.prices.at[day, symbol]))
        budget = self.cash / n

        for symbol in symbols:
            price = self.prices.at[day, symbol]
            if pd.notna(price):
                units = budget / (price * (1 + self.fee_rate))
                fee = units * price * self.fee_rate
                self.lots.append(Lot(symbol, units, day, price, maturity_date))
                self.cash -= units * price + fee
                self.trades.append({"day": day, "side": "buy", "symbol": symbol,
                                    "units": units, "price": price, "fee": fee})
            else:
                self.skipped_buys += 1

    def record(self, day):
        """Append today's cash, positions value (units * last price), total value,
        and number of stale positions (held but no quote today).
        """
        self.history.append({
            "day": day,
            "cash": self.cash,
            "value": self.cash + sum(lot.units * self.last_price[lot.symbol] for lot in self.lots),
            "n_stale": sum(1 for lot in self.lots if pd.isna(self.prices.at[day, lot.symbol])),
        })
