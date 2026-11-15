"""Daily cash and adjusted-unit accounting for precommitted close orders."""
from dataclasses import dataclass
from datetime import timedelta
import math

import pandas as pd

CAPITAL = 1_000_000.0
FEE = 0.001
HOLD = timedelta(days=90)
NAMES = 8
SLEEVE = 1 / 14


@dataclass
class Lot:
    symbol: str
    entry: pd.Timestamp
    units: float
    mark: float
    last_quote: pd.Timestamp

    @property
    def maturity(self):
        return self.entry + HOLD


def simulate(prices, sessions, rankings, fee=FEE, capital=CAPITAL,
             names=NAMES, sleeve=SLEEVE):
    """Rankings map entry session to (prior close, ranked symbols)."""
    cash = capital
    book = {}
    trades, daily = [], []
    for i, day in enumerate(sessions):
        # Plan with yesterday's portfolio. Today's sale proceeds cannot fund orders.
        orders = []
        if day in rankings:
            signal, candidates = rankings[day]
            assert prices.index[prices.index.get_loc(day) - 1] == signal < day
            prior_value = cash + sum(lot.units * lot.mark for lot in book.values())
            selected = [s for s in candidates if s not in book][:names]
            budget = min(cash, prior_value * sleeve)
            if selected and budget > 0.01:
                orders = [(symbol, budget / len(selected)) for symbol in selected]

        quote = prices.loc[day]
        for symbol, lot in list(book.items()):
            price = quote[symbol]
            if math.isfinite(price):
                lot.mark = price
                lot.last_quote = day
                if day >= lot.maturity:
                    gross = lot.units * price
                    cash += gross * (1 - fee)
                    trades.append(dict(side='sell', symbol=symbol, date=day,
                                       signal=None, entry=lot.entry, maturity=lot.maturity,
                                       price=price, units=lot.units, fee=gross * fee,
                                       cash_flow=gross * (1 - fee)))
                    del book[symbol]
        for symbol, budget in orders:
            price = quote[symbol]
            if not math.isfinite(price):
                trades.append(dict(side='unfilled_buy', symbol=symbol, date=day,
                                   signal=signal, cash_flow=0.0))
                continue
            gross = budget / (1 + fee)
            units = gross / price
            cash -= budget
            book[symbol] = Lot(symbol, day, units, price, day)
            trades.append(dict(side='buy', symbol=symbol, date=day,
                               signal=signal, entry=day, maturity=day + HOLD,
                               price=price, units=units, fee=gross * fee,
                               cash_flow=-budget))
        assert cash >= -1e-7
        cash = max(0.0, cash)
        invested = sum(lot.units * lot.mark for lot in book.values())
        overdue = [lot for lot in book.values() if day >= lot.maturity]
        stale = [lot for lot in book.values() if lot.last_quote < day]
        daily.append(dict(date=day, nav=cash + invested, cash=cash,
                          invested=invested, positions=len(book),
                          overdue=len(overdue), stale=len(stale),
                          stale_value=sum(lot.units * lot.mark for lot in stale)))
    open_lots = [dict(symbol=lot.symbol, entry=lot.entry, maturity=lot.maturity,
                      units=lot.units, mark=lot.mark, last_quote=lot.last_quote,
                      value=lot.units * lot.mark) for lot in book.values()]
    daily = pd.DataFrame(daily).set_index('date')
    trades = pd.DataFrame(trades)
    if len(trades):
        cash_check = trades.groupby('date').cash_flow.sum().reindex(sessions, fill_value=0)
        assert (daily.cash - (capital + cash_check.cumsum())).abs().max() < 1e-5
    return daily, trades, pd.DataFrame(open_lots)


def metrics(nav, capital=CAPITAL):
    returns = nav / nav.shift(fill_value=capital) - 1
    drawdown = nav / nav.cummax().clip(lower=capital) - 1
    years = (nav.index[-1] - nav.index[0]).days / 365.25
    rf = 1.02 ** (1 / 252) - 1
    return dict(final_value=float(nav.iloc[-1]),
                cagr=float((nav.iloc[-1] / capital) ** (1 / years) - 1),
                max_drawdown=float(drawdown.min()),
                sharpe=float((returns.mean() - rf) / returns.std() * math.sqrt(252)))


def hand_checks():
    dates = pd.bdate_range('2020-01-02', '2020-04-10')
    entry = dates[1]
    due = entry + HOLD
    exit_day = due + timedelta(days=1)
    prices = pd.DataFrame({'A': 10.0, 'B': 20.0}, index=dates)
    prices.loc[due, 'A'] = float('nan')
    prices.loc[exit_day:, 'A'] = 12
    plans = {entry: (dates[0], ['A']),
             exit_day: (due, ['B']),
             dates[dates.get_loc(exit_day) + 1]: (exit_day, ['B'])}
    daily, trades, lots = simulate(prices, dates[1:], plans, fee=0.01,
                                   capital=101, names=1, sleeve=1)
    assert abs(daily.loc[entry, 'nav'] - 100) < 1e-10
    assert daily.loc[due, 'cash'] == 0 and daily.loc[due, 'overdue'] == 1
    assert abs(daily.loc[exit_day, 'cash'] - 118.8) < 1e-10
    assert list(trades.side) == ['buy', 'sell', 'buy']
    assert lots.symbol.tolist() == ['B']
    assert abs(daily.nav.iloc[-1] - 118.8 / 1.01) < 1e-10
    prices.loc[entry, 'A'] = float('nan')
    daily, trades, _ = simulate(prices, dates[1:], {entry: plans[entry]},
                                fee=0.01, capital=101, names=1, sleeve=1)
    assert daily.nav.eq(101).all() and trades.side.tolist() == ['unfilled_buy']

    # Friday entry + 90 days can also land on a closed market session.
    monday = pd.Timestamp('2020-01-06')
    prices.loc[monday, 'A'] = 10
    weekend_due = monday + HOLD
    assert weekend_due.weekday() == 6
    daily, trades, _ = simulate(prices, dates[1:],
                                {monday: (entry, ['A'])}, capital=100)
    assert trades.loc[trades.side.eq('sell'), 'date'].item() == pd.Timestamp('2020-04-06')
    print('PASS: $101 buys $100 exposure; delayed maturity sells for $118.80 net;')
    print('sale cash is reusable next session; open lot marked; missing entry skipped;')
    print('Sunday maturity rolls to Monday; daily cash reconciles to the trade ledger.')
