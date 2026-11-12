"""A small daily backtester: strategy decides, account books, loop advances time.

Written for HW4.  The structure (separate strategy / order-booking / daily loop /
analytics pieces, fee-inclusive budgets, adjusted-price "units") follows the ideas
in Lecture 8 (``Lectures/modeling/backtest.py``); the code here is my own.

Conventions
-----------
* Positions are held in **adjusted-price units**: buying $X at adjusted close P
  gives X/P units, so dividends are already inside the price path and are never
  added again.
* Each purchase is an independent **lot** with its own 90-calendar-day maturity.
  Two lots of the same stock never merge.
* Day ``d`` processing order (all at the close):
    1. record today's quotes;
    2. sell every lot whose maturity is <= d *and* that has a quote today
       (if the stock has no quote, the lot waits for the next quoted close);
    3. buy the orders the strategy planned after the **previous** close.  Those
       budgets were sized from the previous close's cash, which can only have
       grown since (step 2 adds cash), so cash is never spent twice;
    4. value the account: cash + units x last observed close.
* Fees: ``fee_bps`` of notional on every buy and sell.  Budgets include the
  entry fee.  Cash earns no interest.
* No fill is ever invented: a buy order for a stock without a quote on the fill
  day is cancelled (cash stays), and a sale only happens at an observed close.
"""

from dataclasses import dataclass, asdict, field

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
    entry_fee: float
    due_date: pd.Timestamp


@dataclass
class Snapshot:
    """What the strategy may see when it plans orders (as of the signal close)."""
    date: pd.Timestamp
    cash: float
    nav: float
    holdings: pd.Series      # market value by symbol, marked at last observed close


class Account:
    """Cash, open lots and the fill ledger."""

    def __init__(self, initial_cash=1_000_000.0, fee_bps=10.0, holding_days=90):
        self.initial_cash = float(initial_cash)
        self.cash = float(initial_cash)
        self.fee = fee_bps / 10_000
        self.holding_days = holding_days
        self.lots: dict[int, Lot] = {}
        self.fills: list[dict] = []
        self._next_id = 0

    def buy(self, symbol, budget, price, date, signal_date):
        """Spend ``budget`` dollars *including* the fee."""
        if not (np.isfinite(price) and price > 0 and np.isfinite(budget) and budget > 0):
            raise ValueError("buy needs a positive finite price and budget")
        if signal_date >= date:
            raise ValueError("a decision must precede its fill")
        if budget > self.cash + 1e-6:
            raise ValueError(f"insufficient cash: budget {budget:,.2f} > cash {self.cash:,.2f}")
        units = budget / (price * (1 + self.fee))
        notional = units * price
        fee = notional * self.fee
        self.cash -= notional + fee
        if self.cash < 0:            # only floating-point dust can get here
            self.cash = 0.0
        lot = Lot(self._next_id, symbol, units, signal_date, date, price, fee,
                  date + pd.Timedelta(days=self.holding_days))
        self.lots[lot.lot_id] = lot
        self._next_id += 1
        self.fills.append(dict(date=date, side="BUY", lot_id=lot.lot_id, symbol=symbol, units=units,
                               price=price, notional=notional, fee=fee, cash_change=-(notional + fee),
                               signal_date=signal_date, due_date=lot.due_date, pnl=0.0))
        return lot

    def sell(self, lot_id, price, date):
        lot = self.lots.pop(lot_id)
        if date < lot.due_date:
            raise ValueError("lots are only sold at maturity")
        notional = lot.units * price
        fee = notional * self.fee
        pnl = notional - fee - lot.units * lot.entry_price - lot.entry_fee
        self.cash += notional - fee
        self.fills.append(dict(date=date, side="SELL", lot_id=lot.lot_id, symbol=lot.symbol, units=lot.units,
                               price=price, notional=notional, fee=fee, cash_change=notional - fee,
                               signal_date=lot.signal_date, due_date=lot.due_date, pnl=pnl,
                               entry_date=lot.entry_date, entry_price=lot.entry_price,
                               days_held=(date - lot.entry_date).days))
        return pnl

    def holdings(self, marks):
        values = {}
        for lot in self.lots.values():
            values[lot.symbol] = values.get(lot.symbol, 0.0) + lot.units * marks[lot.symbol]
        return pd.Series(values, dtype=float)


class TopNStrategy:
    """Buy the top-N predicted stocks on each signal date.

    ``scores`` has columns date, symbol, score, sector; one row per tradable
    stock on a signal date, computed with information up to that close.

    Sizing: the week's cohort budget is ``min(cash, cohort_fraction x NAV)``, split
    equally over N names.  With weekly signals and 90-day holds about 13 cohorts
    overlap, so ``cohort_fraction = 1/12`` keeps the book close to fully invested
    while staggering maturities.  Concentration limits (including planned buys):
    one name <= ``name_cap`` of NAV, one sector <= ``sector_cap`` of NAV.
    """

    def __init__(self, scores, sectors, top_n=25, cohort_fraction=1 / 12,
                 name_cap=0.05, sector_cap=0.30):
        self.scores = {d: g.sort_values(["score", "symbol"], ascending=[False, True])
                       for d, g in scores.groupby("date")}
        self.sectors = sectors
        self.top_n, self.cohort_fraction = top_n, cohort_fraction
        self.name_cap, self.sector_cap = name_cap, sector_cap

    def orders(self, signal_date, snapshot: Snapshot):
        ranked = self.scores.get(signal_date)
        if ranked is None or snapshot.cash <= 0:
            return pd.Series(dtype=float)
        budget = min(snapshot.cash, self.cohort_fraction * snapshot.nav)
        per_name = budget / self.top_n
        held = snapshot.holdings.to_dict()
        sector_held = snapshot.holdings.groupby(self.sectors.reindex(snapshot.holdings.index)).sum().to_dict()
        orders = {}
        for symbol, sector in zip(ranked.symbol, ranked.sector):
            if len(orders) == self.top_n or budget <= 1.0:
                break
            room = min(self.name_cap * snapshot.nav - held.get(symbol, 0.0),
                       self.sector_cap * snapshot.nav - sector_held.get(sector, 0.0))
            amount = min(per_name, budget, room)
            if amount < 0.25 * per_name:          # skip names that are already (nearly) at a cap
                continue
            orders[symbol] = amount
            budget -= amount
            held[symbol] = held.get(symbol, 0.0) + amount
            sector_held[sector] = sector_held.get(sector, 0.0) + amount
        return pd.Series(orders, dtype=float)


@dataclass
class BacktestResult:
    history: pd.DataFrame
    fills: pd.DataFrame
    open_lots: pd.DataFrame
    events: pd.DataFrame
    account: Account = field(repr=False)


class Backtester:
    def __init__(self, prices: pd.DataFrame, strategy, initial_cash=1_000_000.0, fee_bps=10.0,
                 holding_days=90, stale_after=5, settle_stale_after=None):
        """``prices``: date x symbol adjusted closes on the market calendar, NaN = no quote.

        ``settle_stale_after`` (default None = never) is a *sensitivity* switch: a
        matured lot whose stock has had no quote for that many sessions is settled
        at its last observed close.  That is an assumed (invented) fill, so it is
        never used for the official result.
        """
        self.prices = prices
        self.calendar = prices.index
        self.strategy = strategy
        self.account = Account(initial_cash, fee_bps, holding_days)
        self.stale_after = stale_after
        self.settle_stale_after = settle_stale_after

    def _snapshot(self, date, marks):
        h = self.account.holdings(marks)
        return Snapshot(date, self.account.cash, self.account.cash + h.sum(), h)

    def run(self, start, end):
        acct = self.account
        days = self.calendar[(self.calendar >= pd.Timestamp(start)) & (self.calendar <= pd.Timestamp(end))]
        first = self.calendar.get_loc(days[0])
        assert first > 0, "need the session before start as the first signal date"
        px = self.prices.to_numpy()
        cols = self.prices.columns
        col_of = {s: i for i, s in enumerate(cols)}
        marks, last_seen = {}, {}
        events, history = [], []
        snapshot = Snapshot(self.calendar[first - 1], acct.cash, acct.cash, pd.Series(dtype=float))
        for day in days:
            i = self.calendar.get_loc(day)
            signal_date = self.calendar[i - 1]
            # Plan with the previous close only (the strategy is never given today's prices).
            orders = self.strategy.orders(signal_date, snapshot)
            if orders.sum() > snapshot.cash + 1e-6:
                raise ValueError("strategy asked for more cash than it had")

            row = px[i]
            def quote(sym):
                v = row[col_of[sym]]
                return v if np.isfinite(v) else None
            for sym in {lot.symbol for lot in acct.lots.values()} | set(orders.index):
                q = quote(sym)
                if q is not None:
                    marks[sym], last_seen[sym] = q, i

            # 1) Maturity exits at today's close (if the stock traded today).
            for lot in sorted(acct.lots.values(), key=lambda l: l.lot_id):
                if day >= lot.due_date:
                    q = quote(lot.symbol)
                    if q is not None:
                        acct.sell(lot.lot_id, q, day)
                    elif (self.settle_stale_after is not None
                          and i - last_seen[lot.symbol] > self.settle_stale_after):
                        acct.sell(lot.lot_id, marks[lot.symbol], day)
                        events.append(dict(date=day, symbol=lot.symbol, lot_id=lot.lot_id, event="assumed_settlement_at_last_close"))
                    else:
                        events.append(dict(date=day, symbol=lot.symbol, lot_id=lot.lot_id, event="exit_delayed_no_quote"))
            # 2) New entries planned after the previous close.
            for sym, budget in orders.items():
                q = quote(sym)
                if q is None:
                    events.append(dict(date=day, symbol=sym, lot_id=-1, event="buy_cancelled_no_quote"))
                    continue
                acct.buy(sym, budget, q, day, signal_date)
            # 3) Mark to market with the last observed close; flag stale marks.
            stale = [l for l in acct.lots.values() if i - last_seen[l.symbol] > self.stale_after]
            snapshot = self._snapshot(day, marks)
            history.append(dict(date=day, cash=acct.cash, holdings=snapshot.nav - acct.cash, nav=snapshot.nav,
                                open_lots=len(acct.lots), stale_lots=len(stale),
                                stale_value=sum(l.units * marks[l.symbol] for l in stale)))
        open_lots = pd.DataFrame([asdict(l) for l in acct.lots.values()])
        if len(open_lots):
            open_lots["mark"] = open_lots.symbol.map(marks)
            open_lots["last_quote_date"] = open_lots.symbol.map(lambda s: self.calendar[last_seen[s]])
            open_lots["value"] = open_lots.units * open_lots.mark
        return BacktestResult(pd.DataFrame(history).set_index("date"), pd.DataFrame(acct.fills),
                              open_lots, pd.DataFrame(events, columns=["date", "symbol", "lot_id", "event"]), acct)


# --------------------------------------------------------------------------- analytics
def performance(nav: pd.Series, initial=1_000_000.0, rf=0.02):
    """Final value, CAGR, max drawdown, Sharpe (daily, 252 sessions, 2% rf)."""
    nav = nav.dropna()
    returns = nav.pct_change().dropna()
    excess = returns - ((1 + rf) ** (1 / 252) - 1)
    years = (nav.index[-1] - nav.index[0]).days / 365.25
    return pd.Series({
        "final_value": nav.iloc[-1],
        "total_return": nav.iloc[-1] / initial - 1,
        "CAGR": (nav.iloc[-1] / initial) ** (1 / years) - 1,
        "max_drawdown": (nav / nav.cummax() - 1).min(),
        "annual_volatility": returns.std() * np.sqrt(252),
        "Sharpe": np.sqrt(252) * excess.mean() / returns.std(),
    })
