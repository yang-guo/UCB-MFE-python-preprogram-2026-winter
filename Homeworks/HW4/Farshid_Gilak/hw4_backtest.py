"""A daily, lot-based, long-only backtester for HW4.

Written for this homework. Its structure (strategy / order book / daily loop /
analytics, fee-inclusive dollar budgets, adjusted-price units, sector and name
caps) follows the design discussed in Lecture 8 (Lectures/modeling/backtest.py);
the cap allocation is adapted from its capped_budgets function.

Timing contract: on session t the strategy receives the portfolio valued at
the close of session t-1 and returns fee-inclusive dollar budgets. The
backtester then observes session t's closes, sells matured lots, and fills
the purchases at those closes. Budgets may not exceed the cash held at t-1's
close, so the same cash can never be spent twice.
"""

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Lot:
    """One purchase with its own cost basis and calendar-day maturity."""

    lot_id: int
    symbol: str
    units: float
    signal_date: pd.Timestamp
    entry_date: pd.Timestamp
    entry_price: float
    entry_fee: float
    due_date: pd.Timestamp


@dataclass(frozen=True)
class Snapshot:
    """Account state at a close, given to the strategy for planning."""

    date: pd.Timestamp
    cash: float
    nav: float
    holdings: pd.Series  # Market value by symbol.


class Ledger:
    """Cash, open lots, fills, fees, and realized PnL."""

    FILL_COLUMNS = ["date", "signal_date", "lot_id", "symbol", "side", "price", "units",
                    "notional", "fee", "cash_change", "due_date", "realized_pnl"]

    def __init__(self, initial_cash=1_000_000, fee_bps=10):
        if initial_cash <= 0 or fee_bps < 0:
            raise ValueError("Capital must be positive and fees nonnegative")
        self.initial_cash = float(initial_cash)
        self.cash = float(initial_cash)
        self.fee_rate = fee_bps / 10_000
        self.lots = {}
        self.fills = []
        self.realized_pnl = 0.
        self.total_fees = 0.
        self._next_id = 0

    def buy(self, symbol, budget, price, date, signal_date, holding_days):
        """Spend a fee-inclusive budget at the fill price."""
        if not (np.isfinite(price) and price > 0 and np.isfinite(budget) and budget > 0):
            raise ValueError("Price and budget must be finite and positive")
        if not signal_date < date:
            raise ValueError("The decision must precede the fill")
        if budget > self.cash + 1e-6:
            raise ValueError("Insufficient cash")
        units = budget / (price * (1 + self.fee_rate))
        notional = units * price
        fee = notional * self.fee_rate
        lot = Lot(self._next_id, symbol, units, signal_date, date, price, fee,
                  date + pd.Timedelta(days=holding_days))
        self._next_id += 1
        self.lots[lot.lot_id] = lot
        self.cash = max(0., self.cash - notional - fee)
        self._record(lot, "BUY", date, price, notional, fee, -(notional + fee), 0.)
        return lot

    def sell(self, lot_id, price, date):
        """Close a whole lot at the fill price."""
        lot = self.lots.pop(lot_id)
        if not (np.isfinite(price) and price > 0) or date < lot.entry_date:
            raise ValueError("Invalid sale")
        notional = lot.units * price
        fee = notional * self.fee_rate
        pnl = notional - fee - lot.units * lot.entry_price - lot.entry_fee
        self.cash += notional - fee
        self.realized_pnl += pnl
        self._record(lot, "SELL", date, price, notional, fee, notional - fee, pnl)
        return pnl

    def _record(self, lot, side, date, price, notional, fee, cash_change, pnl):
        self.total_fees += fee
        self.fills.append({"date": date, "signal_date": lot.signal_date, "lot_id": lot.lot_id,
                           "symbol": lot.symbol, "side": side, "price": price, "units": lot.units,
                           "notional": notional, "fee": fee, "cash_change": cash_change,
                           "due_date": lot.due_date, "realized_pnl": pnl})

    def holdings(self, marks):
        values = {}
        for lot in self.lots.values():
            values[lot.symbol] = values.get(lot.symbol, 0.) + lot.units * marks[lot.symbol]
        return pd.Series(values, dtype=float)

    def unrealized_pnl(self, marks):
        return sum(lot.units * marks[lot.symbol] - lot.units * lot.entry_price - lot.entry_fee
                   for lot in self.lots.values())


class Strategy:
    """Interface: plan purchases from information at the previous close."""

    holding_days = 90

    def generate_orders(self, signal_date, execution_date, snapshot):
        raise NotImplementedError

    def should_sell(self, lot, date):
        return date >= lot.due_date


class TopScoreStrategy(Strategy):
    """Buy the highest predicted 90-day returns on scheduled signal dates.

    The first `cohorts` entries ramp in with equal budgets (cash / entries
    left in the ramp); afterwards each entry spends all available cash, which
    is mostly the proceeds of the lots that matured since the last entry. The
    budget is split equally over top_n names, subject to name and sector caps
    measured on NAV at the signal close. Names skipped by a cap are replaced
    by the next-ranked name.
    """

    def __init__(self, scores, sectors, top_n=25, cohorts=13, max_rank=1000,
                 name_cap=0.05, sector_cap=0.30, fee_bps=10, min_order=100.):
        required = {"date", "symbol", "score", "universe_rank", "tradable"}
        if not required.issubset(scores.columns):
            raise ValueError(f"Scores need columns {sorted(required)}")
        if scores.duplicated(["date", "symbol"]).any() or not np.isfinite(scores.score).all():
            raise ValueError("Scores must be finite and unique by date and symbol")
        usable = scores.loc[scores.tradable & scores.universe_rank.le(max_rank)]
        self.ranked = {date: g.sort_values(["score", "symbol"], ascending=[False, True]).symbol.to_list()
                       for date, g in usable.groupby("date")}
        self.sectors = sectors
        self.top_n, self.cohorts = top_n, cohorts
        self.name_cap, self.sector_cap = name_cap, sector_cap
        self.fee_rate = fee_bps / 10_000
        self.min_order = min_order
        self.entries = 0

    def generate_orders(self, signal_date, execution_date, snapshot):
        ranked = self.ranked.get(signal_date)
        if not ranked:
            return pd.Series(dtype=float)
        budget = snapshot.cash / max(self.cohorts - self.entries, 1)
        self.entries += 1
        return capped_allocation(ranked, budget, snapshot, self.sectors, self.top_n,
                                 self.name_cap, self.sector_cap, self.fee_rate, self.min_order)


def capped_allocation(ranked, budget, snapshot, sectors, top_n, name_cap, sector_cap, fee_rate, min_order):
    """Equal fee-inclusive slices down a ranked list, respecting exposure caps."""
    per_name = budget / top_n
    name_exposure = snapshot.holdings.to_dict()
    sector_exposure = snapshot.holdings.groupby(sectors.reindex(snapshot.holdings.index)).sum().to_dict()
    orders, remaining = {}, budget
    for symbol in ranked:
        if len(orders) == top_n or remaining < min_order:
            break
        sector = sectors.get(symbol, "Unknown")
        room = min(name_cap * snapshot.nav - name_exposure.get(symbol, 0.),
                   sector_cap * snapshot.nav - sector_exposure.get(sector, 0.))
        amount = min(per_name, remaining, max(room, 0.) * (1 + fee_rate))
        if amount < min_order:
            continue
        orders[symbol] = amount
        remaining -= amount
        notional = amount / (1 + fee_rate)
        name_exposure[symbol] = name_exposure.get(symbol, 0.) + notional
        sector_exposure[sector] = sector_exposure.get(sector, 0.) + notional
    return pd.Series(orders, dtype=float)


@dataclass(frozen=True)
class BacktestResult:
    history: pd.DataFrame
    fills: pd.DataFrame
    open_lots: pd.DataFrame
    diagnostics: pd.DataFrame
    marks: pd.Series
    last_quote: pd.Series
    pnl: pd.Series


class Backtester:
    """Advance one market session at a time."""

    def __init__(self, adj_close, benchmark, strategy, ledger, stale_limit=5):
        if not adj_close.index.equals(benchmark.index):
            raise ValueError("Prices and benchmark must share the market calendar")
        self.quotes = adj_close
        self.benchmark = benchmark
        self.calendar = adj_close.index
        self.strategy = strategy
        self.ledger = ledger
        self.stale_limit = stale_limit
        if ledger.fills or ledger.lots:
            raise ValueError("Use a fresh ledger for each run")

    def run(self, start, end):
        calendar = self.calendar
        days = calendar[(calendar >= pd.Timestamp(start)) & (calendar <= pd.Timestamp(end))]
        first = calendar.get_loc(days[0])
        if first == 0:
            raise ValueError("Need a signal session before the first test session")
        symbols = self.quotes.columns
        quotes = self.quotes.to_numpy()
        marks, last_seen, diagnostics = {}, {}, []
        ledger = self.ledger
        snapshot = Snapshot(calendar[first - 1], ledger.cash, ledger.cash, pd.Series(dtype=float))
        history = [{"date": snapshot.date, "cash": ledger.cash, "holdings_value": 0., "nav": ledger.cash,
                    "open_lots": 0, "buys": 0, "sells": 0, "fees": 0., "stale_holdings": 0}]
        for day in days:
            i = calendar.get_loc(day)
            signal_date = calendar[i - 1]
            orders = self.strategy.generate_orders(signal_date, day, snapshot)
            if orders.lt(0).any() or orders.sum() > snapshot.cash + 1e-6:
                raise ValueError(f"Orders on {day.date()} exceed the cash known at {signal_date.date()}")

            row = quotes[i]
            observed = {symbols[j]: row[j] for j in np.flatnonzero(np.isfinite(row))}
            for symbol in {lot.symbol for lot in ledger.lots.values()} | set(orders.index):
                if symbol in observed:
                    marks[symbol], last_seen[symbol] = observed[symbol], day

            n_fills = len(ledger.fills)
            for lot in list(ledger.lots.values()):
                if self.strategy.should_sell(lot, day):
                    if lot.symbol in observed:
                        ledger.sell(lot.lot_id, observed[lot.symbol], day)
                    else:
                        diagnostics.append({"date": day, "symbol": lot.symbol, "lot_id": lot.lot_id,
                                            "event": "exit_delayed_no_quote"})
            for symbol, budget in orders.items():
                if symbol not in observed:
                    diagnostics.append({"date": day, "symbol": symbol, "lot_id": np.nan,
                                        "event": "buy_rejected_no_quote"})
                    continue
                ledger.buy(symbol, budget, observed[symbol], day, signal_date, self.strategy.holding_days)

            held = {lot.symbol for lot in ledger.lots.values()}
            stale = [s for s in held if last_seen[s] != day]
            for symbol in stale:
                if calendar.get_loc(day) - calendar.get_loc(last_seen[symbol]) == self.stale_limit + 1:
                    diagnostics.append({"date": day, "symbol": symbol, "lot_id": np.nan,
                                        "event": f"no_quote_for_{self.stale_limit + 1}_sessions"})
            holdings = ledger.holdings(marks)
            snapshot = Snapshot(day, ledger.cash, ledger.cash + holdings.sum(), holdings)
            new_fills = ledger.fills[n_fills:]
            history.append({"date": day, "cash": ledger.cash, "holdings_value": holdings.sum(),
                            "nav": snapshot.nav, "open_lots": len(ledger.lots),
                            "buys": sum(f["side"] == "BUY" for f in new_fills),
                            "sells": sum(f["side"] == "SELL" for f in new_fills),
                            "fees": sum(f["fee"] for f in new_fills), "stale_holdings": len(stale)})

        history = pd.DataFrame(history).set_index("date")
        bench = self.benchmark.loc[days]
        history["benchmark_nav"] = ledger.initial_cash
        history.loc[days, "benchmark_nav"] = ledger.initial_cash * bench / bench.iloc[0]
        held_marks = {lot.symbol: marks[lot.symbol] for lot in ledger.lots.values()}
        pnl = pd.Series({"initial_cash": ledger.initial_cash, "final_cash": ledger.cash,
                         "open_positions_value": ledger.holdings(marks).sum(),
                         "final_nav": history.nav.iloc[-1], "realized_pnl": ledger.realized_pnl,
                         "unrealized_pnl": ledger.unrealized_pnl(marks), "total_fees": ledger.total_fees})
        return BacktestResult(
            history=history,
            fills=pd.DataFrame(ledger.fills, columns=Ledger.FILL_COLUMNS),
            open_lots=pd.DataFrame([asdict(lot) for lot in ledger.lots.values()], columns=list(Lot.__dataclass_fields__)),
            diagnostics=pd.DataFrame(diagnostics, columns=["date", "symbol", "lot_id", "event"]),
            marks=pd.Series(held_marks, dtype=float),
            last_quote=pd.Series({s: last_seen[s] for s in held_marks}, dtype="datetime64[ns]"),
            pnl=pnl)


def performance(nav, annual_rf=0.02):
    """Final value, CAGR (calendar days / 365.25), max drawdown, and daily Sharpe."""
    nav = nav.sort_index()
    returns = nav.pct_change().dropna()
    daily_rf = (1 + annual_rf) ** (1 / 252) - 1
    years = (nav.index[-1] - nav.index[0]).days / 365.25
    excess = returns - daily_rf
    return pd.Series({
        "final_value": nav.iloc[-1],
        "total_return": nav.iloc[-1] / nav.iloc[0] - 1,
        "CAGR": (nav.iloc[-1] / nav.iloc[0]) ** (1 / years) - 1,
        "max_drawdown": (nav / nav.cummax() - 1).min(),
        "annualized_volatility": returns.std(ddof=1) * np.sqrt(252),
        "Sharpe": np.sqrt(252) * excess.mean() / excess.std(ddof=1),
    })


def compare(result, annual_rf=0.02):
    h = result.history
    return pd.DataFrame({"strategy": performance(h.nav, annual_rf),
                         "S&P 500 TR": performance(h.benchmark_nav, annual_rf)})
