"""Daily long-only backtester written for HW4.

Design (own implementation; the separation of strategy / ledger / simulation
follows the structure taught in Lecture 8, but the code is new):

* ``Ledger`` owns cash and open lots. Every purchase is an independent lot with
  its own entry price, fee and 90-calendar-day maturity.
* ``Backtester.run`` walks the market calendar one session at a time:
    1. the strategy plans orders using information through the previous close
       (the *signal date*) and the cash/NAV known at that close;
    2. today's quotes are read; mature lots with a quote are sold at the close;
    3. planned purchases are filled at today's close (rejected if no quote);
    4. the account is marked to market and recorded.
* Prices are adjusted closes, so a position's value already includes
  dividends; nothing is added on top. Fractional units are allowed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Lot:
    lot_id: int
    symbol: str
    units: float
    signal_date: pd.Timestamp
    entry_date: pd.Timestamp
    entry_price: float
    entry_fee: float
    due_date: pd.Timestamp


class Ledger:
    """Cash, open lots and the list of fills."""

    def __init__(self, initial_cash: float = 1_000_000.0, fee_bps: float = 5.0):
        if initial_cash <= 0 or fee_bps < 0:
            raise ValueError("initial_cash must be positive and fee_bps nonnegative")
        self.initial_cash = float(initial_cash)
        self.cash = float(initial_cash)
        self.fee_rate = fee_bps / 10_000
        self.lots: dict[int, Lot] = {}
        self.fills: list[dict] = []
        self._next_id = 0

    def buy(self, symbol, budget, price, *, signal_date, date, holding_days):
        """Spend ``budget`` dollars *including* the entry fee."""
        if not (np.isfinite(budget) and np.isfinite(price)) or budget <= 0 or price <= 0:
            raise ValueError("budget and price must be finite and positive")
        if not signal_date < date:
            raise ValueError("a fill must come strictly after its signal")
        if budget > self.cash + 1e-6:
            raise ValueError(f"insufficient cash: budget {budget:.2f} > cash {self.cash:.2f}")
        units = budget / (price * (1 + self.fee_rate))
        notional = units * price
        fee = notional * self.fee_rate
        lot = Lot(self._next_id, symbol, units, pd.Timestamp(signal_date), pd.Timestamp(date),
                  float(price), fee, pd.Timestamp(date) + pd.Timedelta(days=holding_days))
        self._next_id += 1
        self.cash = max(0.0, self.cash - notional - fee)
        self.lots[lot.lot_id] = lot
        self._record(lot, "BUY", date, price, notional, fee, -(notional + fee), 0.0)
        return lot

    def sell(self, lot_id, price, *, date):
        """Sell a whole lot at ``price`` and return its realized PnL."""
        lot = self.lots.pop(lot_id)
        if not np.isfinite(price) or price <= 0:
            raise ValueError("sale price must be finite and positive")
        if date < lot.due_date:
            raise ValueError("lots are only sold at or after maturity")
        notional = lot.units * price
        fee = notional * self.fee_rate
        pnl = notional - fee - lot.units * lot.entry_price - lot.entry_fee
        self.cash += notional - fee
        self._record(lot, "SELL", date, price, notional, fee, notional - fee, pnl)
        return pnl

    def _record(self, lot, side, date, price, notional, fee, cash_change, pnl):
        self.fills.append({
            "date": pd.Timestamp(date), "side": side, "lot_id": lot.lot_id, "symbol": lot.symbol,
            "signal_date": lot.signal_date, "entry_date": lot.entry_date, "due_date": lot.due_date,
            "price": price, "units": lot.units, "notional": notional, "fee": fee,
            "cash_change": cash_change, "realized_pnl": pnl,
        })

    def holdings_value(self, marks) -> pd.Series:
        """Market value by symbol using the supplied marks (last observed prices)."""
        values: dict[str, float] = {}
        for lot in self.lots.values():
            values[lot.symbol] = values.get(lot.symbol, 0.0) + lot.units * marks[lot.symbol]
        return pd.Series(values, dtype=float)


@dataclass
class BacktestResult:
    history: pd.DataFrame      # one row per session: cash, holdings, nav, ...
    fills: pd.DataFrame        # every purchase and sale
    open_lots: pd.DataFrame    # lots still open at the end (valued in final NAV)
    events: pd.DataFrame       # rejected orders, delayed exits, stale marks
    last_marks: pd.Series      # last observed price of every symbol ever held
    incidents: pd.DataFrame    # held-price jumps beyond the plausibility bounds


class Backtester:
    """Run a strategy on a wide table of adjusted closes.

    Parameters
    ----------
    quotes : DataFrame indexed by market session, one column per symbol,
        NaN where the source has no valid quote. Never forward-filled.
    strategy : object with ``orders(signal_date, exec_date, cash, nav, holdings)``
        returning a Series of fee-inclusive dollar budgets by symbol.
    """

    def __init__(self, quotes: pd.DataFrame, strategy, *, initial_cash=1_000_000.0,
                 fee_bps=5.0, holding_days=90, stale_warning_sessions=10, jump_up=1.0, jump_down=-0.5,
                 settle_after_sessions=None):
        if not quotes.index.is_monotonic_increasing or quotes.index.has_duplicates:
            raise ValueError("quotes must have a sorted, unique session index")
        values = quotes.to_numpy(dtype=float)
        if (values[np.isfinite(values)] <= 0).any():
            raise ValueError("quotes must be positive where present")
        self.quotes = quotes
        self.calendar = pd.DatetimeIndex(quotes.index)
        self.strategy = strategy
        self.ledger = Ledger(initial_cash, fee_bps)
        self.holding_days = holding_days
        self.stale_warning_sessions = stale_warning_sessions
        # Held-price moves beyond these bounds between two marks are logged as
        # incidents to investigate (accounting still uses the quoted price).
        self.jump_up, self.jump_down = jump_up, jump_down
        # Sensitivity only (off by default): a mature lot whose stock has not
        # quoted for this many sessions is closed at its last observed price.
        # This is an assumption about delisting proceeds, not an observed fill.
        self.settle_after_sessions = settle_after_sessions

    def run(self, start, end) -> BacktestResult:
        days = self.calendar[(self.calendar >= pd.Timestamp(start)) & (self.calendar <= pd.Timestamp(end))]
        first = self.calendar.get_loc(days[0])
        if first == 0:
            raise ValueError("need one session before the start to act as first signal date")
        ledger = self.ledger
        marks: dict[str, float] = {}
        last_seen: dict[str, int] = {}
        events: list[dict] = []
        incidents: list[dict] = []
        # Initial row: capital before any trade, dated at the first signal close.
        history = [{"date": self.calendar[first - 1], "cash": ledger.cash, "holdings": 0.0,
                    "nav": ledger.cash, "open_lots": 0, "fees": 0.0, "bought": 0.0, "sold": 0.0}]
        cash_at_close, nav_at_close = ledger.cash, ledger.cash
        holdings_at_close = pd.Series(dtype=float)

        for day in days:
            i = self.calendar.get_loc(day)
            signal_date = self.calendar[i - 1]
            # 1. Plan with information through the previous close only.
            orders = self.strategy.orders(signal_date, day, cash_at_close, nav_at_close, holdings_at_close)
            orders = orders[orders > 1e-6] if len(orders) else orders
            if len(orders) and (orders.index.duplicated().any() or orders.sum() > cash_at_close + 1e-6):
                raise ValueError(f"{day.date()}: orders must be unique and covered by signal-date cash")

            # 2. Read today's quotes (used for fills and marks only).
            today = self.quotes.iloc[i]
            observed = today[today.notna()]
            held_symbols = {lot.symbol for lot in ledger.lots.values()}
            for symbol, price in observed.items():
                if symbol in held_symbols:
                    move = float(price) / marks[symbol] - 1
                    if move > self.jump_up or move < self.jump_down:
                        units = sum(l.units for l in ledger.lots.values() if l.symbol == symbol)
                        incidents.append({"date": day, "symbol": symbol, "previous_mark": marks[symbol],
                                          "price": float(price), "move": move,
                                          "pnl_impact": units * (float(price) - marks[symbol])})
                if symbol in marks or symbol in orders.index:
                    marks[symbol], last_seen[symbol] = float(price), i
            n_fills = len(ledger.fills)

            # 3. Sell mature lots that have a quote today; otherwise wait.
            for lot in list(ledger.lots.values()):
                if day >= lot.due_date:
                    if lot.symbol in observed.index:
                        ledger.sell(lot.lot_id, float(observed[lot.symbol]), date=day)
                    elif (self.settle_after_sessions is not None
                          and i - last_seen[lot.symbol] >= self.settle_after_sessions):
                        ledger.sell(lot.lot_id, marks[lot.symbol], date=day)
                        events.append({"date": day, "symbol": lot.symbol, "lot_id": lot.lot_id,
                                       "event": "settled_at_last_quote_ASSUMPTION"})
                    else:
                        events.append({"date": day, "symbol": lot.symbol, "lot_id": lot.lot_id,
                                       "event": "exit_delayed_no_quote"})

            # 4. Fill planned purchases at today's close.
            for symbol, budget in orders.items():
                if symbol not in observed.index:
                    events.append({"date": day, "symbol": symbol, "lot_id": np.nan,
                                   "event": "buy_rejected_no_quote"})
                    continue
                ledger.buy(symbol, float(budget), float(observed[symbol]), signal_date=signal_date,
                           date=day, holding_days=self.holding_days)
                marks[symbol], last_seen[symbol] = float(observed[symbol]), i

            # 5. Mark to market with the last observed price of each holding.
            for symbol in {lot.symbol for lot in ledger.lots.values()}:
                gap = i - last_seen[symbol]
                if gap >= self.stale_warning_sessions and gap % self.stale_warning_sessions == 0:
                    events.append({"date": day, "symbol": symbol, "lot_id": np.nan,
                                   "event": f"stale_mark_{gap}_sessions"})
            holdings_at_close = ledger.holdings_value(marks)
            cash_at_close = ledger.cash
            nav_at_close = cash_at_close + holdings_at_close.sum()
            new = ledger.fills[n_fills:]
            history.append({
                "date": day, "cash": cash_at_close, "holdings": holdings_at_close.sum(),
                "nav": nav_at_close, "open_lots": len(ledger.lots),
                "fees": sum(f["fee"] for f in new),
                "bought": sum(f["notional"] for f in new if f["side"] == "BUY"),
                "sold": sum(f["notional"] for f in new if f["side"] == "SELL"),
            })

        held = sorted({f["symbol"] for f in ledger.fills})
        return BacktestResult(
            history=pd.DataFrame(history).set_index("date"),
            fills=pd.DataFrame(ledger.fills),
            open_lots=pd.DataFrame([asdict(lot) for lot in ledger.lots.values()],
                                   columns=list(Lot.__dataclass_fields__)),
            events=pd.DataFrame(events, columns=["date", "symbol", "lot_id", "event"]),
            last_marks=pd.Series({s: marks[s] for s in held}, dtype=float),
            incidents=pd.DataFrame(incidents, columns=["date", "symbol", "previous_mark", "price",
                                                       "move", "pnl_impact"]),
        )


def performance(nav: pd.Series, annual_rf: float = 0.02) -> pd.Series:
    """Final value, CAGR, max drawdown and Sharpe from a daily NAV series."""
    nav = nav.sort_index()
    daily = nav.pct_change().dropna()
    rf_daily = (1 + annual_rf) ** (1 / 252) - 1
    excess = daily - rf_daily
    years = (nav.index[-1] - nav.index[0]).days / 365.25
    return pd.Series({
        "final_value": nav.iloc[-1],
        "total_return": nav.iloc[-1] / nav.iloc[0] - 1,
        "CAGR": (nav.iloc[-1] / nav.iloc[0]) ** (1 / years) - 1,
        "max_drawdown": (nav / nav.cummax() - 1).min(),
        "annual_volatility": daily.std(ddof=1) * np.sqrt(252),
        "Sharpe": np.sqrt(252) * excess.mean() / daily.std(ddof=1),
    })


def accounting_checks(result: BacktestResult, quotes: pd.DataFrame, initial_cash: float,
                      holding_days: int = 90) -> pd.Series:
    """Independent checks of the books; every value should be True or ~0."""
    h, f = result.history, result.fills
    out = {}
    # Cash rebuilt from fills only.
    flows = f.groupby("date").cash_change.sum().reindex(h.index, fill_value=0.0)
    out["max_cash_reconstruction_error"] = float((initial_cash + flows.cumsum() - h.cash).abs().max())
    out["cash_never_negative"] = bool((h.cash >= -1e-6).all())
    # NAV = cash + holdings, and holdings rebuilt from open lots on the last day.
    out["max_nav_identity_error"] = float((h.cash + h.holdings - h.nav).abs().max())
    final_holdings = sum(r.units * result.last_marks[r.symbol] for r in result.open_lots.itertuples())
    out["final_holdings_rebuild_error"] = abs(final_holdings - h.holdings.iloc[-1])
    buys, sells = f[f.side == "BUY"], f[f.side == "SELL"]
    out["every_buy_after_its_signal"] = bool((buys.date > buys.signal_date).all())
    held_days = (sells.date - sells.entry_date).dt.days
    out["every_sale_held_at_least_90_days"] = bool((held_days >= holding_days).all())
    out["max_days_held"] = int(held_days.max()) if len(held_days) else 0
    # Fills use the quoted adjusted close of the fill date.
    q = quotes.stack()
    fill_quotes = q.reindex(pd.MultiIndex.from_arrays([f.date, f.symbol])).to_numpy()
    quoted = np.isfinite(fill_quotes)
    out["fills_at_that_day_close"] = bool(np.allclose(fill_quotes[quoted], f.price.to_numpy()[quoted]))
    out["fills_without_same_day_quote"] = int((~quoted).sum())   # only the settlement sensitivity
    # Each lot bought once and sold at most once.
    out["lots_sold_at_most_once"] = bool(not sells.lot_id.duplicated().any())
    out["sold_plus_open_equals_bought"] = len(sells) + len(result.open_lots) == len(buys)
    # PnL identity: final NAV - initial = realized + unrealized.
    unrealized = sum(r.units * (result.last_marks[r.symbol] - r.entry_price) - r.entry_fee
                     for r in result.open_lots.itertuples())
    out["pnl_identity_error"] = abs(h.nav.iloc[-1] - initial_cash - sells.realized_pnl.sum() - unrealized)
    return pd.Series(out)
