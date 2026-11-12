"""A small long-only, fixed-holding-period backtester (own implementation).

Timeline for one market session ``t`` (everything happens at the close):

1. BUY   orders that the strategy planned after the close of ``t-1`` are
         filled at the close of ``t``. Their dollar budgets were capped by the
         cash that existed at ``t-1``, so the same cash is never spent twice.
2. SELL  every lot whose 90-calendar-day maturity is on or before ``t`` and
         that has a quote today. Proceeds are added to cash.
3. MARK  the portfolio: cash + units * last observed adjusted close.
4. PLAN  the strategy sees a snapshot (cash, value, holdings) and returns the
         orders to fill at the close of ``t+1``. It never sees ``t+1`` prices.

Positions are fractional units of the *adjusted* close, so dividends are
already inside the price series and are never added again.

Missing prices are never replaced by invented fills:
* no quote on the entry session  -> the order is rejected and cash is kept;
* no quote on the maturity session -> the lot stays open, is marked at its
  last observed price (flagged as stale) and is sold at the next real quote.

``assume_cash_out_after`` exists only for a sensitivity analysis: if set to N,
a matured lot that has had no quote for N sessions is converted to cash at
its last observed price (as if the company had been bought for cash). This IS
an assumed fill, so it is off by default and never used for the main result.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class Lot:
    """One purchase. It is sold in full at maturity."""

    lot_id: int
    symbol: str
    units: float
    entry_date: pd.Timestamp
    entry_price: float
    cost: float  # cash paid, including the purchase fee
    due_date: pd.Timestamp  # entry_date + holding period (a calendar date)
    signal_date: pd.Timestamp  # close at which the order was decided
    last_price: float  # last observed adjusted close (the mark)
    last_quote_date: pd.Timestamp
    stale_sessions: int = 0  # consecutive sessions without a quote


@dataclass(frozen=True)
class Snapshot:
    """What the strategy is allowed to know after the close of ``date``."""

    date: pd.Timestamp
    cash: float
    total_value: float
    positions: dict  # symbol -> market value of all open lots


@dataclass
class BacktestResult:
    daily: pd.DataFrame  # one row per session: cash, positions, total value
    trades: pd.DataFrame  # one row per fill (BUY or SELL)
    open_lots: pd.DataFrame  # lots still open at the end, marked at last price
    rejected: pd.DataFrame  # orders that could not be filled (no quote)
    initial_cash: float
    cost_rate: float
    holding_days: int
    audit: dict = field(default_factory=dict)


class Backtester:
    """Run a strategy over ``[start, end]`` on a dates x symbols price table.

    Parameters
    ----------
    prices : DataFrame
        Adjusted closes indexed by market session, one column per symbol.
        NaN means "no quote that day". Must contain the session before
        ``start`` (the first orders are planned at that close).
    start, end : date-like
        First and last sessions on which fills may happen.
    cost_rate : float
        Proportional transaction cost charged on every buy and every sell.
    """

    def __init__(self, prices, start, end, initial_cash=1_000_000.0,
                 cost_rate=0.001, holding_days=90, assume_cash_out_after=None):
        if not prices.index.is_monotonic_increasing or prices.index.has_duplicates:
            raise ValueError("prices must be indexed by unique, sorted sessions")
        self.prices = prices
        self.sessions = prices.index
        self.start = self.sessions[self.sessions.searchsorted(pd.Timestamp(start))]
        end = pd.Timestamp(end)
        self.end = self.sessions[self.sessions.searchsorted(end, side="right") - 1]
        if self.sessions.get_loc(self.start) == 0:
            raise ValueError("prices must include the session before `start`")
        self.initial_cash = float(initial_cash)
        self.cost_rate = float(cost_rate)
        self.holding_days = int(holding_days)
        self.assume_cash_out_after = assume_cash_out_after

    # ------------------------------------------------------------------ run
    def run(self, strategy):
        values = self.prices.to_numpy(dtype=float)
        column_of = {symbol: j for j, symbol in enumerate(self.prices.columns)}
        first = self.sessions.get_loc(self.start)
        last = self.sessions.get_loc(self.end)

        cash = self.initial_cash
        lots: list[Lot] = []
        pending: dict = {}  # symbol -> budget, planned at the previous close
        pending_signal_date = None
        trades, daily, rejected = [], [], []
        next_lot_id = 0

        # The loop starts one session early: nothing can fill there, but the
        # strategy plans its first orders and the history records the
        # untouched starting capital (so entry fees show up in returns).
        for i in range(first - 1, last + 1):
            date = self.sessions[i]

            # 1. BUY orders planned at the previous close.
            for symbol, budget in pending.items():
                price = values[i, column_of[symbol]] if symbol in column_of else np.nan
                if not np.isfinite(price) or price <= 0:
                    rejected.append({"date": date, "symbol": symbol, "budget": budget,
                                     "reason": "no quote on entry session"})
                    continue
                units = budget / (price * (1 + self.cost_rate))
                fee = units * price * self.cost_rate
                cash -= budget
                lots.append(Lot(next_lot_id, symbol, units, date, price, budget,
                                date + pd.Timedelta(days=self.holding_days),
                                pending_signal_date, price, date))
                trades.append({"date": date, "side": "BUY", "symbol": symbol,
                               "lot_id": next_lot_id, "units": units, "price": price,
                               "gross": units * price, "fee": fee, "cash_flow": -budget,
                               "signal_date": pending_signal_date,
                               "due_date": lots[-1].due_date, "assumed_fill": False})
                next_lot_id += 1
            if cash < -1e-6:
                raise RuntimeError(f"Negative cash {cash:.2f} on {date.date()}")
            pending = {}

            # 2. SELL matured lots that have a quote today; update marks.
            still_open = []
            for lot in lots:
                price = values[i, column_of[lot.symbol]]
                quoted = np.isfinite(price) and price > 0
                if quoted:
                    lot.last_price, lot.last_quote_date, lot.stale_sessions = price, date, 0
                else:
                    lot.stale_sessions += 1
                assumed = (not quoted and self.assume_cash_out_after is not None
                           and lot.stale_sessions >= self.assume_cash_out_after)
                if date >= lot.due_date and (quoted or assumed):
                    gross = lot.units * lot.last_price
                    fee = gross * self.cost_rate
                    cash += gross - fee
                    trades.append({"date": date, "side": "SELL", "symbol": lot.symbol,
                                   "lot_id": lot.lot_id, "units": lot.units,
                                   "price": lot.last_price, "gross": gross, "fee": fee,
                                   "cash_flow": gross - fee, "signal_date": lot.signal_date,
                                   "due_date": lot.due_date, "assumed_fill": assumed})
                    continue
                still_open.append(lot)
            lots = still_open

            # 3. MARK at the last observed price (stale marks are reported).
            position_value = sum(lot.units * lot.last_price for lot in lots)
            stale_value = sum(lot.units * lot.last_price for lot in lots
                              if lot.last_quote_date < date)
            overdue_value = sum(lot.units * lot.last_price for lot in lots
                                if lot.due_date <= date)
            total = cash + position_value
            daily.append({"date": date, "cash": cash, "positions": position_value,
                          "total_value": total, "open_lots": len(lots),
                          "stale_value": stale_value, "overdue_value": overdue_value})

            # 4. PLAN orders for the next close (never on the final session).
            if i < last:
                positions = {}
                for lot in lots:
                    positions[lot.symbol] = positions.get(lot.symbol, 0.0) + lot.units * lot.last_price
                orders = strategy.plan(Snapshot(date, cash, total, positions)) or {}
                orders = {s: float(b) for s, b in orders.items() if b > 0}
                if sum(orders.values()) > cash + 1e-6:
                    raise ValueError(f"Orders on {date.date()} exceed available cash")
                pending, pending_signal_date = orders, date

        trades = pd.DataFrame(trades, columns=["date", "side", "symbol", "lot_id", "units", "price",
                                               "gross", "fee", "cash_flow", "signal_date", "due_date",
                                               "assumed_fill"])
        for column in ["date", "signal_date", "due_date"]:
            trades[column] = pd.to_datetime(trades[column])
        result = BacktestResult(
            daily=pd.DataFrame(daily).set_index("date"), trades=trades,
            open_lots=pd.DataFrame([vars(lot) for lot in lots]),
            rejected=pd.DataFrame(rejected, columns=["date", "symbol", "budget", "reason"]),
            initial_cash=self.initial_cash, cost_rate=self.cost_rate,
            holding_days=self.holding_days)
        result.audit = audit(result, self.prices)
        return result


def audit(result, prices):
    """Re-derive the accounting from the trade list, independently of the loop.

    Returns a dict of named checks. Boolean entries must all be True.
    """
    daily, trades = result.daily, result.trades
    buys, sells = trades[trades.side == "BUY"], trades[trades.side == "SELL"]
    checks = {}

    # Cash: never negative, and equal to initial + all signed cash flows.
    checks["cash_never_negative"] = bool((daily.cash >= -1e-6).all())
    flows = trades.groupby("date").cash_flow.sum().reindex(daily.index, fill_value=0.0)
    rebuilt_cash = result.initial_cash + flows.cumsum()
    checks["cash_reconciles"] = bool(np.allclose(rebuilt_cash, daily.cash, atol=1e-4))

    # Holdings: rebuild units per symbol from fills and re-price them with the
    # last observed close (forward fill = causal mark, never a future price).
    signed = trades.assign(signed=np.where(trades.side == "BUY", trades.units, -trades.units))
    held = (signed.pivot_table(index="date", columns="symbol", values="signed", aggfunc="sum")
            .reindex(daily.index).fillna(0.0).cumsum())
    held = held.where(held.abs() > 1e-9, 0.0)
    marks = prices.reindex(columns=held.columns).ffill().reindex(daily.index)
    rebuilt_value = (held * marks).sum(axis=1) + rebuilt_cash
    checks["value_reconciles"] = bool(np.allclose(rebuilt_value, daily.total_value,
                                                  rtol=1e-9, atol=1e-3))
    checks["no_short_positions"] = bool((held >= 0).all().all())

    # Units: everything bought is either sold or still open.
    open_units = result.open_lots.units.sum() if len(result.open_lots) else 0.0
    checks["units_conserved"] = bool(np.isclose(buys.units.sum(), sells.units.sum() + open_units))

    # Timing: each buy fills on the session right after its signal, and each
    # sale happens on the first session on/after maturity that has a quote.
    sessions = prices.index
    signal_pos = sessions.get_indexer(buys.signal_date)
    checks["buys_fill_next_session"] = bool((sessions[signal_pos + 1] == buys.date.values).all())
    entry = buys.set_index("lot_id").date
    held_days = (sells.date.to_numpy() - entry.reindex(sells.lot_id).to_numpy()) / np.timedelta64(1, "D")
    checks["all_holds_at_least_90d"] = bool((held_days >= result.holding_days).all())
    real_sells = sells[~sells.assumed_fill.astype(bool)]
    on_time = 0
    for sale in real_sells.itertuples():
        quotes = prices[sale.symbol].loc[sale.due_date:sale.date].dropna()
        on_time += int(len(quotes) > 0 and quotes.index[0] == sale.date
                       and quotes.iloc[0] == sale.price)
    checks["sales_on_first_quote_after_maturity"] = bool(on_time == len(real_sells))
    checks["n_assumed_cash_outs"] = int(len(sells) - len(real_sells))
    first_session_after_due = sessions[sessions.searchsorted(sells.due_date.values)]
    checks["n_sales_delayed_by_missing_quote"] = int((sells.date.values > first_session_after_due).sum())
    checks["max_hold_days"] = float(held_days.max()) if len(held_days) else 0.0

    checks["n_buys"], checks["n_sells"] = len(buys), len(sells)
    checks["n_open_lots_at_end"] = len(result.open_lots)
    checks["n_rejected_orders"] = len(result.rejected)
    checks["total_fees"] = float(trades.fee.sum())
    checks["max_stale_value_share"] = float((daily.stale_value / daily.total_value).max())
    checks["final_overdue_value"] = float(daily.overdue_value.iloc[-1])
    return checks
