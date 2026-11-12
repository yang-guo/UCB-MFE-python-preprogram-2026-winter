"""Pending orders, cash reservations, lots, and simulated fill accounting."""

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .contracts import Portfolio, plain_amount


@dataclass
class Order:
    order_id: int
    strategy_id: str
    symbol: str
    side: str
    signal_date: pd.Timestamp
    execution_date: pd.Timestamp
    budget: float = 0.0
    lot_id: int | None = None
    trigger: str = "strategy"
    status: str = "pending"
    model_version: int = 0


@dataclass(frozen=True)
class Lot:
    lot_id: int
    strategy_id: str
    symbol: str
    units: float
    entry_date: pd.Timestamp
    entry_price: float
    entry_fee: float
    due_date: pd.Timestamp
    model_version: int = 0


class OMS:
    """Reservations reduce available cash; only fills change actual cash."""

    def __init__(self, initial_cash=1_000_000.0, fee_bps=5.0):
        try:
            initial_cash, fee_bps = plain_amount(initial_cash), plain_amount(fee_bps)
        except ValueError:
            raise ValueError("Capital must be positive; fees must be between 0 and 10000 bps") from None
        if initial_cash <= 0 or not 0 <= fee_bps <= 10_000:
            raise ValueError("Capital must be positive; fees must be between 0 and 10000 bps")
        self.initial_cash = self.cash = initial_cash
        self.fee_rate = fee_bps / 10_000
        self.orders: dict[int, Order] = {}
        self.lots: dict[int, Lot] = {}
        self.fills: list[dict] = []

    @property
    def available_cash(self):
        reserved = sum(o.budget for o in self.orders.values()
                       if o.status == "pending" and o.side == "BUY")
        available = self.cash - reserved
        # Reservation sums can round a few ulps above cash. Treat that dust as
        # zero; a larger deficit is a real accounting error.
        if available < 0 and -available <= 1e-12 * max(1.0, self.initial_cash):
            return 0.0
        return available

    def submit_buy(self, strategy_id, symbol, budget, signal_date, execution_date, *, model_version=0):
        signal_date, execution_date = pd.Timestamp(signal_date), pd.Timestamp(execution_date)
        if not strategy_id or type(symbol) is not str or not symbol:
            raise ValueError("Order needs a strategy and symbol")
        if pd.isna(signal_date) or pd.isna(execution_date) or signal_date >= execution_date:
            raise ValueError("Purchase must follow its signal")
        try:
            budget = plain_amount(budget)
        except ValueError:
            raise ValueError("Purchase budget must fit unreserved cash") from None
        # Scale with opening capital: subtraction error accumulates over the
        # whole book, not just the cash left for the last order.
        tolerance = 1e-12 * max(1.0, self.initial_cash)
        if budget <= 0 or budget > self.available_cash + tolerance:
            raise ValueError("Purchase budget must fit unreserved cash")
        # Tolerate representable rounding, but never reserve more than is
        # actually available; the order itself must fit unreserved cash.
        budget = min(budget, self.available_cash)
        if budget <= 0:
            raise ValueError("Purchase budget must fit unreserved cash")
        if not isinstance(model_version, int) or model_version < 0:
            raise ValueError("Model version must be a nonnegative integer")
        order = Order(len(self.orders), strategy_id, symbol, "BUY", signal_date,
                      execution_date, float(budget), model_version=model_version)
        self.orders[order.order_id] = order
        return order.order_id

    def submit_maturity_sale(self, lot_id, execution_date):
        if lot_id not in self.lots:
            raise ValueError("Unknown lot")
        lot = self.lots[lot_id]
        execution_date = pd.Timestamp(execution_date)
        if pd.isna(execution_date) or execution_date < lot.due_date:
            raise ValueError("Cannot sell before 90-calendar-day maturity")
        if any(o.lot_id == lot_id and o.side == "SELL" and o.status == "pending"
               for o in self.orders.values()):
            raise ValueError("A pending sale already exists for this lot")
        # The exit was fixed at entry, rather than chosen using today's price.
        order = Order(len(self.orders), lot.strategy_id, lot.symbol, "SELL",
                      lot.entry_date, execution_date, lot_id=lot_id, trigger="maturity",
                      model_version=lot.model_version)
        self.orders[order.order_id] = order
        return order.order_id

    def forced_sale(self, lot_id, *, date, price):
        """Book an early exit at the last observed mark.

        Used only when a held security stops quoting. The normal 90-day
        maturity check is deliberately bypassed, and the trigger is fixed to
        ``delisting`` so the fill cannot masquerade as a maturity sale.
        """
        if lot_id not in self.lots:
            raise ValueError("Unknown lot")
        lot = self.lots[lot_id]
        date = pd.Timestamp(date)
        if pd.isna(date) or date < lot.entry_date:
            raise ValueError("Forced sale cannot precede entry")
        if any(o.lot_id == lot_id and o.side == "SELL" and o.status == "pending"
               for o in self.orders.values()):
            raise ValueError("A pending sale already exists for this lot")
        order = Order(len(self.orders), lot.strategy_id, lot.symbol, "SELL",
                      lot.entry_date, date, lot_id=lot_id,
                      trigger="delisting", model_version=lot.model_version)
        self.orders[order.order_id] = order
        try:
            self.fill(order.order_id, symbol=lot.symbol, quote_date=order.execution_date,
                      price=price, session=order.execution_date, allow_early=True)
        except BaseException:
            self.orders.pop(order.order_id, None)
            raise
        return order.order_id

    def reject(self, order_id):
        if order_id not in self.orders:
            raise ValueError("Unknown order")
        order = self.orders[order_id]
        if order.status != "pending":
            raise ValueError("Only pending orders can be rejected")
        order.status = "rejected_missing_quote"

    def fill(self, order_id, *, symbol, quote_date, price, session, allow_early=False):
        if order_id not in self.orders:
            raise ValueError("Unknown order")
        order = self.orders[order_id]
        if order.status != "pending":
            raise ValueError("Order is already filled or rejected")
        if symbol != order.symbol or pd.Timestamp(quote_date) != order.execution_date:
            raise ValueError("Quote must match the order symbol and execution date")
        if pd.Timestamp(session) != order.execution_date:
            raise ValueError("Cannot book a quote outside the current session")
        if not np.isfinite(price) or price <= 0:
            raise ValueError("Fill price must be finite and positive")
        if order.side == "BUY":
            # Sequential fills can leave cash a few ulps below a later reserved
            # budget. Use a capital-relative tolerance, then spend only actual
            # cash so a rounding difference cannot create credit.
            tolerance = 1e-12 * max(1.0, self.initial_cash)
            if order.budget > self.cash + tolerance:
                raise ValueError("Insufficient cash at execution")
            budget = min(order.budget, self.cash)
            if budget <= 0:
                raise ValueError("Insufficient cash at execution")
            with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
                # Divide budget first to avoid overflowing price * (1 + fee).
                units = (budget / (1 + self.fee_rate)) / price
                if not np.isfinite(units) or units <= 0:
                    raise ValueError("Fill arithmetic must produce finite, representable amounts")
                # Floating-point division can round the quantity up. Move down
                # by representable units until notional plus fee fits the actual
                # budget; never solve rounding by allowing extra borrowing.
                for _ in range(8):
                    notional = units * price
                    fee = notional * self.fee_rate
                    cost = notional + fee
                    if np.isfinite(cost) and cost <= budget:
                        break
                    units = np.nextafter(units, 0.0)
                else:
                    raise ValueError("Cannot represent a purchase within its budget")
                change = -cost
            lot_id = order.order_id
            lot = Lot(lot_id, order.strategy_id, symbol, units, order.execution_date,
                      price, fee, order.execution_date + pd.Timedelta(days=90), order.model_version)
            realized = 0.0
        else:
            if order.lot_id not in self.lots:
                raise ValueError("Unknown lot")
            lot = self.lots[order.lot_id]
            if not allow_early and order.execution_date < lot.due_date:
                raise ValueError("Premature sale")
            units, lot_id = lot.units, lot.lot_id
            with np.errstate(over="ignore", invalid="ignore", under="ignore"):
                notional = units * price
                fee = notional * self.fee_rate
                change = notional - fee
                realized = change - units * lot.entry_price - lot.entry_fee
        next_cash = self.cash + change
        if (not np.isfinite([units, notional, fee, change, realized, next_cash]).all()
                or units <= 0 or notional <= 0 or fee < 0):
            raise ValueError("Fill arithmetic must produce finite, representable amounts")
        if next_cash < 0:
            raise ValueError("Fill would require borrowing")
        # Commit only after all computations validate: rejected fills leave the
        # order, reservations, lots, cash, and ledger unchanged.
        if order.side == "BUY":
            self.lots[lot_id] = lot
        else:
            del self.lots[lot_id]
        self.cash = 0.0 if next_cash == 0 else next_cash
        order.status = "filled"
        self.fills.append(dict(order_id=order_id, lot_id=lot_id,
                               strategy_id=order.strategy_id, trigger=order.trigger, model_version=order.model_version,
                               symbol=symbol, side=order.side, signal_date=order.signal_date,
                               date=order.execution_date, due_date=lot.due_date,
                               price=price, units=units, notional=notional, fee=fee,
                               cash_change=change, realized_pnl=realized))

    def snapshot(self, marks):
        holdings = {}
        for lot in self.lots.values():
            price = marks[lot.symbol]
            if not np.isfinite(price) or price <= 0:
                raise ValueError("Invalid valuation mark")
            with np.errstate(over="ignore", invalid="ignore"):
                value = holdings.get(lot.symbol, 0.0) + lot.units * price
            if not np.isfinite(value):
                raise ValueError("Holding valuation is not finite")
            holdings[lot.symbol] = value
        nav = self.cash + sum(holdings.values())
        if not np.isfinite([nav, self.available_cash]).all():
            raise ValueError("Portfolio valuation is not finite")
        return Portfolio(self.cash, self.available_cash, nav, holdings)

    def order_records(self):
        return [asdict(o) for o in self.orders.values()]
