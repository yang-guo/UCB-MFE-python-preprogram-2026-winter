"""OMS indexed pending-sale lookup used by the final backtesting notebook.

The original engine and coordinator are reused unchanged. Only the duplicate
pending-sale check uses a lot-ID lookup instead of scanning historical orders.
Imports do not patch any existing module or account. Strategies, refits, clock,
execution prices, fees, maturity, and the complete audit ledger are unchanged.
"""

import json
import os
from pathlib import Path
import time

import pandas as pd

from .engine import Backtester as OriginalBacktester
from .oms import OMS, Order
from .portfolio import ShadowPortfolioBacktester as OriginalCoordinator


class IndexedOMS(OMS):
    """Keep pending sales indexed by lot; retain all orders in the ledger."""

    def __init__(self, initial_cash=1_000_000.0, fee_bps=5.0):
        super().__init__(initial_cash, fee_bps)
        self._pending_sales = {}

    def submit_maturity_sale(self, lot_id, execution_date):
        if lot_id not in self.lots:
            raise ValueError("Unknown lot")
        lot = self.lots[lot_id]
        execution_date = pd.Timestamp(execution_date)
        if pd.isna(execution_date) or execution_date < lot.due_date:
            raise ValueError("Cannot sell before 90-calendar-day maturity")
        if lot_id in self._pending_sales:
            raise ValueError("A pending sale already exists for this lot")
        order = Order(len(self.orders), lot.strategy_id, lot.symbol, "SELL",
                      lot.entry_date, execution_date, lot_id=lot_id, trigger="maturity",
                      model_version=lot.model_version)
        self.orders[order.order_id] = order
        self._pending_sales[lot_id] = order.order_id
        return order.order_id

    def _forget_sale(self, order):
        if (order.side == "SELL"
                and self._pending_sales.get(order.lot_id) == order.order_id):
            del self._pending_sales[order.lot_id]

    def fill(self, order_id, **kwargs):
        # A failed fill must retain both its pending status and its index entry.
        super().fill(order_id, **kwargs)
        self._forget_sale(self.orders[order_id])

    def reject(self, order_id):
        super().reject(order_id)
        self._forget_sale(self.orders[order_id])

    def discard(self, order_id):
        order = super().discard(order_id)
        self._forget_sale(order)
        return order


def _install(engine):
    """Replace only a fresh account, never an account from a running test."""
    old = engine.oms
    if old.orders or old.lots or old.fills or engine._started:
        raise ValueError("Indexed OMS installation requires a fresh account")
    replacement = IndexedOMS(old.initial_cash, fee_bps=0)
    # Preserve the exact already-validated rate; avoid a basis-point round trip.
    replacement.fee_rate = old.fee_rate
    engine.oms = replacement


class Backtester(OriginalBacktester):
    """Original single-account engine with a fresh indexed OMS."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _install(self)


class ShadowPortfolioBacktester(OriginalCoordinator):
    """Original coordinator; replace the fresh main and shadow OMS instances."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for engine in [self._main, *self._shadows.values()]:
            _install(engine)

    def run(self, start, end, *, shadow_start=None, allow_partial=False,
            progress_every=63, progress_path=None):
        """Report completed main-account sessions without changing the clock.

        Percentages measure sessions, not CPU time. Logging is outside the
        execution transaction; failed sessions never count as completed.
        """
        if (isinstance(progress_every, bool) or not isinstance(progress_every, int)
                or progress_every < 1):
            raise ValueError("progress_every must be a positive integer")
        target = progress_path or os.environ.get("HW4_INDEXED_PROGRESS_PATH")
        target = Path(target) if target else None
        started = time.monotonic()
        advance = self._main.advance

        def advance_with_progress():
            advance()
            completed, total = self._main._cursor, len(self._main._days)
            if completed != 1 and completed % progress_every and completed != total:
                return
            state = dict(simulated_date=str(self._main._days[completed - 1].date()),
                         completed_sessions=completed, total_sessions=total,
                         percent_sessions=round(100 * completed / total, 2),
                         elapsed_seconds=round(time.monotonic() - started, 1))
            print("Backtest progress: " + json.dumps(state), flush=True)
            if target:
                try:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    pending = target.with_suffix(".tmp")
                    pending.write_text(json.dumps(state, indent=2) + "\n")
                    pending.replace(target)
                except OSError as error:
                    # A telemetry write failure must not invalidate a booked
                    # session or change the simulation's transaction behavior.
                    print(f"Progress file unavailable: {error}", flush=True)

        self._main.advance = advance_with_progress
        try:
            return super().run(start, end, shadow_start=shadow_start,
                               allow_partial=allow_partial)
        finally:
            self._main.advance = advance
