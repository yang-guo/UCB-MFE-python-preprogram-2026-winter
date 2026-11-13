"""Microbenchmark the duplicate-sale check, not total backtest runtime.

Historical completed orders are synthetic. One actual open lot is created via
the public OMS API. Each measured iteration submits and discards its pending
sale, keeping the book size fixed. No data or running notebook is accessed.
"""

import json
from pathlib import Path
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backtesting.oms import OMS, Order
from backtesting.indexed_sales import IndexedOMS


def measure(cls, historical_orders, repetitions=100):
    oms = cls(100, 0)
    lot = oms.submit_buy("s", "A", 100, "2020-01-01", "2020-01-02")
    oms.fill(lot, symbol="A", quote_date="2020-01-02", price=10, session="2020-01-02")
    for _ in range(historical_orders):
        i = len(oms.orders)
        oms.orders[i] = Order(i, "historical", "A", "BUY", oms.lots[lot].entry_date,
                              oms.lots[lot].entry_date, status="filled")
    start = perf_counter()
    for _ in range(repetitions):
        order = oms.submit_maturity_sale(lot, "2020-04-01")
        oms.discard(order)
    return (perf_counter() - start) / repetitions


if __name__ == "__main__":
    rows = []
    for size in (0, 10_000, 50_000):
        old = measure(OMS, size)
        new = measure(IndexedOMS, size)
        rows.append({"historical_orders": size, "original_seconds_per_check": old,
                     "indexed_seconds_per_check": new, "check_speedup": old / new})
    result = {"scope": "synthetic submit-and-discard microbenchmark; not end-to-end speedup",
              "repetitions": 100, "measurements": rows}
    path = Path(__file__).with_name("indexed_sales_results.json")
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
