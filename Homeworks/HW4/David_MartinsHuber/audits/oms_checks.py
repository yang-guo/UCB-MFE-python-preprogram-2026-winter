"""Independent OMS round-two audit. Does not import production test fixtures."""
from pathlib import Path
import sys
import random
from copy import deepcopy
from decimal import Decimal, getcontext
import numpy as np

base = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(base))
from backtesting import OMS
getcontext().prec = 45

def state(o):
    return deepcopy((o.cash, o.available_cash, o.orders, o.lots, o.fills))

def expect_atomic_rejection(o, callback):
    before = state(o)
    try:
        callback()
    except ValueError:
        assert state(o) == before
    else:
        raise AssertionError('Invalid input was not rejected')

rng = random.Random(78312)
for trial in range(200):
    cap = rng.uniform(1000, 2_000_000)
    bps = rng.uniform(0, 100)
    o = OMS(cap, bps)
    oracle_cash = Decimal(str(cap))
    fee = Decimal(str(bps)) / Decimal(10000)
    pending = []
    for k in range(6):
        budget = o.available_cash * rng.uniform(.03, .18)
        order = o.submit_buy('oracle', f'S{k}', budget, '2020-01-02', '2020-01-03', model_version=4)
        pending.append((order, budget))
    rng.shuffle(pending)
    owned = []
    for order, budget in pending:
        if rng.random() < .2:
            o.reject(order)
            continue
        price = rng.uniform(.1, 1000)
        exact_units = Decimal(str(budget)) / (Decimal(str(price)) * (1 + fee))
        o.fill(order, symbol=o.orders[order].symbol, quote_date='2020-01-03', price=price, session='2020-01-03')
        assert abs(o.lots[order].units / float(exact_units) - 1) < 1e-12
        oracle_cash -= Decimal(str(budget))
        assert abs(o.cash - float(oracle_cash)) <= max(1e-8, 32 * np.spacing(abs(float(oracle_cash))))
        owned.append((order, exact_units))
    assert abs(o.available_cash - o.cash) <= 1e-8
    for order, exact_units in owned:
        symbol = o.lots[order].symbol
        price = rng.uniform(.1, 1000)
        sale = o.submit_maturity_sale(order, '2020-04-03')
        o.fill(sale, symbol=symbol, quote_date='2020-04-03', price=price, session='2020-04-03')
        oracle_cash += exact_units * Decimal(str(price)) * (1 - fee)
        assert abs(o.cash - float(oracle_cash)) <= max(1e-8, 32 * np.spacing(abs(float(oracle_cash))))
        assert o.fills[-1]['model_version'] == 4
    assert not o.lots
print('PASS: 200 independent Decimal-ledger paths with reservations/rejections/fees/maturity sales')

for bps in [-1, 10000.1, 20000, np.inf, np.nan]:
    try: OMS(100, bps)
    except ValueError: pass
    else: raise AssertionError(f'Unsupported fee {bps} accepted')
for bps, expected in [(0, 100), (10000, 0)]:
    o = OMS(100, bps)
    i = o.submit_buy('s', 'A', 100, '2020-01-02', '2020-01-03')
    o.fill(i, symbol='A', quote_date='2020-01-03', price=10, session='2020-01-03')
    j = o.submit_maturity_sale(i, '2020-04-03')
    o.fill(j, symbol='A', quote_date='2020-04-03', price=10, session='2020-04-03')
    assert abs(o.cash - expected) < 1e-8
    assert abs(o.cash - (100 + sum(f['cash_change'] for f in o.fills))) < 1e-8
print('PASS: accepted/rejected fee bounds; 0% and 100% round trips reconcile')

o = OMS(100, 5)
i = o.submit_buy('s', 'A', 100, '2020-01-02', '2020-01-03')
for price in [float(np.nextafter(0, 1)), np.nextafter(0, 1), np.inf, np.nan, 0, -1]:
    expect_atomic_rejection(o, lambda p=price: o.fill(i, symbol='A', quote_date='2020-01-03', price=p, session='2020-01-03'))
o.fill(i, symbol='A', quote_date='2020-01-03', price=.01, session='2020-01-03')
j = o.submit_maturity_sale(i, '2020-04-03')
expect_atomic_rejection(o, lambda: o.fill(j, symbol='A', quote_date='2020-04-03', price=1e308, session='2020-04-03'))
expect_atomic_rejection(o, lambda: o.snapshot({'A': 1e308}))
print('PASS: six malformed/unrepresentable buys, overflow sale and valuation reject atomically')

# Positive finite prices and $100m capital are ordinary configuration inputs.
# Expected fee-inclusive sizing must fit its accepted budget without fabricated cash.
failures = []
for capital, price in [(100_000_000, 958.2097977234747), (1_000_000_000, 937.9895251897534)]:
    o = OMS(capital, 5)
    i = o.submit_buy('s', 'A', capital, '2020-01-02', '2020-01-03')
    try:
        o.fill(i, symbol='A', quote_date='2020-01-03', price=price, session='2020-01-03')
    except ValueError as error:
        failures.append(f'Valid full-budget buy rejected: capital={capital}, price={price}, error={error}')
    else:
        assert o.cash >= 0
        assert abs(o.cash - (capital + o.fills[-1]['cash_change'])) <= 1e-8
        assert o.fills[-1]['notional'] + o.fills[-1]['fee'] <= capital + 1e-8
for finding in failures:
    print('FINDING:', finding)
if failures:
    raise SystemExit(1)
print('PASS: full-budget large-capital buys respect accepted budgets and ledger')

# Wider independent sweep: no tolerance is used for the purchase-budget cap.
rng = random.Random(83411)
checks = 0
for capital in [1e6, 1e8, 1e9, 1e12, 1e100, 1e300]:
    for bps in [0, 5, 250, 10000]:
        for trial in range(500):
            price = 10 ** rng.uniform(-3, 6)
            o = OMS(capital, bps)
            i = o.submit_buy('s', 'A', capital, '2020-01-02', '2020-01-03')
            o.fill(i, symbol='A', quote_date='2020-01-03', price=price, session='2020-01-03')
            fill = o.fills[-1]
            assert fill['notional'] + fill['fee'] <= capital
            assert o.cash == capital + fill['cash_change']
            assert o.cash >= 0
            assert np.isfinite([fill['units'], fill['fee'], o.cash]).all()
            checks += 1
print(f'PASS: {checks} additional full-budget cases from $1m to 1e300, 0–10000bps; strict budget and cash/ledger checks')
