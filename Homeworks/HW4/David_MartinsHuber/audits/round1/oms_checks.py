"""Independent OMS audit: normal-range invariants and recorded edge failures.

Run from HW4 with .venv/bin/python /tmp/hw4_oms_independent_audit.py.
Findings are printed rather than asserted as desired behavior.
"""
from pathlib import Path
import sys
import random
from copy import deepcopy
from dataclasses import asdict
import numpy as np

base = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(base))
from backtesting import OMS

rng = random.Random(47)
for trial in range(200):
    oms = OMS(10000, 7)
    expected_cash = 10000.0
    pending = []
    for k in range(5):
        budget = oms.available_cash * rng.uniform(.05, .3)
        pending.append((oms.submit_buy('audit', f'S{k}', budget, '2020-01-02', '2020-01-03'), budget))
    rng.shuffle(pending)
    for oid, budget in pending:
        if rng.random() < .3:
            oms.reject(oid)
        else:
            price = rng.uniform(1, 1000)
            expected_units = budget / (price * 1.0007)
            oms.fill(oid, symbol=oms.orders[oid].symbol, quote_date='2020-01-03', price=price, session='2020-01-03')
            expected_cash -= budget
            assert abs(oms.lots[oid].units - expected_units) < 1e-10
    assert abs(oms.cash - expected_cash) < 1e-9
    assert abs(oms.available_cash - expected_cash) < 1e-9
    for lot in list(oms.lots.values()):
        price = rng.uniform(1, 1000)
        expected_cash += lot.units * price * (1 - .0007)
        oid = oms.submit_maturity_sale(lot.lot_id, '2020-04-03')
        oms.fill(oid, symbol=lot.symbol, quote_date='2020-04-03', price=price, session='2020-04-03')
    assert abs(oms.cash - expected_cash) < 1e-8
    assert abs(oms.cash - (10000 + sum(f['cash_change'] for f in oms.fills))) < 1e-8
print('PASS: 200 independent normal-range cash/reservation/fee sequences')

o = OMS(100, 10)
i = o.submit_buy('audit', 'A', 50, '2020-01-02', '2020-01-03', model_version=4)
def snapshot(o):
    return (o.cash, o.available_cash, {k: asdict(v) for k, v in o.orders.items()}, dict(o.lots), deepcopy(o.fills))
for args in [dict(symbol='B', quote_date='2020-01-03', price=10, session='2020-01-03'),
             dict(symbol='A', quote_date='2020-01-02', price=10, session='2020-01-03'),
             dict(symbol='A', quote_date='2020-01-03', price=float('nan'), session='2020-01-03'),
             dict(symbol='A', quote_date='2020-01-03', price=10, session='2020-01-04')]:
    before = snapshot(o)
    try:
        o.fill(i, **args)
    except ValueError:
        pass
    else:
        raise AssertionError('Invalid quote accepted')
    assert snapshot(o) == before
print('PASS: invalid quote rejection is atomic in four independent cases')
o.fill(i, symbol='A', quote_date='2020-01-03', price=10, session='2020-01-03')
sale = o.submit_maturity_sale(i, '2020-04-02')
assert o.lots[i].model_version == 4
assert o.orders[sale].model_version == 4 and o.orders[sale].strategy_id == 'audit'
print('PASS: lot and maturity-sale provenance')

# Independent boundary oracle: cash must equal initial cash plus fill cash flows.
o = OMS(100, 20000)
i = o.submit_buy('audit', 'A', 100, '2020-01-02', '2020-01-03')
o.fill(i, symbol='A', quote_date='2020-01-03', price=10, session='2020-01-03')
i = o.submit_maturity_sale(i, '2020-04-03')
o.fill(i, symbol='A', quote_date='2020-04-03', price=10, session='2020-04-03')
ledger_cash = 100 + sum(f['cash_change'] for f in o.fills)
print('EDGE FINDING at oms.py:120-125: high_fee_cash=', o.cash, 'ledger_cash=', ledger_cash)
print('Expected: reject unsupported fees or insolvent sale before booking, not clip cash.')

# Positive finite scalar input can overflow derived quantity and corrupt a fill.
with np.errstate(all='ignore'):
    o = OMS(100, 5)
    i = o.submit_buy('audit', 'A', 100, '2020-01-02', '2020-01-03')
    o.fill(i, symbol='A', quote_date='2020-01-03', price=np.nextafter(0, 1), session='2020-01-03')
print('EDGE FINDING at oms.py:101-125: units=', o.lots[i].units, 'fee=', o.fills[-1]['fee'], 'status=', o.orders[i].status)
print('Expected: finite derived values or an atomic rejection.')
