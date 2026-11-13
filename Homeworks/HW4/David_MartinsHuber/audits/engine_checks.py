"""Independent round-two engine audit; prints edge probes, no production edits."""
from pathlib import Path
import sys
import warnings
import numpy as np
import pandas as pd

submission = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(submission))
from backtesting import Backtester, Strategy, Phase

class Plan(Strategy):
    def __init__(self, plans):
        super().__init__("independent")
        self.plans = plans
    def propose_buys(self, c):
        assert c.prices.index.max() <= c.signal_date
        assert c.membership.selection_date.le(c.signal_date).all()
        return self.plans.get(c.execution_date, {})

# Direct ledger oracle: quantities, maturities, cash and marks calculated
# independently; no expected values read from OMS, orders, or recorded fills.
for seed in range(15):
    rng = np.random.default_rng(917 + seed)
    calendar = pd.bdate_range("2019-10-01", periods=235).delete([10, 47, 120])
    names = ["A", "B"]
    wide = pd.DataFrame(30 * np.exp(np.cumsum(rng.normal(0, .012, (len(calendar), 2)), axis=0)),
                        index=calendar, columns=names)
    for col in range(2):
        for i in range(13 + col, len(calendar), 29):
            wide.iloc[i, col] = np.nan
    quotes = wide.rename_axis("date").reset_index().melt("date", var_name="symbol", value_name="adj_close").dropna()
    benchmark = pd.DataFrame({"date": calendar, "adj_close": np.linspace(100, 130, len(calendar))})
    membership = pd.DataFrame([dict(symbol=s, quarter_start=q, selection_date=q - pd.Timedelta(days=1))
                              for q in pd.date_range("2019-10-01", "2020-10-01", freq="QS") for s in names])
    plans = {day: {names[i % 2]: 20.0} for i, day in enumerate(calendar[1:]) if i % 5 == 0}
    fee = .0013
    actual = Backtester(quotes, benchmark, membership, Plan(plans), initial_cash=2000, fee_bps=13).run(calendar[1], calendar[-1])
    cash, lots, marks, expected = 2000., [], {}, [2000.]
    for day in calendar[1:]:
        observed = wide.loc[day].dropna().to_dict()
        marks.update(observed)
        remaining = []
        for symbol, units, due in lots:
            if day >= due and symbol in observed:
                cash += units * observed[symbol] * (1 - fee)
            else:
                remaining.append((symbol, units, due))
        lots = remaining
        for symbol, budget in plans.get(day, {}).items():
            if symbol in observed:
                units = budget / (1 + fee) / observed[symbol]
                cash -= budget
                lots.append((symbol, units, day + pd.Timedelta(days=90)))
        expected.append(cash + sum(u * marks[s] for s, u, _ in lots))
    np.testing.assert_allclose(actual.history.nav.to_numpy(), expected, rtol=1e-12, atol=1e-8)
    assert abs(actual.history.cash.iloc[-1] - cash) < 1e-8
    assert len(actual.open_lots) == len(lots)
print("PASS: 15 separate 231-session independent ledger paths")

# Fixed-expected coverage and lifecycle cases, not tests of implementation shape.
def engine(b=None):
    return Backtester(quotes, benchmark if b is None else b, membership, Plan({}), initial_cash=1000)
for method in ("prepare", "advance", "finish"):
    e = engine()
    try:
        getattr(e, method)()
    except ValueError as exc:
        assert "begin" in str(exc)
    else:
        raise AssertionError("pre-begin call accepted")
for start, end in [(calendar[1], "2026-09-30"), ("2000-01-01", calendar[-1]),
                   (calendar[1] + pd.Timedelta(hours=1), calendar[-1]),
                   (pd.NaT, calendar[-1]), (calendar[-1], calendar[1])]:
    e = engine()
    try:
        e.begin(start, end)
    except ValueError:
        assert not e._started
    else:
        raise AssertionError(("invalid interval accepted", start, end))
e = engine()
with warnings.catch_warnings(record=True) as notices:
    warnings.simplefilter("always")
    result = e.run(calendar[1], "2026-09-30", allow_partial=True)
assert any("Partial" in str(n.message) for n in notices)
assert result.coverage.requested_end == pd.Timestamp("2026-09-30")
assert result.coverage.effective_end == calendar[-1]
assert result.coverage.range_clipped
print("PASS: coverage rejection, explicit partial metadata, and lifecycle errors")

# Refit cutoff and non-mutating prepare boundary.
class FitPlan(Plan):
    def needs_refit(self, c): return True
    def refit(self, data, c):
        assert data.audit.label_end.lt(c.signal_date).all()
        assert data.audit.available_date.le(c.signal_date).all()
        assert data.audit.date.le(c.signal_date).all()
        self.targets = list(data.y)
        assert c.fundamentals.filing_date.add(pd.Timedelta(days=1)).le(c.signal_date).all()
first = calendar[0]
panel = pd.DataFrame({"date": [first-pd.Timedelta(days=120)]*3, "symbol": ["A", "B", "C"],
                      "available_date": [first-pd.Timedelta(days=120)]*3,
                      "label_end": [first-pd.Timedelta(days=1), first, calendar[-1]],
                      "target": [1., 2., 999.], "feature": [1., 2., 999.]})
fundamentals = pd.DataFrame({"symbol": ["A", "A"], "fiscal_date": [first-pd.Timedelta(days=90)]*2,
                             "filing_date": [first-pd.Timedelta(days=2), calendar[-1]]})
e = Backtester(quotes, benchmark, membership, FitPlan({}), fundamentals=fundamentals,
               training_panel=panel, feature_columns=["feature"])
e.begin(calendar[1], calendar[2])
e.prepare()
assert e.strategy.targets == [1.]
assert not e.oms.fills and e.history_so_far.index[-1] == first
before = e.history_so_far.copy()
before.iloc[0, before.columns.get_loc("nav")] = 0
assert e.history_so_far.nav.iloc[0] == 1_000_000
e.advance(); e.prepare()
assert e.strategy.targets == [1., 2.]
print("PASS: strict label maturity, future-filing exclusion, detached snapshots, no execution during preparation")

# Fixed expectations for the newly discovered extreme normalization and
# warnings-as-errors cases. These assertions no longer merely print behavior.
huge = benchmark.copy(); huge["adj_close"] = 1e308
e = engine(huge)
probe = e.run(calendar[1], calendar[2])
np.testing.assert_allclose(probe.history.benchmark_nav, [1000., 1000., 1000.], rtol=0, atol=0)
assert e.phase == Phase.COMPLETE
print("PASS: constant finite benchmark near float maximum normalizes correctly")

e = engine()
try:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        e.begin(calendar[1], "2026-09-30", allow_partial=True)
except UserWarning:
    assert not e._started
    assert e.phase == Phase.READY
else:
    raise AssertionError("warnings-as-errors policy was not honored")
try:
    e.history_so_far
except ValueError as exc:
    assert "begin" in str(exc)
else:
    raise AssertionError("uninitialized history accepted")
# A failed initialization must not poison later valid use of this object.
probe = e.run(calendar[1], calendar[2])
assert e.phase == Phase.COMPLETE
assert probe.coverage.effective_end == calendar[2]
print("PASS: warning-as-error leaves engine unstarted and valid retry works")

unrepresentable = benchmark.copy()
unrepresentable.loc[unrepresentable.date.eq(calendar[1]), "adj_close"] = 1e-300
unrepresentable.loc[unrepresentable.date.eq(calendar[2]), "adj_close"] = 1e300
e = engine(unrepresentable)
try:
    e.run(calendar[1], calendar[2])
except ValueError as exc:
    assert "Benchmark normalization" in str(exc)
    assert e.phase == Phase.FAILED
    assert e.phases[-1]["phase"] == "failed"
    assert not any(record["phase"] == "complete" for record in e.phases)
else:
    raise AssertionError("unrepresentable benchmark growth returned a result")
print("PASS: genuinely nonrepresentable benchmark growth stops with FAILED state")
