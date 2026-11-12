"""Hand-checkable accounting tests for backtest.py.  Run:
    .venv/bin/python Homeworks/HW4/Emil_Blaignan/test_backtest.py
"""
import numpy as np
import pandas as pd

from backtest import execute, metrics

days = pd.bdate_range("2020-01-01", periods=200)


def frame(**cols):
    return pd.DataFrame(cols, index=days)


def one_pick(sym, i):
    return pd.DataFrame({"symbol": [sym], "entry_date": [days[i]]})


# 1. One trade, no costs: buy 20,000 sh @ 50, sell @ 60 on the 90-day maturity -> $1.2M exactly.
px = frame(A=np.r_[np.full(5, 50.0), np.linspace(50, 60, 195)])
exit_day = days[days.searchsorted(days[5] + pd.Timedelta(days=90))]
px.loc[exit_day:, "A"] = 60.0
eq, log, stale = execute(one_pick("A", 5), px, tranches=1, n_target=1, cost=0.0)
assert log.shares[0] == 1_000_000 / 50 and log.date[1] == exit_day
assert abs(eq.iloc[-1] - 1_200_000) < 1e-6 and stale == 0
assert abs(eq[days[4]] - 1_000_000) < 1e-9  # nothing before the fill

# 2. Costs: 10 bps each side -> 1,000,000 / 1.001 / 50 shares bought, proceeds shares*60*0.999.
eq, log, _ = execute(one_pick("A", 5), px, tranches=1, n_target=1, cost=0.001)
shares = 1_000_000 / 1.001 / 50
assert abs(log.shares[0] - shares) < 1e-9
assert abs(eq.iloc[-1] - shares * 60 * 0.999) < 1e-6

# 3. Never spend the same cash twice: two tranches the same day cannot exceed available cash.
px2 = frame(A=50.0, B=20.0)
picks = pd.DataFrame({"symbol": ["A", "B"], "entry_date": [days[5], days[5]]})
eq, log, _ = execute(picks, px2, tranches=1, n_target=2, cost=0.0)
spent = (log[log.side == "BUY"].shares * log[log.side == "BUY"].price).sum()
assert abs(spent - 1_000_000) < 1e-6 and abs(eq.iloc[10] - 1_000_000) < 1e-6

# 4. A later tranche only sees leftover cash (long only, no leverage).
picks = pd.DataFrame({"symbol": ["A", "B"], "entry_date": [days[5], days[6]]})
eq, log, _ = execute(picks, px2, tranches=1, n_target=1, cost=0.0)
assert min(eq) > 0
# cash after first buy is 0, so the second budget is min(equity, 0) = 0 -> no second purchase
assert (log.symbol == "B").sum() == 0

# 5. Gap handling: no quote at maturity -> next quote within 5 sessions; none -> last close, flagged.
px3 = frame(A=50.0)
m = days.searchsorted(days[5] + pd.Timedelta(days=90))
px3.iloc[m, 0] = np.nan
px3.iloc[m + 1:, 0] = 55.0
eq, log, stale = execute(one_pick("A", 5), px3, tranches=1, n_target=1, cost=0.0)
assert log.date[1] == days[m + 1] and stale == 0
px4 = frame(A=50.0)
px4.iloc[m:, 0] = np.nan  # stock stops trading
eq, log, stale = execute(one_pick("A", 5), px4, tranches=1, n_target=1, cost=0.0, exit_haircut=0.5)
assert stale == 1 and abs(eq.iloc[-1] - 500_000) < 1e-6  # carried 50 with a 50% haircut

# 5b. Exposure overlay: a 0.5 multiplier invests half the tranche; the rest stays as cash.
eq, log, _ = execute(one_pick("A", 5), px, tranches=1, n_target=1, cost=0.0,
                     exposure=pd.Series({days[5]: 0.5}))
assert log.shares[0] == 500_000 / 50 and abs(eq.iloc[-1] - (500_000 + 10_000 * 60)) < 1e-6

# 5c. Unequal weights: 70/30 split of a one-tranche budget, no costs.
pxw = frame(A=50.0, B=20.0)
pk = pd.DataFrame({"symbol": ["A", "B"], "entry_date": [days[5], days[5]], "weight": [0.7, 0.3]})
eq, log, _ = execute(pk, pxw, tranches=1, n_target=2, cost=0.0)
spent = dict(zip(log[log.side == "BUY"].symbol, log[log.side == "BUY"].shares * log[log.side == "BUY"].price))
assert abs(spent["A"] - 700_000) < 1e-6 and abs(spent["B"] - 300_000) < 1e-6

# 5d. Cash interest: with no trades, equity compounds at 2% by calendar days.
eq, _, _ = execute(pd.DataFrame({"symbol": [], "entry_date": []}), frame(A=50.0), cost=0.0, cash_rate=0.02)
exp = 1_000_000.0
for gap in np.diff(days.values).astype("timedelta64[D]").astype(int):
    exp *= 1 + 0.02 * gap / 365.0
assert abs(eq.iloc[-1] - exp) < 1e-6

# 6. Metrics on a known path.
v = pd.Series([100.0, 110.0, 99.0, 120.0], index=pd.bdate_range("2020-01-01", periods=4))
r = metrics(v)
assert abs(r["max_drawdown"] - (-0.10)) < 1e-12 and r["final_value"] == 120.0
print("all backtest accounting tests passed")
