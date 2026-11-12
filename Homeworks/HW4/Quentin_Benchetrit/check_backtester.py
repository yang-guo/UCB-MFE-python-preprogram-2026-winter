"""Accounting cases small enough to work out on paper.

Run with:  python check_backtester.py

The toy market has three stocks on a Monday-to-Friday calendar. Every expected
number below is computed from the rules of the assignment, not read back from
the backtester.
"""

import numpy as np
import pandas as pd

from hw4.backtest import Backtester, WeeklyTopN
from hw4.data import Market

FEE = 0.001     # 10 bps per side


class Scripted:
    """Strategy that places fixed orders on fixed decision dates."""

    def __init__(self, plan):
        self.plan = {pd.Timestamp(date): orders for date, orders in plan.items()}

    def orders(self, date, next_date, portfolio):
        return self.plan.get(date, {})


def toy_market():
    calendar = pd.bdate_range("2021-01-04", "2021-06-30")
    prices = pd.DataFrame(index=calendar, columns=["AAA", "BBB", "CCC"], dtype=float)
    prices["AAA"] = 50.0
    prices.loc["2021-04-09":, "AAA"] = 58.0
    prices.loc["2021-04-12":, "AAA"] = 60.0
    prices["BBB"] = 20.0
    prices.loc["2021-04-08", "BBB"] = 21.0
    prices.loc["2021-04-09":"2021-04-14", "BBB"] = np.nan     # halted through maturity
    prices.loc["2021-04-15":, "BBB"] = 22.0
    prices["CCC"] = 10.0
    prices.loc["2021-02-26", "CCC"] = 12.0
    prices.loc["2021-03-01":, "CCC"] = np.nan                 # never quoted again
    benchmark = pd.Series(100.0, index=calendar)
    sectors = pd.Series({"AAA": "one", "BBB": "two", "CCC": "three"})
    return Market(calendar, prices, prices, prices, prices * 0 + 1e6, benchmark, sectors)


def close(a, b):
    return abs(a - b) < 1e-6


def run(plan, **kwargs):
    return Backtester(toy_market(), Scripted(plan), fee_bps=10, max_gap=20, **kwargs).run(
        "2021-01-05", "2021-06-30")


# Decided Friday 8 Jan, filled Monday 11 Jan.
plan = {"2021-01-08": {"AAA": 100_000.0, "BBB": 50_000.0, "CCC": 30_000.0},
        "2021-03-01": {"CCC": 10_000.0}}
result = run(plan)
history, fills = result.history, result.fills
buys = fills[fills.side == "BUY"].set_index("symbol")
sells = fills[fills.side == "SELL"].set_index("symbol")

# 1. Purchase: the fee comes out of the budget, and cash falls by the budget.
units_a = 100_000 / (50 * (1 + FEE))
units_b = 50_000 / (20 * (1 + FEE))
units_c = 30_000 / (10 * (1 + FEE))
assert buys.loc["AAA", "date"] == pd.Timestamp("2021-01-11")
assert buys.loc["AAA", "signal_date"] == pd.Timestamp("2021-01-08")
assert close(buys.loc["AAA", "units"], units_a)
assert close(history.loc["2021-01-08", "cash"], 1_000_000)
assert close(history.loc["2021-01-11", "cash"], 820_000)
assert close(history.loc["2021-01-11", "nav"], 820_000 + units_a * 50 + units_b * 20 + units_c * 10)
print(f"1. buy 100,000 of AAA at 50: {units_a:.4f} units, NAV after fees "
      f"{history.loc['2021-01-11', 'nav']:,.2f}")

# 2. 11 Jan + 90 days is Sunday 11 Apr, so AAA is sold on Monday 12 Apr at 60,
#    not on Friday 9 Apr at 58.
assert buys.loc["AAA", "due_date"] == pd.Timestamp("2021-04-11")
assert sells.loc["AAA", "date"] == pd.Timestamp("2021-04-12")
proceeds_a = units_a * 60 * (1 - FEE)
assert close(sells.loc["AAA", "cash_change"], proceeds_a)
assert close(sells.loc["AAA", "pnl"], proceeds_a - 100_000)
assert close(history.loc["2021-04-12", "cash"], 820_000 + proceeds_a)
print(f"2. AAA due Sunday 11 Apr, sold Monday 12 Apr for {proceeds_a:,.2f}")

# 3. BBB has no quote from 9 to 14 April. It is carried at the 8 April close
#    and sold at the first quote after maturity, 15 April.
assert close(history.loc["2021-04-13", "stale_value"], units_b * 21)
assert sells.loc["BBB", "date"] == pd.Timestamp("2021-04-15")
proceeds_b = units_b * 22 * (1 - FEE)
assert close(sells.loc["BBB", "cash_change"], proceeds_b)
print(f"3. BBB halted over maturity, marked at 21 and sold on 15 Apr for {proceeds_b:,.2f}")

# 4. CCC stops quoting after 26 Feb. No sale is booked; the lot stays open at 12.
assert "CCC" not in sells.index
assert list(result.open_lots.symbol) == ["CCC"]
assert bool(result.open_lots.orphaned.iloc[0])
final_cash = 820_000 + proceeds_a + proceeds_b
assert close(history.cash.iloc[-1], final_cash)
assert close(history.nav.iloc[-1], final_cash + units_c * 12)
assert close(history.orphan_value.iloc[-1], units_c * 12)
print(f"4. CCC never quotes again: no fill, lot kept at {units_c * 12:,.2f}, "
      f"final NAV {history.nav.iloc[-1]:,.2f}")

# 5. The 1 March order for CCC meets no quote on 2 March: dropped, cash untouched.
dropped = result.events[result.events.event.str.startswith("no quote")]
assert len(dropped) == 1 and dropped.date.iloc[0] == pd.Timestamp("2021-03-02")
assert close(history.loc["2021-03-02", "cash"], 820_000)
print("5. order on a day without a quote is dropped, cash unchanged")

# 6. The two bracketing assumptions for the vanished stock.
lost = run(plan, orphan_recovery=0.0).history.nav.iloc[-1]
paid = run(plan, orphan_recovery=1.0).history
assert close(lost, final_cash)
assert close(paid.nav.iloc[-1], final_cash + units_c * 12) and close(paid.cash.iloc[-1], paid.nav.iloc[-1])
print(f"6. CCC written off: {lost:,.2f}; paid out at last close: {paid.nav.iloc[-1]:,.2f}")

# 7. Cash from fills alone reproduces the cash column, and NAV = cash + holdings.
rebuilt = 1_000_000 + fills.groupby("date").cash_change.sum().reindex(history.index).fillna(0).cumsum()
assert np.allclose(rebuilt, history.cash)
assert np.allclose(history.cash + history.holdings, history.nav)
print("7. cash rebuilt from the fills ledger matches every day")

# 8. Spending more than the cash on hand is refused.
try:
    run({"2021-01-08": {"AAA": 700_000.0, "BBB": 400_000.0}})
    raise AssertionError("overspending went through")
except RuntimeError as error:
    print("8. overspending refused:", error)

# 9. Weekly strategy on ten stocks with a fixed ranking: four tranches of five
#    names, 5% of NAV per name, 12% cap. Weeks 1 and 2 buy the top five; a third
#    purchase would take them to 15%, so week 3 moves on to names six to ten.
calendar = pd.bdate_range("2021-01-04", "2021-03-31")
names = [f"S{k:02d}" for k in range(10)]
flat = pd.DataFrame(100.0, index=calendar, columns=names)
market = Market(calendar, flat, flat, flat, flat, pd.Series(100.0, index=calendar),
                pd.Series({name: name for name in names}))
fridays = calendar[calendar.weekday == 4]
schedule = pd.DataFrame({"date": fridays - pd.Timedelta(days=1), "entry_date": fridays})
scores = pd.DataFrame([(date, name, -k) for date in schedule.date for k, name in enumerate(names)],
                      columns=["date", "symbol", "prediction"])
strategy = WeeklyTopN(scores, schedule, market.sectors, n_names=5, tranches=4,
                      name_cap=0.12, sector_cap=1.0)
weekly = Backtester(market, strategy, fee_bps=0).run("2021-01-05", "2021-03-31")
bought = weekly.fills[weekly.fills.side == "BUY"].groupby("date").symbol.apply(list)
assert bought.iloc[0] == names[:5] and bought.iloc[1] == names[:5] and bought.iloc[2] == names[5:]
assert np.allclose(weekly.fills[weekly.fills.side == "BUY"].notional.head(5), 50_000)
assert close(weekly.history.cash.loc[bought.index[3]], 0)
print("9. weekly tranches: 1/4 of NAV per week, name cap moves week 3 to the next five names")

print("\nall checks passed")
