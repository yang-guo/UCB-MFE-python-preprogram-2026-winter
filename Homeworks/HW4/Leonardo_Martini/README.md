# HW4 - Leonardo Martini

## How to run

Put `equities_top2000.duckdb` in `Lectures/modeling/data/` and run from the repo root:

```bash
uv run python Homeworks/HW4/Leonardo_Martini/hw4.py
```

Results go to `results/` (tables and checks in `RESULTS.md`, the chart, daily NAV, trades and the fitted model for HW5).

## What I did

- Features and labels: I reused the course code in `Lectures/modeling/course_data.py` (momentum, volatility, drawdown, ratios from the quarterly reports, sector). Financial data is used only after the filing date.
- Target: 90-day return from the next close. I train on its rank inside each month, because the raw returns have huge outliers from bad data (up to +100,000%).
- Model: I compared Ridge and gradient boosting on 2014 and 2015 (training only on outcomes already finished) and kept the one with the best rank IC (boosting, 0.073 vs 0.072). I refit it once a year.
- Strategy: every month-end I buy the 20 best-scored stocks among the 1,000 largest with a price of at least $5, equal weight, with at most one third of NAV. Each lot is sold after 90 days.

## Backtester

My own (`Backtester` in `hw4.py`). Each day it sells matured lots, buys the orders from the previous close, and values cash plus positions. Assumptions: $1M start, long only, 5 bps per trade, adjusted prices, cash at 0%. If there is no price on the entry day I skip the order. If a held stock has no price I use its last price.

Checks (in `results/RESULTS.md`): cash never negative, cash from the trade list equals the book cash, all sales after 90 days, all buys the session after the signal, and one lot checked by hand.

## Results (2016-04-01 to 2026-09-30)

| | Strategy | S&P 500 TR |
|---|---|---|
| Final value | $1,938,411 | $4,402,747 |
| CAGR | 6.51% | 15.17% |
| Max drawdown | -38.66% | -33.79% |
| Sharpe | 0.33 | 0.77 |

![Cumulative return](results/cumulative_returns.png)

I did not beat the benchmark (about $2.46M less). The signal is weak (IC around 0.07), the equal-weight portfolio has little exposure to the big tech stocks that drove the index, and about 19% of the money stayed in cash. The strategy lost much more in 2022 and never recovered.

## Limitations and next steps

- My first version used raw returns and I also ran it on the test period (CAGR about 1%). I then switched to the rank target and the size/price filter, so my choices were not fully fixed before seeing a test result.
- The data has no delisting proceeds; stocks that stop trading are kept at their last price, which can be too optimistic.
- In validation Ridge had a better top-20 return than boosting, so it might be the better choice. Next I would try investing all the cash, sector limits and more features.

## Use of AI

I used Claude to help me write and debug the code and this README. I checked the results and can explain each step.
