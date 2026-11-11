# HW4: Gobind Singh

Predict the next 90-calendar-day return with a Ridge regression, then buy the 10 highest predictions at the start of each month. The graded window is April 1, 2016 through September 30, 2026. The strategy did not beat the S&P 500 total-return index.

## Run

From this folder, with the course environment and `equities_top2000.duckdb` in `Lectures/modeling/data/`:

```bash
python test_backtest.py
python 03_run_test.py
```

`test_backtest.py` checks the accounting on a few made-up prices. `03_run_test.py` rebuilds the full test. `04_report_gaps.py` reruns the development rank check and the overdue-lot list from the saved batch files. `01_fit_model.py` and `02_run_class.py` are the earlier class-database checks. An optional integer argument to `03_run_test.py` limits how many symbols are loaded.

Feature construction and the sklearn pipeline come from `Lectures/modeling/course_data.py`. The backtester in `backtest.py` is separate from the lecture backtester.

## Model

The features are the lecture set: 21-day, 63-day, and 12-month-minus-1-month momentum, 63-day volatility, 252-day drawdown, six filing-dated ratios, filing age, and sector. A filing is used only from the day after its filing date. Membership is the quarter of the next session, so a new buy has to be in that quarter's universe. An open lot is kept if the name later drops out.

The pipeline clips each numeric column at the training 1st and 99th percentiles, median-imputes, scales, one-hot encodes sector, and fits `Ridge(alpha=100)`. It is fit once on eligible signals from January 1, 2010 through December 31, 2015 whose 90-day outcome ended before April 1, 2016 (2,967,493 rows). It is not refit during the test. The saved object is `artifacts/ridge_90d_full.joblib`.

On the full universe, a separate Ridge fit using only signals through October 1, 2013 (so a 90-day label ends before 2014) has a 2014–2015 mean daily rank IC of about -0.025 across 503 dates and 1,001,652 validation rows. The class-file check had been about +0.04. This full-universe check was computed from the saved development rows after the test and was not used to change the submitted model. `04_report_gaps.py` reproduces it.

Symbols are loaded in batches of 200 so the full panel is never held in memory. Each batch keeps training rows, test-period features, and adjusted closes from March 2016 on.

## Trading

Start with $1,000,000, long only, no leverage. Cash earns zero. A signal is the month-end close. The buy fills at the next session's close, using a budget fixed at the signal: one third of portfolio value, capped by the cash available then, split equally across the top 10 scores. A sale on the fill date does not increase that budget. Each lot is held 90 calendar days and sold at the first later close. If that close is missing, the lot stays open and is marked at the last close. The fee is 10 basis points of notional on the buy and on the sale, taken out of the buy budget. Marks use adjusted closes, so dividends are not added again.

`test_backtest.py` checks three cases: the fee stays inside the budget, a same-day sale does not enlarge an order planned the day before, and a Sunday maturity sells on Monday.

## Test result

Both portfolios start at $1,000,000. The benchmark buys `^SP500TR` at the first test close and holds it. CAGR uses calendar days and 365.25. Sharpe uses daily simple returns. The risk-free rate is the 2% annual rate required by the assignment, converted to a daily rate with `(1.02) ** (1 / 252) - 1` and subtracted from each daily return. Cash in the strategy still earns zero. The chart is `artifacts/cumulative_return.png`.

| | Strategy | S&P 500 total return |
| --- | ---: | ---: |
| Final value | $1,018,134 | $4,402,747 |
| CAGR | 0.17% | 15.2% |
| Max drawdown | -73.1% | -33.8% |
| Sharpe | 0.16 | 0.77 |

There were 2,263 fills and 33 lots still open on September 30, 2026. Ten of those lots were still inside their 90-day hold. The other 23 were past due and had no later close, so they stayed marked at the last close. Together those 23 are about $469,035 of the ending $1,018,134, and the cash spent on them was never returned. The row-level list is `artifacts/overdue_open_lots.csv`.

| Symbol | Due | Last close | Marked value |
| --- | --- | --- | ---: |
| SOFIW | 2022-03-01 | 2021-12-06 | $49,901 |
| ECOL | 2022-06-30 | 2022-04-29 | $33,793 |
| BTRS | 2023-01-30 | 2022-12-15 | $17,835 |
| VVNT | 2023-05-02 | 2023-03-09 | $28,276 |
| MAXR | 2023-05-30 | 2023-05-02 | $8,579 |
| TA | 2023-07-02 | 2023-05-15 | $25,429 |
| TA | 2023-07-30 | 2023-05-15 | $3,001 |
| RETA | 2023-11-30 | 2023-09-26 | $26,201 |
| IMGN | 2024-02-29 | 2024-02-13 | $20,976 |
| BVH | 2024-04-01 | 2024-01-16 | $0 |
| IMGN | 2024-05-01 | 2024-02-13 | $25,105 |
| KAMN | 2024-06-30 | 2024-04-18 | $1,060 |
| DCPH | 2024-09-01 | 2024-06-11 | $24,274 |
| RDFN | 2025-07-30 | 2025-07-01 | $25,179 |
| CIFRW | 2025-12-30 | 2025-12-26 | $12,308 |
| SCS | 2025-12-30 | 2025-12-09 | $14,038 |
| MFH | 2026-02-01 | 2026-01-12 | $15,709 |
| CIFRW | 2026-03-01 | 2025-12-26 | $4,787 |
| MFH | 2026-03-01 | 2026-01-12 | $3,408 |
| RCB | 2026-05-31 | 2026-04-21 | $93,206 |
| FCRX | 2026-06-30 | 2026-05-22 | $5,607 |
| ASGN | 2026-07-30 | 2026-05-22 | $14,526 |
| FCRX | 2026-07-30 | 2026-05-22 | $15,840 |

The strategy finished slightly above its starting capital and well behind the benchmark. The full-universe rank IC was negative, the book is concentrated in 30 overlapping names, and a large drawdown did most of the damage. The next change I would try is a second reported value that drops a lot from the portfolio after five sessions without a quote, so a stuck last close is not carried for years.

## Limits

The quarterly universe is availability-aware, but it is not a certified point-in-time membership list. Filings in the file are the latest vendor values, sector is a current snapshot, and there is no reliable delisting-cash table. A name that stops trading is marked at its last close until a price comes back, which can overstate the position. The loader dropped 275 nonpositive adjusted closes and a few off-calendar price rows. Transaction costs are a flat 10 basis points. The model was not tuned on the test window.

I used Cursor to help structure the batching and the backtest. The feature definitions and the pipeline are the lecture code.
