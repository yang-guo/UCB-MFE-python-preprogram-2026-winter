# HW2 - Quentin Benchetrit

## How to run

1. Install the packages of `requirements.txt` (Python 3.12), or use the course environment (`uv sync --locked` at the repository root).
2. Download `data.db` from bCourses into `Lectures/Lecture 4/data/`.
3. From this folder, in this order:
   - run all cells of `01_data_cleaning.ipynb`. It copies the database to `data/data.db` (gitignored) and writes `ohlc_hw_cleaned` there;
   - `python 03_run_all_stocks.py`. It runs `02_stock_report_template.ipynb` for every ticker and writes the reports, `portfolio_summary.csv`, `risk_return_profile.png`, `failed_stocks.txt` and `run.log` to `reports/<run date>/`.

`reports/2026-10-29/` contains three of the thirty reports (IBM, best Sharpe ratio; MRK, worst; AAPL, shorter history) and the summary of the full run.

## Key design decisions

- The supplied `ohlc` table is used for the trading calendar, for the adjustment factor `adj_close / close` of each date, and as evidence. The delivery is cleaned, not replaced by it.
- All checks run before any repair, every changed value is flagged in `ohlc_hw_cleaned`, and every raw row is either kept, removed (exact duplicate) or quarantined with a reason. The notebook checks that the counts add up.
- Problems found: 150 ticker labels for 30 tickers (HTML, case, spaces), 29,460 exact duplicates, 18,374 missing and 10,417 inflated `adj_close` values (4 to 24 times too large), and 5 rows with an open of 0. The raw prices are identical to the reference, so problems in them are also in the reference.
- Missing or inflated `adj_close` is rebuilt as close times the reference factor of the same date. A factor is never carried to another date, so it cannot cross a split or a dividend.
- The 5 zero opens are on the first day of a history, with no earlier price to carry, so these rows are quarantined.
- Impossible OHLC rows would be repaired with Option 3 (keep open and close, widen high and low). The delivery has none; the rule is tested on injected data, like the other cases absent from the database.

## Outliers

The 3-sigma screen flags 5,056 returns. Over sixty years of daily prices this many large moves is expected, so a flag only triggers a review:

- 3 one-day spikes on TRV (1976, 1978, 1981): the price doubles for one day with zero volume and comes back the next day. They are treated as errors and interpolated between the neighbouring days.
- 5 corporate actions: the raw price drops but the adjustment factor steps up, so the adjusted price does not move much. Kept.
- MCD on 1968-01-02: the price falls by 75% for good while volume rises about four times. This looks like a split the vendor did not adjust for, but the data cannot confirm it, so the row is kept and flagged as `unexplained_level_shift`.
- All other flags are the same move in the reference raw and adjusted prices, with nothing pointing to an error. They are kept with the flag `market_move`.

## Assumptions

- The trading sessions of a ticker are its dates in `ohlc`. With that calendar, no session is missing.
- The six-month window is the six calendar months before the month of the latest date in the cleaned table (2024-12-06): 2024-06-01 to 2024-11-30, the same for every ticker.
- 22 tickers have no data after 2024-10-18. Their figures cover 97 sessions instead of 126, so comparing them with the other 8 is not like for like.
- Returns, volatility and Sharpe ratio use daily log returns of the adjusted close, 252 days a year and a 2% risk-free rate.

## How to read the reports

- Price chart: red when the 20-day volatility is above its average over the report, green below, gray before the first estimate. The average is computed on the whole window, so the coloring describes the period after the fact.
- Drawdown: distance below the previous peak on each date. The lowest point is the maximum drawdown.
- Summary table: annualized volatility uses all returns of the window. The `avg_volatility` column of `portfolio_summary.csv` is the average of the 20-day rolling volatility instead.
- The figures describe six months of history. A Sharpe ratio measured over such a short period says little about the next six months.
