# HW2: Stock Analysis Pipeline (Emil Blaignan)

Cleans the `ohlc_hw` table, builds a parameterized risk report for one stock, and runs that report for all 30 stocks.

## How to run

Run everything from `Homeworks/HW2/Emil_Blaignan/` with the course environment active (`uv sync --locked` at the repo root, then `source .venv/bin/activate`; `requirements.txt` lists the packages used).

1. **Database:** put the bCourses `data.db` at `data/data.db` in this folder. It is not committed. If `data/data.db` does not exist, the cleaning notebook copies it from `Lectures/Lecture 4/data/data.db`.
2. **`01_data_cleaning.ipynb`:** run all cells. It reads `ohlc_hw` and the reference table `ohlc` and writes `ohlc_hw_cleaned` (plus `ohlc_hw_quarantine` and `ohlc_hw_ticker_audit`) into the local `data/data.db`. The raw `ohlc_hw` table is not changed.
3. **`02_stock_report_template.ipynb`:** the parameterized report (the `parameters` cell is tagged). You can run it by hand for one stock.
4. **`python 03_run_all_stocks.py`:** runs the template with papermill for every ticker in `ohlc_hw_cleaned`, saves everything in `reports/<run date>/` and prints `Successfully generated N/M reports`.

`reports/<run date>/` has one `<TICKER>_report.ipynb` per stock, `portfolio_summary.csv`, `risk_return_profile.png` and `failed_stocks.txt` (empty when nothing failed).
Only 3 sample reports (AAPL, IBM, MRK) are committed, with the CSV, the PNG and `failed_stocks.txt`. The rest of the batch stays local (see `.gitignore`).

## What the data looked like (Part 1)

| Finding | Count |
|---|---|
| Raw ticker strings (HTML markup around the symbol) | 150 -> 30 tickers |
| Exact duplicate rows | 29,460 (no conflicting `(ticker, ts)` keys) |
| Missing `adj_close` | 18,374 (raw prices and volume are complete) |
| `adj_close` different from the `ohlc` table | 10,417 (4x to 24x too large) |
| `open = 0` (also outside high/low) | 5 |
| Zero volume days | 75 |
| Returns more than 3 std from the ticker mean | 5,056 |
| Bad prints (extreme return and zero volume, back to normal the next day) | 4 (AXP 1972-12-14, TRV 1976-10-28, 1978-12-20, 1981-08-10) |
| Missing dates compared with `ohlc` | 0 |

Rows: 397,711 raw -> 368,251 after removing duplicates -> 368,251 kept, 0 in quarantine.

## Key design decisions

- **Tickers** are cleaned (tags removed, stripped, upper-cased) before deduplicating. The original labels are saved in `ohlc_hw_ticker_audit`.
- **Missing dates:** the dates of each ticker in the `ohlc` table are the reference calendar. There are none missing; I tested the detector on a small example with one removed date. No prices are invented for missing dates.
- **Adjusted close:** missing or wrong `adj_close` values are rebuilt as `close * factor`, with `factor = adj_close / close` from the same `(ticker, ts)` in `ohlc`. Good values are not touched and no factor is forward-filled. Rows with no factor would go to quarantine (none did).
- **Impossible prices (the 5 rows with `open = 0`):** I use the previous day's close if it is inside today's `[low, high]`, otherwise the middle of low and high. All 5 are the first day of a ticker, so they get the middle.
- **Other cases the data does not have** (all prices missing, missing high/low/close, missing volume, a gap at the start of a ticker) are handled in `clean_data` and tested on a small made-up table in the notebook: rows with all prices missing are dropped before filling, missing high/low are forward-filled inside the ticker, unfixable rows go to quarantine, and missing volume becomes 0 with a `volume_repaired` flag.
- **Severity** in the Data Quality Report: high = impossible or unusable values (missing/non-positive prices, open or close outside the range, adj_close different from `ohlc`), medium = needs a decision (missing adj_close or volume, missing dates, extreme returns), low = zero volume (only a warning).
- Every repair has a flag column in `ohlc_hw_cleaned`: `price_repaired`, `volume_repaired`, `adj_close_repaired`, `price_interpolated`, and `return_outlier`.

## How I handled the outliers

- The raw prices in `ohlc_hw` are identical to `ohlc`, so a big return is not automatically an error. Most of the 5,056 flagged returns are normal big market days (for example AAPL -52% on 2000-09-29, and the adjusted price fell with it). I did not check the news for each day.
- **Bad prints:** 4 of the flagged days have zero volume, the price jumps, and the next day it is back to normal. A real move would not come back right away with nothing traded. So an extreme return **together with** zero volume is treated as a data error. The prices and `adj_close` on those days are replaced by the average of the day before and the day after (`price_interpolated = True`). The `ohlc` table has the same wrong values on those days, so `adj_close` there is interpolated too.
- **Everything else is kept** and only flagged with `return_outlier`. After the 4 fixes there are 5,173 flagged returns (a bit more than before, because removing the TRV bad prints lowers TRV's standard deviation).

## Assumptions

- The `ohlc` table is the reference for each ticker's dates and for the adjustment factor.
- Risk-free rate 2% and 252 trading days per year (course approximation). The Sharpe ratio shows N/A when there are fewer than 2 returns or zero volatility.
- The batch window is the six calendar months before the month of the latest cleaned date (latest = 2024-12-06, so 2024-06-01 to 2024-11-30). The report folder is named after the day the script is run.

## How to read the reports

- **Top left:** adjusted close and 50-day moving average. The line is red when the rolling volatility is above its average for the report, green otherwise, gray while it is not available yet (first 20 returns). The color is decided looking back at the whole period.
- **Top right:** daily log returns. **Bottom left:** 20-day annualized volatility and its average. **Bottom right:** drawdown from the running peak with the maximum drawdown marked.
- The summary table gives total return, annualized return and volatility, Sharpe ratio, maximum drawdown and daily-return statistics.
- **Different coverage:** in the data, 22 tickers stop on 2024-10-18 and the others go to the end of November. Each report prints its real first and last date, and `portfolio_summary.csv` has `first_date`, `last_date` and `trading_days`. Compare the shorter histories carefully, since they cover fewer days. Nothing is extended past the last observation.
- The summary is descriptive. It is not investment advice.
