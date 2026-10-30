# HW2: Cleaning, Risk Reports, and Automated Multi-Stock Analysis

Ryan Beauchamp

This folder contains a cleaning notebook, a parameterized single-stock risk report, and a script that runs the report for every stock and ranks the results.

| File | Purpose |
|---|---|
| `01_data_cleaning.ipynb` | Inspects, diagnoses, repairs, and saves the cleaned table `ohlc_hw_cleaned` |
| `02_stock_report_template.ipynb` | Parameterized risk report for one ticker |
| `03_run_all_stocks.py` | Runs the report for all tickers with papermill and builds the summary |
| `requirements.txt` | Package versions used (Python 3.12) |
| `reports/<run date>/` | Sample reports, `portfolio_summary.csv`, `coverage.csv`, `risk_return.png` |

The database and most of the generated reports are intentionally not committed.

## How to run

1. From the repo root: `uv sync --locked` (or `uv pip install -r requirements.txt`).
2. Copy `data.db` from Canvas into `data/` in this folder.
3. Run `01_data_cleaning.ipynb` top to bottom. It creates `ohlc_hw_cleaned` in your local copy of the database and never modifies `ohlc_hw` or `ohlc`.
4. From this folder: `uv run python 03_run_all_stocks.py`.
   - `--tickers AAPL MSFT` runs a quick test on a few tickers.
   - `--test-failure` adds a fake ticker to demonstrate the error handling.
5. `02_stock_report_template.ipynb` can also be run alone by editing its tagged `parameters` cell (`ticker`, `start_date`, `end_date`, `db_path`).

## Part 1: Data cleaning

**Result:** 397,711 raw rows, minus 29,460 exact duplicates, minus 5 quarantined rows, leaves **368,246 rows across 30 tickers**.

| Issue | Count | Decision |
|---|---|---|
| Exact duplicate rows | 29,460 | Dropped (0 conflicting keys) |
| Missing `adj_close` | 18,374 | Rebuilt from validated close times a reference factor |
| Invalid `adj_close` (roughly 4 to 24 times too large) | 10,417 | Same |
| `open = 0` (first day of 5 tickers) | 5 | Quarantined |
| One-day spike bars | 4 | Interpolated and flagged |
| Zero-volume days | 75 | Kept and flagged |
| 3-sigma return outliers | 5,174 | Kept and flagged |
| Missing trading dates | 0 | None found; detector tested on an injected gap |

**Tickers.** The raw ticker column had 150 strings, many wrapped in HTML. I stripped the markup, trimmed whitespace, and uppercased them, which gives 30 tickers. This has to happen before grouping or duplicate checks, otherwise one stock's history is split across several keys and its returns are wrong. There was an assortment of characters in the string which had to be removed to get down to the intended ticker within.

**Duplicates.** Every duplicate was an exact copy, so dropping them loses nothing. Rows that share a `(ticker, ts)` key but disagree would be quarantined for review instead of keeping the first row. I demonstrated this on a small injected example because the real data has none.

**Missing dates.** I compared each ticker against the supplied `ohlc` calendar. All 30 tickers are complete, so the result is zero. I tested the detector by removing one real date, which it found.

**Adjusted close.** Raw open, high, low, close, and volume match the reference `ohlc` table exactly. Only `adj_close` is damaged: 18,374 values are missing and 10,417 are present but wrong. Two detectors that do not use the reference found all of the wrong values, with no misses and no false alarms against the reference: (1) `adj_close > close` with a 1e-6 tolerance, and (2) a spike test comparing each day's `adj_close / close` to the ticker's centered 11-day median, with a 2.5 threshold chosen from the visible gap between good and bad rows. I rebuilt only the flagged rows as validated close times the reference's date-specific adjustment factor. The other 339,455 vendor values already equal the reference. Because raw close matches the reference, the repaired values also equal the reference's `adj_close`. This is a documented use of a trusted source, not new information.

**Quarantined rows.** Five rows have `open = 0` (PG, CVX, JNJ, and MRK on 1962-01-02, and MCD on 1966-07-05). Each is that ticker's first day, probably a missing open stored as zero. There is no earlier day to fill from, so I quarantined them.

**Zero volume** is kept and flagged. A zero is a review item, not a proven error.

### Outlier handling

I flag returns more than 3 sample standard deviations from the ticker's own mean (simple raw-close returns, per ticker, both tails). A flag is not an error.

- **Kept and flagged:** large moves that cluster across many tickers on the same days (for example 23 tickers moved more than 10% on 1987-10-19, with more clusters in 2008 and 2020). Many unrelated stocks moving together looks market-wide, not like data errors.
- **Repaired:** four isolated spike days (three TRV, one AXP). Each had zero volume, a large move, and an immediate reversal the next day. I interpolated them from neighboring days and recomputed `adj_close`.
- **Caveat:** the AXP case is lower confidence, since it is a 12.6% dip and not an exact doubling like TRV. It meets the same rule, which I applied consistently.
- After the repair TRV's standard deviation fell, so its 3-sigma cutoff tightened and the outlier count rose (5,056 to 5,174).
## Part 2: Report conventions

- Adjusted close is used for returns and drawdown. Daily **log returns** are used for volatility, Sharpe, and return statistics.
- Rolling volatility uses a 20-day window, annualized with the square root of 252. Sharpe is `(mean*252 - 0.02) / (std*sqrt(252))`.
- Drawdown is measured from the running peak. Maximum drawdown is the minimum of that series.
- The 50-day moving average and the first 20 volatility values stay missing, never filled with zero. They show as absent or gray.
- The price line is red when rolling volatility is above the report's own mean, green otherwise, and gray during warm-up. This is retrospective coloring.
- A report needs at least 21 prices (20 returns); otherwise it raises a clear error. An undefined Sharpe (zero volatility or too few returns) shows as `N/A`.
- Calculations keep fractions. Percentages are only formatted for display.

## Part 3: Batch run

- **Window:** the six calendar months before the month of the latest cleaned observation. The latest date is 2024-12-06, so the window is 2024-06-01 to 2024-11-30. It is anchored to the data, not the computer clock.
- **Output folder:** `reports/{run date}/` uses the date the script was run, which is independent of the analysis window.
- **Errors:** each ticker runs inside its own try/except. A failure is logged, the loop continues, and the script prints `Successfully generated N/M reports` and writes `failed_stocks.txt`. I tested this with `--test-failure` (2 of 3 succeeded; the fake ticker failed). If every ticker failed, the script prints a message and creates no summary.
- **Collecting results:** each report records its metrics with Scrapbook, and the script reads those values only from successfully executed reports, never by scanning the folder (a failed or stale report could be there).
- **Result:** 30 of 30 reports generated.
- **Summary:** `portfolio_summary.csv` has seven columns (`ticker`, `avg_daily_return`, `avg_volatility`, `max_drawdown`, `sharpe_ratio`, `total_return`, `report_path`), sorted by Sharpe. Undefined Sharpe values are excluded from the top and bottom 10. `risk_return.png` plots average