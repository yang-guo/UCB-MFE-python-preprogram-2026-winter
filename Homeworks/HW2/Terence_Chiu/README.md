# HW2: Stock Analysis Pipeline – Terence Chiu

## Overview

1. `01_data_cleaning.ipynb` finds and fixes data quality issues in `ohlc_hw` and saves the validated `ohlc_hw_cleaned` table.
2. `02_stock_report_template.ipynb` is a parameterized risk report for one ticker and date range.
3. `03_run_all_stocks.py` runs the report for every ticker with papermill and builds a portfolio summary.

## Setup

- Install dependencies with `uv sync --locked` at the repo root, or `pip install -r requirements.txt` (Python 3.12).
- Put `data.db` from bCourses at `data/data.db` in this folder. It is git-ignored. The cleaning notebook adds `ohlc_hw_cleaned` to this copy and leaves `ohlc_hw` unchanged.

## How to run

From this folder, with the environment active:

1. Run `01_data_cleaning.ipynb` top to bottom (Restart & Run All) to create `ohlc_hw_cleaned`.
2. Optional: run `02_stock_report_template.ipynb` for one stock by editing the `parameters` cell (`ticker`, `start_date`, `end_date`, `db_path`).
3. Run `python 03_run_all_stocks.py`. Output goes to `reports/{run date}/`: one `{TICKER}_report.ipynb` per successful ticker, `failed_stocks.txt`, `portfolio_summary.csv` and `risk_return_profile.png`.

## Key design decisions

- The raw `ohlc_hw` table is never modified. `ohlc_hw_cleaned` keeps the original ticker label (`raw_ticker`) and the repair/review flags (`adj_close_repaired`, `volume_repaired`, `price_filled`, `3sig_tail`, `zero_volume`).
- The supplied `ohlc` table is only used as each ticker's trading calendar and as the source of date-specific adjustment factors. The dirty data is cleaned row by row, not replaced with it.
- Where the real data has no cases of an issue (missing dates, `high < close`, `low > close`, infinite prices, missing prices or volume), a small injected example shows the check and repair work.
- Reports read `ohlc_hw_cleaned` read-only and stop with a clear error if the table is missing, the dates are reversed, or there are fewer than 21 prices.
- Metrics use daily log returns of adjusted close, 252 trading days and a 2% risk-free rate. Undefined values (e.g. Sharpe with zero volatility) are `NaN` and shown as `N/A`. Rolling warm-up periods are left empty, not filled.
- The summary only collects metrics (saved with scrapbook) from reports that succeeded in the current run, so failed or old notebooks in the folder are ignored.
- The script registers a temporary kernel pointing at the current Python interpreter, like the lecture's `report_runtime` helper, so nothing gets installed.

## Data cleaning and outlier handling

Details and before/after statistics are in `01_data_cleaning.ipynb`.

- Tickers: the 150 raw ticker strings (HTML-wrapped, mixed styles) reduce to 30 after stripping tags and whitespace and upper-casing. This is done before any grouping.
- Duplicates: 29,460 exact duplicate rows removed. No conflicting `(ticker, ts)` keys.
- Missing trading days: none, checked against each ticker's own dates in `ohlc`. The detector is shown catching a deliberately removed date.
- Missing prices and volume: none, so nothing was dropped or filled. The repair code is shown on an injected example.
- `adj_close`: 18,374 values were missing and 10,417 disagreed with the reference factor (the raw max was $10,442 vs. $605 after cleaning). All 28,791 are rebuilt as `close × (ohlc.adj_close / ohlc.close)` for the same `(ticker, ts)` and flagged. The factor is never carried across dates.
- Impossible prices: no `high < close` or `low > close`. Five rows have `open = 0` (CVX, JNJ, MRK, PG on 1962-01-02; MCD on 1966-07-05). Each is the ticker's first day, so there's no earlier price for options 1 or 2, and option 3 would make up a value. I quarantined them with a reason.
- Extreme returns: 5,056 returns are more than 3 standard deviations from their ticker's mean. The raw OHLC matches the `ohlc` reference exactly, so these look like real market moves (the most flags are in 2008, 2000, 2020 and 1987), not data errors. They are kept and flagged in `3sig_tail`; nothing is interpolated.
- Zero volume: 75 days, all 1962–1982, kept and flagged.
- Row count: 397,711 raw − 29,460 duplicates − 5 quarantined = 368,246 rows saved.
- Severity: high means the price is wrong or unusable and would break the returns (missing or bad `adj_close`, nonpositive or non-finite prices, OHLC violations); medium means it needs review but may be real (extreme returns, missing days); low is informational (zero volume).

## Assumptions

- Analysis window: the six calendar months before the month of the latest cleaned date. Data ends 2024-12-06, so the window is 2024-06-01 to 2024-11-30. The output folder is named by run date, separately.
- 22 of the 30 tickers end on 2024-10-18, so their reports cover about 4.5 months. Each report shows its actual first and last dates, and the CSV has `first_date`, `last_date` and `trading_days`, so keep this in mind when comparing tickers.
- `avg_volatility` in the summary is the mean of the 20-day rolling volatility, not the whole-period volatility in each report.
- `ohlc` is trusted for trading calendars and adjustment factors, so holidays are never counted as missing days.
- The vendor's raw OHLC is already split-adjusted but not dividend-adjusted, and `adj_close` adjusts for both. So splits don't show up as jumps in raw returns.

## How to interpret the reports

- Top left: adjusted close, red when 20-day volatility is above its mean for the period, green at or below, gray during warm-up. The dashed line is the 50-day moving average.
- Top right: daily log returns, green up and red down.
- Bottom left: annualized 20-day volatility, shaded red above its mean and green below.
- Bottom right: drawdown from the running peak, with the maximum drawdown marked.
- Summary table: total return uses first and last adjusted close; annualized figures are compounded from log returns. These are backward-looking statistics over at most six months, not forecasts.

## Outputs included

The full batch ran locally on 2026-10-30 (30/30 succeeded). Committed from `reports/2026-10-30/`:

- `IBM_report.ipynb`: highest Sharpe, full coverage
- `MRK_report.ipynb`: lowest Sharpe, full coverage
- `AAPL_report.ipynb`: history ends 2024-10-18, showing a short-coverage report
- `portfolio_summary.csv`: all 30 tickers sorted by Sharpe (`report_path` also lists the reports that aren't committed)
- `risk_return_profile.png`: risk/return scatter

## Acknowledgement

Parts 2 and 3, this README, and parts of Part 1 (the repair, validation and write-up cells) were drafted with Claude Code from the Lecture 4 notebooks. I reviewed and ran all of it.
