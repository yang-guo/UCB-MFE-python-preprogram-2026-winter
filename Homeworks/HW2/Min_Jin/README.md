# HW2 – Stock Analysis Pipeline (Min Jin/ cashier42)

## How to run

Run all commands from this folder (`Homeworks/HW2/Min_Jin/`).

1. **Install dependencies:** `pip install -r requirements.txt`
2. **Add the database:** copy the course database from bCourses to `data/data.db`.
3. **Clean the data:** run all in `01_data_cleaning.ipynb`. It reads the raw `ohlc_hw` table and writes `ohlc_hw_cleaned` to `data/data.db`, plus `data_quality_report_before.csv`.
4. **Single report run:** open `02_stock_report_template.ipynb`, change the parameters cell
   (`ticker`, `start_date`, `end_date`) and Run All.
5. **All reports:** `python 03_run_all_stocks.py`. Outputs saved in `reports/{run date}/`:
   - `{TICKER}_report.ipynb`: one executed report per stock
   - `portfolio_summary.csv`: key metrics for every successful report, sorted by Sharpe ratio
   - `risk_return_profile.png`: risk vs. return scatter plot
   - `failed_stocks.txt`: tickers whose report failed (empty if none)

Only three sample reports (IBM, MRK, AAPL) and the summary CSV are committed; run step 5 to regenerate the full batch.

## Key design decisions

- **Keep, repair or quarantine, with flags.** Every change/ updates is explicit: every repaired value has a `flag_*` column in `ohlc_hw_cleaned`, and `adj_close_source` records whether `adj_close` came from the vendor or was rebuilt. The raw `ohlc_hw` table is never modified.
- **Reference data.** The supplied `ohlc` table is used as the reference for validating closes and rebuilding the adjustment factor.
- **One date window for all stocks.** The batch uses the six calendar months before the month of the latest cleaned observation (latest 2024-12-06 → 2024-06-01 to 2024-11-30), anchored to the data.
- **Collecting results.** Each report writes its key metrics to `reports/{date}/metrics/{TICKER}_metrics.json` via the `metrics_path` parameter, borrowing from the lecture's Scrapbook outputs. The batch script reads only the files from reports that succeeded in the current run.
- **Kernel.** The script runs reports with a temporary Jupyter kernel that points to the same Python running the script (similar to the lecture's `prepare_report_run`), so reports use consistent packages on any machine.
- **Failures.** Each ticker runs inside `try/except`. Failures are printed and saved to `failed_stocks.txt`, and the loop continues. If every ticker fails, no summary is created.

## Data cleaning summary (Part 1)

| Issue | Count | Handling |
|---|---|---|
| Exact duplicate rows | 29,460 | Removed (397,711 → 368,251 rows) |
| Open = 0 (outside the day's low/high) | 5 | Replaced with the same day's close and flagged |
| `adj_close` missing or inconsistent with the reference factor (4–24× inflated) | 28,791 | Rebuilt as `close × (ref_adj_close / ref_close)` and flagged |
| Zero-volume days | 75 | Kept and flagged (potentially no trading) |
| Return outliers (> 3σ from the ticker mean) | 5,056 | Kept and flagged (see below) |
| Missing trading days (vs. each ticker's reference calendar) | 0 | – |

Rules for problems not present in this data but handled by the pipeline (tested on injected demo rows): missing prices are forward-filled within each ticker, leading gaps with nothing to fill from are quarantined, rows with all prices missing are removed, missing volume becomes 0 with a flag, and a close outside [low, high] is clipped into that range.

## How outliers were handled, and why

Outliers were **flagged, not removed or changed**. Daily returns have fat tails, so large moves are expected, and all 5,056 flagged closes match the reference `ohlc` table. They are real prices, not data errors. Deleting them would understate volatility and drawdowns. The `flag_return_outlier` column allows a reviewer to inspect them.

## Assumptions

- The supplied `ohlc` table is a trustworthy reference for closes and adjustment factors.
- A missing trading day is judged against each ticker's own dates in the reference table, so exchange holidays and dates before listing aren't counted as missing.
- Risk-free rate is 2% per year; 252 trading days per year.
- Returns and drawdown use adjusted close. Volatility, Sharpe and daily-return statistics use daily log returns.
- **Shorter histories:** 22 stocks' supplied data ends on 2024-10-18 instead of 2024-11-29. No prices are extrapolated. Each report shows its actual coverage, the CSV includes `first_obs`, `last_obs` and `trading_days`, and the batch script lists these stocks.

## How to read the reports

Each report shows the requested window, the actual coverage, and notes about short or incomplete history.
- **Price & 50-day MA (top left):** price line colored by volatility: red = rolling volatility above the report's mean, green = at or below, gray = the first 20 days, before a volatility estimate exists. The dashed line is the 50-day moving average.
- **Daily returns (top right):** green bars for up days, red for down days.
- **Rolling volatility (bottom left):** 20-day annualized volatility with its mean (dashed). Red shading means above-average risk. green means below average.
- **Drawdown (bottom right):** percentage below the previous peak, and the deepest point is marked.
- **Summary table:** total and annualized return, volatility, Sharpe ratio, max drawdown, best/worst day, share of positive days, average up/down day. `N/A` means the value can't be calculated (e.g. too few returns).

**Portfolio summary:** in `risk_return_profile.png`, points further up and left have more return for less risk. Color shows the Sharpe ratio, and the five highest-return stocks are labeled.