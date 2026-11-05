# HW2: Scalable Stock Analysis Pipeline

## How to run

1. Copy the course database (bCourses) to `data/data.db` in this folder. It is gitignored and never committed.
2. Set up the environment: `uv sync --locked` at the repository root (or `pip install -r requirements.txt`), and select that Python as the notebook kernel.
3. Run `01_data_cleaning.ipynb` top to bottom. It writes `ohlc_hw_cleaned` into `data/data.db`; the raw `ohlc_hw` table is left unchanged.
4. Optional: run `02_stock_report_template.ipynb` on its own (defaults: AAPL, 2024).
5. Run `python 03_run_all_stocks.py`. Output goes to `reports/{run date}/`.

## Key design decisions

- **Order:** normalize tickers → remove duplicates → diagnose (Quality Report) → repair → validate → save. Each step is a function, so the same code runs on the real data and on the small injected examples.
- **Reference table:** the supplied `ohlc` table is used as each ticker's trading calendar and as the source of date-specific adjustment factors (`adj_close / close`).
- **Quarantine over guessing:** rows that cannot be repaired defensibly are set aside with a reason, not filled from another ticker or date.
- **Report template:** it raises a clear error when a window has fewer than 21 prices, so the runner records that ticker as failed.
- **Runner:** results are collected with scrapbook only from reports that succeeded in this run. Stale or failed notebooks in the folder are never read.

## Data issues and handling

| Issue | Found | Handling |
|---|---|---|
| HTML markup in ticker labels | 150 labels → 30 tickers | Normalized; original kept in `raw_ticker` |
| Exact duplicate rows | 29,460 | Removed |
| Conflicting `(ticker, date)` rows | 0 | Would be quarantined (shown on an injected example) |
| Missing trading days | 0 | Reported against the `ohlc` calendar (detector shown on an injected example) |
| Missing `adj_close` | 18,374 | Rebuilt as `close ×` that date's `ohlc` factor |
| `adj_close` 4–24× too large | 10,417 | Detected against the reference factor; rebuilt the same way |
| `open = 0` on a ticker's first day | 5 | No earlier price to fill from → quarantined |
| Zero volume | 75 | Flagged (possible stale quote); kept |
| `high < close`, `low > close`, missing raw prices or volume | 0 | Repairs shown on an injected example |

Result: 368,246 rows kept and 5 quarantined. All validation checks pass after cleaning. Repair flags (`price_filled`, `price_interpolated`, `adj_close_rebuilt`, `volume_filled`) and the review flag `return_outlier` are saved with the data.

## Outliers

- **3σ screen (simple raw-close returns, per ticker, `ddof=1`):** 5,056 flags.
- **Confirmed errors (4):** price printed on zero volume, flagged, and fully reversed by a flagged return the next day. These are three TRV days at double the price and one AXP day 13% too low. `close` and `adj_close` are interpolated between the neighboring days; open/high/low are set to the new close (the original bars were flat).
- **Kept (5,052):** raw OHLC matches the vendor `ohlc` table on every row, so these are not transcription errors in this file. They stay in the data with the `return_outlier` flag. Their causes (news, corporate actions) are not verified. Raw prices are already split-adjusted, so splits do not cause jumps.
- **Impossible price relationships:** Option 1 (set to NaN, then forward-fill within the ticker). The only real cases are the 5 first-day zero opens, which have no earlier value, so they are quarantined.

## Assumptions

- The `ohlc` reference is trusted for trading dates and adjustment factors. Rows dated on weekends in 1974–75 also appear in the reference, so they are kept as vendor trading dates.
- **Analysis window:** the six calendar months before the month of the latest cleaned date (2024-12-06), which is 2024-06-01 to 2024-11-30.
- **Shorter histories:** 22 of the 30 tickers end on 2024-10-18, so their reports cover only June to mid-October. Only AMZN, CVX, DIS, IBM, KO, MRK, NKE and PG reach December. Each report prints its actual coverage. Returns and Sharpe ratios for shorter histories are not directly comparable.
- **Metric conventions:** 252 trading days per year and a 2% annual risk-free rate. Report metrics use daily log returns on adjusted close.

## Interpreting the reports

- **Price (top left):** adjusted close, red when 20-day volatility is above the report's mean, green at or below it, and gray during the 20-day warm-up. The blue line is the 50-day moving average, which starts once 50 prices exist.
- **Daily returns (top right):** green for up days, red for down days.
- **Rolling volatility (bottom left):** 20-day annualized volatility, shaded above and below its mean.
- **Drawdown (bottom right):** distance below the running peak, with the worst point marked.
- **Summary table:** total return (`last / first − 1`), compounded annualized return, whole-period annualized volatility, Sharpe ratio, maximum drawdown, best and worst day, share of positive days, and average up and down days. Undefined values show as N/A.
- **`portfolio_summary.csv`:** one row per successful report, sorted by Sharpe ratio. Its `avg_volatility` is the mean of the 20-day rolling volatility, which differs from the whole-period volatility in each report's table. `risk_return.png` plots average volatility against total return, colored by Sharpe ratio.

## Sample output

`reports/2026-10-30/` holds the IBM (highest Sharpe), MRK (lowest Sharpe) and AAPL (history ends early) reports, plus `portfolio_summary.csv`, `risk_return.png` and an empty `failed_stocks.txt` (30 of 30 succeeded). The other 27 reports are generated locally and excluded by `.gitignore`.
