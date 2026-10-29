# HW2: Stock Analysis Pipeline (Ziang Gao)

## Files
| File | Purpose |
|---|---|
| `01_data_cleaning.ipynb` | Finds data problems in `ohlc_hw`, repairs them, writes `ohlc_hw_cleaned` |
| `02_stock_report_template.ipynb` | Parameterized risk report for one ticker |
| `03_run_all_stocks.py` | Runs the template for all tickers and builds the portfolio summary |
| `requirements.txt` | Python packages |
| `reports/<run date>/` | Sample reports, `portfolio_summary.csv`, `risk_return_scatter.png` |

## How to run
1. Put a working copy of the course database at `data/data.db` (next to these files). The database is not in Git.
2. Install packages: `uv sync --locked` at the repository root, or `pip install -r requirements.txt`.
3. Run `01_data_cleaning.ipynb` from top to bottom (Restart and Run All). It reads `ohlc_hw` and writes `ohlc_hw_cleaned`, plus `ohlc_hw_quarantine`, into `data/data.db`. It can run again on a fresh copy. The raw table is never changed.
4. Run `python 03_run_all_stocks.py` from this folder. It runs `02_stock_report_template.ipynb` for every ticker with papermill. Do not run notebook 02 first. The script gives it the parameters.
5. Output: `reports/<run date>/<TICKER>_report.ipynb`, `portfolio_summary.csv`, `risk_return_scatter.png`, and `failed_stocks.txt` (one ticker per line, empty if none).

Only 3 sample reports and the summary are committed. The full batch is made locally.

## Key design decisions
- **Tickers.** The raw table has 150 labels (HTML markup, spaces, case) for 30 real tickers. I clean the labels first. The original label is kept in an audit table.
- **Duplicates.** About 29,460 exact duplicate rows are removed. No `(ticker, ts)` key has conflicting values.
- **Missing dates.** I use the dates of the supplied `ohlc` table for each ticker as the reference calendar. No ticker has missing dates. I test the detector on a small example.
- **Adjusted close.** 18,374 values are missing, and about 10,400 are present but 4 to 24 times too large. I detect the wrong values with a spike check on `adj_close / close`. I rebuild both kinds as `close x (ref_adj_close / ref_close)` from the `ohlc` table, only for those rows. Rows with a good vendor value keep it. I use `ohlc` only because its raw close equals mine on every row.
- **`open = 0` (5 rows).** I replace open with the midpoint of that day's high and low.
- **Quarantine.** Rows I cannot repair are saved with a reason in `ohlc_hw_quarantine`. No row needed it in this database. I test the code on injected examples.
- **Report.** Adjusted close, daily log returns, volatility annualized with sqrt(252), and Sharpe with a 2% risk-free rate, as the homework defines.

## How I handled outliers and why
- I flag days where the close-to-close return is more than 3 sample standard deviations from the ticker mean (both tails). This is a review flag, not a deletion rule.
- For each flagged day I compare the raw return with the adjusted return. For 5,173 of 5,175 days they agree, so these are real market moves. I keep them and flag them.
- Three TRV days (1976-10-28, 1978-12-20, 1981-08-10) are confirmed data errors: the price exactly doubles for one day, the volume is 0, and the price returns to its old level the next day. I replace them by interpolation between the neighbor days.
- Zero volume (75 rows) is kept and flagged. Zero volume can be real.

## Assumptions
- The supplied `ohlc` table is a trusted source for the adjustment factor, because its raw prices equal mine.
- Raw prices are already adjusted for splits. The adjustment factor then reflects dividends only.
- The report window is the 6 complete calendar months before the month of the latest cleaned date (2024-06-01 to 2024-11-30). I use one window for all tickers.

## How to read the reports and the summary
- Each report has a 2x2 figure and a 10-line summary table. Percentages are shown for display. All saved values are fractions.
- In the price chart, red means volatility above the mean for that report, green means at or below it, gray means the estimate is not yet available.
- `portfolio_summary.csv` is sorted by Sharpe ratio (highest first). Undefined Sharpe values are not ranked.
- **Coverage differs.** 22 of 30 tickers end on 2024-10-18 (97 trading days). 8 tickers end on 2024-11-29 (126 trading days). Do not compare total return or Sharpe ratio across these two groups without care. The columns `first_observation`, `last_observation`, and `trading_days` show each ticker's real coverage.

## Known doubts
- AAPL, 1990-10-19: the adjusted close equals the close on this one day, while the neighbor days have an adjusted close about 20% below the close. The supplied reference has the same value, so I could not check it. I kept the row.
- MSFT, 2004-11-15: the raw return is far below the adjusted return. This fits a large dividend, but I did not check it in an outside source.
- The `ohlc` reference may contain other errors that I cannot see.