# Homework 2: Scalable Stock Analysis Pipeline

Author: Jiajing Wang

## 1. How to Run the Code

Use Python 3.10 or later. Open a terminal in this homework directory and install the dependencies in a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Place a local copy of the supplied database at `data/data.db`. In VS Code, select `.venv/bin/python` for the script and the corresponding environment for the notebooks.

Run the files in this order:

1. Run all cells in `01_data_cleaning.ipynb` to create `ohlc_hw_cleaned`.
2. Set `ticker`, `start_date`, and `end_date` in the tagged `parameters` cell of `02_stock_report_template.ipynb`, then run all cells to review one stock.
3. Generate reports for all stocks:

```bash
python 03_run_all_stocks.py
```

Reports, `portfolio_summary.csv`, the portfolio chart, and failure records are saved under `reports/{run_date}/`.

## 2. Key Design Decisions

- Normalize ticker labels before grouping and deduplication; preserve the original labels for comparison.
- Remove exact duplicates and quarantine unresolved conflicting ticker/date records with reasons.
- Sort by ticker and date before calculations. Forward-fill missing raw prices within each ticker; drop all-missing raw OHLC rows before filling. Fill missing volume with zero and keep a repair flag.
- Reconstruct adjusted close using validated raw close and the supplied `ohlc` adjustment factor matched to the same ticker/date. Preserve repair flags and validate the retained data before saving; keep the raw tables unchanged.
- Reuse the parameterized report with Papermill. Collect each successful execution's metrics CSV to build the portfolio summary; exclude failed or stale results.

## 3. How Outliers Are Handled and Why

- Flag raw-close simple returns when `abs(return - mean) > 3 * std` within each ticker, using sample standard deviation. Retain uncertain extremes with a review flag because a large return alone does not establish a data error.
- Repair invalid open/close values with the previous validated close only when it fits the current day's high/low range. Quarantine unresolved prices or leading gaps rather than inventing values.
- Recheck price validity, OHLC bounds, and adjusted-price consistency after repairs. Do not claim a split or news event without supporting evidence.

## 4. Assumptions

- The supplied `ohlc` table provides each ticker's reference trading dates and matched adjustment factors. Weekends and exchange holidays are not treated as missing sessions merely because they are absent.
- Report returns and drawdown use adjusted close. Volatility, Sharpe, and daily-return statistics use log returns, 252 trading days per year, and a 2% annual risk-free rate for the course Sharpe approximation.
- Batch analysis uses the six calendar months before the month containing the latest cleaned observation, even at month-end. The folder date is the execution date in Asia/Shanghai.
- Stock histories may cover different actual dates within the common requested window. Actual coverage is disclosed in each report; trailing prices are not extrapolated.

## 5. How to Interpret the Reports

- **Price chart:** adjusted close and the 50-day moving average. Red/green segments indicate volatility above/at or below the report's mean rolling volatility; gray indicates unavailable estimates.
- **Daily returns:** green bars show nonnegative log returns; red bars show negative returns.
- **Rolling volatility:** the annualized standard deviation of the latest 20 daily log returns. The first estimate needs 21 prices; the moving average needs 50 prices. Warm-up gaps remain missing.
- **Drawdown:** the decline from the running price peak. The most negative value is maximum drawdown.
- **Summary table:** total/annualized return, annualized volatility, Sharpe, maximum drawdown, and daily-return statistics. Undefined statistics display `N/A`.
- **Portfolio summary:** stocks are sorted by descending Sharpe; undefined Sharpe values are excluded from rankings. The scatter plot compares mean rolling volatility with total return, colors points by Sharpe, and labels the five highest returns. Compare actual coverage when interpreting stocks with shorter histories.

CSV returns, volatilities, and drawdowns are fractions: `0.12` means 12%. Percentages are formatted for display only. Whole-period annualized volatility differs from the mean rolling volatility in the portfolio summary.
