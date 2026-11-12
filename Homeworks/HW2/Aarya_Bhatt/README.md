# HW2 - Stock Analysis Pipeline

Aarya Bhatt

This project cleans the dirty OHLC data in the course database, builds a risk report for each stock, and then runs
that report for every ticker and ranks them.

## Files

- `01_data_cleaning.ipynb` - cleans `ohlc_hw` and saves the result as `ohlc_hw_cleaned`
- `02_stock_report_template.ipynb` - papermill template that makes a risk report for one ticker
- `03_run_all_stocks.py` - runs the template for every ticker and builds the summary
- `requirements.txt` - dependencies
- `reports/` - a few sample reports and the summary CSV

## Setup

1. Put the course database at `data/data.db` (it is not committed). It needs the tables `ohlc_hw` and `ohlc`.
2. Install the packages: `python -m pip install -r requirements.txt`

Run everything from this homework folder, since all paths are relative.

## How to run

1. Run `01_data_cleaning.ipynb` top to bottom. This creates the `ohlc_hw_cleaned` table. The original tables are never changed.
2. (Optional) Open `02_stock_report_template.ipynb`, change the parameters in the first cell and run it to check one stock.
3. Run `python 03_run_all_stocks.py`.

The script saves everything in `reports/YYYY-MM-DD/`: a notebook for each stock, `portfolio_summary.csv`, and a
risk-return plot. Tickers that fail are written to `failed_stocks.txt`. I only committed 2-3 of the sample report
notebooks and the summary CSV, not the whole batch.

## Cleaning decisions

- Tickers had HTML tags, extra spaces and mixed case, so I stripped and upper-cased them (the original is kept in `raw_ticker`).
- Exact duplicate rows are dropped. Rows with the same ticker and date but different values are quarantined because I can't tell which one is right.
- Rows with no usable OHLC prices at all are quarantined.
- Bad prices (zero, negative, or impossible like high < close) are set to missing and forward-filled within the same ticker. High/low are then widened if needed so the bar still contains the open and close. If there is nothing earlier to fill from (start of a ticker's history), the row is quarantined.
- `adj_close` is rebuilt when it is missing or doesn't match the reference table, using `close * (ref adj_close / ref close)` for that exact ticker and date.
- Missing or negative volume becomes 0.
- Every repair has a `_repaired` flag, quarantined rows have a reason, and a set of assertions runs before the table is saved.
- The notebook also checks for missing trading days against the `ohlc` calendar. It includes a small synthetic example to show the check works.

### Outliers

I treated errors and big moves differently. Impossible values (negative prices, high below close, conflicting
duplicates) can't be real, so they are fixed or quarantined. Returns more than 3 standard deviations from a stock's
mean are only flagged and left in. Those moves are often real (earnings, news), and removing them would make the
volatility and drawdown look smaller than they actually were.

## Assumptions

- The `ohlc` reference table is correct for trading dates and adjustment factors.
- A tolerance of `rtol=1e-3, atol=0.02` is used when comparing `adj_close` to the expected value, so tiny rounding differences are not treated as errors.
- Forward-filling is fine for isolated bad prices.
- 252 trading days a year and a 2% risk-free rate.
- The window for the batch run is the last 6 complete calendar months in the data. A stock with a shorter history uses what it has, and each report shows its actual first and last dates.
- A stock needs at least 21 trading days, otherwise the 20-day volatility can't be computed and the report fails (and is logged).

## Reading the reports

All metrics use adjusted close.

- **Return**: daily log return. Total return is last price / first price - 1.
- **Volatility**: 20-day rolling standard deviation of returns, annualised with sqrt(252). 0.20 means about 20% a year.
- **Max drawdown**: the biggest drop from a previous peak, as a negative number (-0.13 = -13%).
- **Sharpe ratio**: (average return x 252 - 0.02) / (std x sqrt(252)). Higher is better, and a negative value means it did worse than the risk-free rate.

Each report has four plots: price with a 50-day moving average (coloured by whether volatility is above or below
average), daily returns, rolling volatility, and drawdown.

`portfolio_summary.csv` is sorted by Sharpe ratio and the numbers are fractions, not percentages. In the scatter plot,
stocks toward the top left gave more return for less volatility. Six months is a short window, so this only describes
that period. A stock with a high return can still have a bad Sharpe ratio or a big drawdown, so I compare all the
metrics together.
