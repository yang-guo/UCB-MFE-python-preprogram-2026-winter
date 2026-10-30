# Homework 2

### 1

Run commands from this folder.

```bash
python -m pip install -r requirements.txt
```

Place the local database at `data/data.db`, then run `01_data_cleaning.ipynb`.
The notebook creates `ohlc_hw_cleaned` and preserves the raw table.
Missing raw prices are forward-filled within each ticker. Missing volume becomes
a flagged zero, meaning unknown volume. Adjusted close uses matched date-specific
`ohlc` factors. Unresolved rows are quarantined; unconfirmed extreme returns stay
flagged without interpolation.

### 2

Run `02_stock_report_template.ipynb`. Change the tagged parameters cell to choose
a ticker and date range. Metrics use adjusted-close log returns. Displayed
percentages are stored as numeric fractions; undefined values show `N/A`.

### 3

```bash
python 03_run_all_stocks.py
```

Use the `python3` kernel from the earlier notebooks. All tickers use the six
calendar months before the month containing the latest cleaned observation.
The folder date uses New York time, independently of the historical window.

Outputs are in `reports/YYYY-MM-DD/`: report notebooks, `portfolio_summary.csv`,
`portfolio_summary.png`, `failed_stocks.txt`, and the local `errors.log`.
Only successful reports from the current run enter the summary. Undefined Sharpe
ratios are excluded from rankings. Average volatility means average 20-day rolling
volatility. Check actual coverage before comparing stocks; shorter histories
are not extended.

The full batch is generated locally. AAPL, AMZN, and MSFT are the three report
samples included for submission; the other generated notebooks are gitignored.
