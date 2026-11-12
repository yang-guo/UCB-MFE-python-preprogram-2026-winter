# HW2: Stock Analysis Pipeline

This submission contains the three HW2 deliverables: a data-cleaning notebook, a parameterized single-stock report, and a script that runs reports for the full ticker list. The pipeline cleans `ohlc_hw`, calculates risk metrics from adjusted closing prices, and creates individual reports plus a portfolio summary.

## Setup and run order

Run commands from the repository root. Activate the course environment and install this submission's dependencies:

```bash
source .venv/bin/activate
uv pip install -r Homeworks/HW2/Josie_Wu/requirements.txt
jupyter lab
```

Place a local copy of the course database at `Homeworks/HW2/Josie_Wu/data/data.db`. The database is input data and must not be committed.

1. In JupyterLab, open and run `Homeworks/HW2/Josie_Wu/01_data_cleaning.ipynb` from top to bottom. It reads `ohlc_hw` and the supplied `ohlc` reference table, writes `ohlc_hw_cleaned` and `ohlc_hw_quarantine` to the local database, and displays before/after validation. Rerun this notebook whenever starting with a fresh database copy.
2. Open `Homeworks/HW2/Josie_Wu/02_stock_report_template.ipynb`. Edit the tagged `parameters` cell to choose `ticker`, `start_date`, and `end_date`, then run the notebook. It displays the actual dates and number of observations available for that ticker.
3. From the repository root, generate reports and the portfolio summary for all tickers:

   ```bash
   python Homeworks/HW2/Josie_Wu/03_run_all_stocks.py
   ```

   Outputs are written to `Homeworks/HW2/Josie_Wu/reports/YYYY-MM-DD/`. The script uses the six complete calendar months before the month of the latest cleaned observation; the folder date is the run date. It continues after individual failures and records them in `failed_stocks.txt`.

## Cleaning decisions and assumptions

- Ticker HTML and formatting are normalized before duplicate checks; original labels are retained for audit. Exact duplicate rows are removed. Conflicting ticker/date keys are investigated instead of arbitrarily keeping a row.
- Missing trading dates are checked against the supplied `ohlc` history for each ticker, not weekdays alone (weekdays include market holidays). A small example with a deliberately removed date demonstrates the detector; the supplied histories have no remaining date gaps under this reference.
- The pre-repair data-quality report has one row per issue and the required `ticker`, `issue_type`, `issue_date`, `description`, and `severity` fields. High severity means a row cannot be trusted as-is; medium marks missing data or statistical flags for review; low is informational (for example, zero volume or exact duplicates). Multiple issues on one observation can produce multiple report rows.
- Data is sorted by ticker and date before return calculations or within-ticker filling. Missing `adj_close` values, and values inconsistent with the matched reference adjustment factor, are reconstructed using the same ticker and date from `ohlc`: validated raw close multiplied by that date's reference adjustment factor. No factor is carried across dates. The notebook records repair flags.
- Five rows have an open price of zero and violate valid OHLC bounds. The reference does not provide a defensible replacement, so these rows are retained in `ohlc_hw_quarantine` with reasons rather than guessed into the cleaned table. The cleaned table is checked for unique ticker/date keys, valid positive prices, valid OHLC bounds, and finite nonnegative volume.
- Daily simple returns from raw close are screened within each ticker using the sample standard deviation (`ddof=1`); both tails beyond three standard deviations are flagged. The largest flagged moves are reviewed against the available reference data. Because the reference repeats the same market history and cannot establish that these moves are errors, outlier flags are retained for review and are not an automatic deletion or interpolation rule. Missing-date and repair behavior are also demonstrated with small injected examples where the supplied data has no such cases.
- Zero volume is reported as a possible stale-data signal, not by itself treated as proof that a price is wrong. Unknown volume is filled with zero only when repairing a missing volume value, with a repair flag.

## Reading the reports

The single-stock report uses adjusted close for total return and drawdown, and daily log returns for volatility, Sharpe ratio, and daily-return statistics. Volatility and Sharpe are annualized using 252 trading days; Sharpe uses a 2% annual risk-free rate. Rolling volatility uses 20 returns and the price chart includes a 50-observation moving average, so initial warm-up values are unavailable. The report shows actual data coverage because ticker histories can differ.

The portfolio CSV reports average daily log return, average 20-day annualized volatility, maximum drawdown, Sharpe ratio, total return, report path, and actual coverage. The scatter plot compares average volatility with total return and colors points by Sharpe ratio. Rankings omit undefined Sharpe values; comparisons should account for each ticker's actual coverage.

## Submitted sample outputs

The `reports/` folder includes three example report notebooks (AAPL, NVDA, and TRV), the generated portfolio summary CSV, and the risk-return plot. The full 30-ticker batch was run locally; generated reports beyond these examples are excluded from version control. `failed_stocks.txt` records per-ticker failures for each run.

The database, virtual environments, and generated reports beyond the submitted examples are local files and are not part of the submission.
