# HW2 — Stock Risk Analysis Pipeline

## Run the analysis

1. Download the course SQLite database from bCourses and place a working copy at `data/data.db` inside the HW2 folder. Keep the database and any credentials local.
2. From the HW2 folder, launch JupyterLab with the required packages:

   ```bash
   uv run --no-project --with-requirements requirements.txt jupyter lab
   ```

3. Open `01_data_cleaning.ipynb` and run all cells. This creates `ohlc_hw_cleaned` in the working database. To preview a single-stock report, run `02_stock_report_template.ipynb` with its default AAPL parameters.
4. Open a terminal in the HW2 folder and run the batch:

   ```bash
   uv run --no-project --with-requirements requirements.txt python 03_run_all_stocks.py
   ```

The batch writes the executed notebooks, `portfolio_summary.csv`, `portfolio_risk_return.png`, and `failed_stocks.txt` under `reports/YYYY-MM-DD/`. The folder date is the date the script ran. The analysis window is the six calendar months before the month of the latest cleaned observation. For this dataset, the requested window was June 1 through November 30, 2024.

## Cleaning decisions

I standardized ticker labels before checking duplicates and removed 29,460 extra copies. I used each ticker’s dates in the supplied `ohlc` table to check trading-day coverage. I treated zero opens as missing and quarantined five first observations because there was no earlier open for the same ticker. I used matching ticker/date adjustment factors from `ohlc` to repair 28,791 adjusted closes. All 5,056 flagged raw-close returns matched the reference returns, so I kept those rows with their flags. The cleaned table contains 368,246 validated rows.

## Read the reports

The reports use adjusted closing prices for daily log returns and drawdowns. The Sharpe ratio uses 252 trading days and a 2% annual risk-free rate. Gray price segments indicate that rolling volatility is still unavailable. The portfolio chart compares total return with mean annualized 20-day rolling volatility, and its color scale shows the Sharpe ratio.

Each notebook publishes a named JSON output, `portfolio_metrics`, which the batch script reads from successful runs. The CSV includes actual observation dates and trading-day counts. Some stocks have shorter histories within the requested window, so compare those dates before interpreting differences in total return.
