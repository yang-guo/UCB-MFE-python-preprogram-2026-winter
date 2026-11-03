# Stock Data Cleaning and Risk Analysis

## Overview

This project builds a reproducible stock-data cleaning and reporting pipeline using Python, pandas, SQLite, and Jupyter notebooks. It cleans historical OHLCV (Open, High, Low, Close, Volume) data, validates adjusted closing prices, calculates stock performance and risk metrics, and generates reports for multiple tickers.

The pipeline consists of three parts:

1. **Data cleaning:** Identify and handle duplicate records, missing adjusted closing prices, invalid price observations, and data-quality issues. Preserve quarantined records for auditing and save the cleaned dataset to SQLite.
2. **Stock report:** Analyze a selected ticker over a configurable date range, calculate performance and risk metrics, and visualize price, returns, rolling volatility, and drawdown.
3. **Batch reporting:** Execute the report notebook for all available tickers, record successes and failures, and generate a portfolio-level summary and risk-return visualization.

## Project Structure

```text
Vanditha_Toleti/
├── 01_data_cleaning.ipynb
├── 02_stock_report_template.ipynb
├── 03_run_all_stocks.py
├── requirements.txt
├── README.md
└── reports/
    └── YYYY-MM-DD/
        ├── run_results.csv
        ├── portfolio_summary.csv
        ├── top_10_sharpe.csv
        ├── bottom_10_sharpe.csv
        ├── risk_return_profile.png
        └── *_report.ipynb
```

The database is expected at `course_file_export/data.db` relative to the repository root. Do not commit the database or other large source-data copies unless explicitly required.

## Setup

Run the following commands from the repository root.

### 1. Create and activate a virtual environment

**macOS/Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r Homeworks/HW2/Vanditha_Toleti/requirements.txt
```

### 3. Register the Jupyter kernel

```bash
python -m ipykernel install --user --name ucb-mfe-hw2 --display-name "Python (UCB MFE HW2)"
```

Open the notebooks in VS Code and select the project environment or the `Python (UCB MFE HW2)` kernel.

## How to Run

### Part 1: Clean the data

Open and run all cells in:

`Homeworks/HW2/Vanditha_Toleti/01_data_cleaning.ipynb`

This notebook reads the source data, performs cleaning and validation, and writes the cleaned and quarantined datasets to SQLite. Run this notebook before generating reports if the cleaned tables do not already exist or need to be rebuilt.

Expected tables:

- `ohlc_hw_cleaned`: cleaned observations used for analysis.
- `ohlc_hw_quarantine`: records excluded from the main cleaned dataset and retained for review.

The original source tables should remain unchanged.

### Part 2: Generate a report for one ticker

Open:

`Homeworks/HW2/Vanditha_Toleti/02_stock_report_template.ipynb`

Edit the parameter cell as needed:

```python
ticker = "AAPL"
start_date = "2024-06-01"
end_date = "2024-11-30"
rolling_window = 21
risk_free_rate = 0.0
```

Run all cells. The notebook loads the selected ticker's cleaned data, calculates the metrics, performs validation checks, and displays the risk dashboard.

The actual data-coverage dates may differ from the requested dates because observations are available only on trading days and depend on the source dataset.

### Part 3: Run reports for all tickers

From the repository root, run:

```bash
python Homeworks/HW2/Vanditha_Toleti/03_run_all_stocks.py
```

The script discovers tickers from `ohlc_hw_cleaned`, executes the report notebook for each ticker, records execution results, and generates the portfolio summary and visualization.

## Metrics

The analysis uses adjusted closing prices.

- **Daily log return:** Natural logarithm of the ratio of consecutive adjusted closing prices.
- **Cumulative return:** Final adjusted closing price divided by the initial adjusted closing price, minus one.
- **Annualized volatility:** Standard deviation of daily log returns multiplied by the square root of 252.
- **Annualized Sharpe ratio:** Annualized mean daily log return minus the annual risk-free rate, divided by annualized volatility. The daily risk-free adjustment is approximated using the annual rate divided by 252.
- **Maximum drawdown:** The minimum percentage decline from a running peak in adjusted closing prices.
- **Rolling volatility:** Rolling standard deviation of daily log returns, annualized using the square root of 252.

A 21-observation rolling window is used by default. Metrics may be undefined when insufficient observations or zero volatility prevent their calculation.

## Data Quality and Validation

The cleaning workflow checks for duplicate ticker-date observations, missing values, invalid prices, OHLC consistency, volume validity, adjusted-close inconsistencies, and extreme returns.

Exact duplicate records are removed. Missing or inconsistent adjusted closing prices are checked against the reference source where possible. Records that cannot be safely repaired are retained in the quarantine table rather than silently discarded. Extreme returns and unusual volume observations are flagged for investigation rather than automatically removed solely because they are extreme.

The report notebook validates its output before presenting the results. The batch runner records execution status and error information for individual tickers.

## Outputs

Each batch run creates a date-stamped directory under:

`Homeworks/HW2/Vanditha_Toleti/reports/`

Expected outputs include:

- `run_results.csv`: execution status for each ticker.
- `portfolio_summary.csv`: portfolio metrics for successfully processed tickers.
- `top_10_sharpe.csv`: highest Sharpe ratios among defined values.
- `bottom_10_sharpe.csv`: lowest Sharpe ratios among defined values.
- `risk_return_profile.png`: scatter plot of return against volatility, colored by Sharpe ratio.
- `*_report.ipynb`: executed report notebooks for individual tickers.

The portfolio summary contains the ticker, average daily log return, average rolling volatility, maximum drawdown, annualized Sharpe ratio, total return, and report path.

## Reproducibility Notes

- Run the cleaning notebook before the report pipeline when rebuilding the cleaned tables.
- Use the same requested date window when comparing tickers.
- Check actual coverage dates and observation counts when interpreting results.
- Review `run_results.csv` for failed executions.
- Keep source data and local database files out of version control.
- Results depend on the input dataset, cleaning decisions, selected date window, rolling window, and risk-free rate.

## Limitations

Historical returns and risk statistics describe the selected sample period and do not guarantee future performance. Differences in trading calendars, missing observations, and sample length can affect comparisons across tickers. Extreme-return flags indicate observations requiring review; they are not, by themselves, evidence of data errors.
