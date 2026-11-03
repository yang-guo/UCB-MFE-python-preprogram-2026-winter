# Homework 2: Scalable Stock Analysis Pipeline

**Student:** Shravani Koli

## Overview

This project builds an automated stock analysis pipeline using Python, SQLite, Jupyter notebooks, Papermill, and Scrapbook.

The pipeline cleans historical stock-price data, calculates stock performance metrics, and automatically generates reports for 30 stocks.

## Project Files

- `01_data_cleaning.ipynb` — Cleans the stock-price data and creates the `ohlc_hw_cleaned` table in SQLite.
- `02_stock_report_template.ipynb` — Calculates performance metrics for a selected stock and date range.
- `03_run_all_stocks.py` — Automates report generation for all 30 stocks.
- `requirements.txt` — Lists the required Python packages.
- `reports/` — Contains generated stock reports, portfolio summary, and risk-return visualization.

## Data Cleaning

The data-cleaning notebook:

- Removes duplicate records.
- Cleans ticker symbols containing HTML formatting.
- Removes rows with missing adjusted closing prices.
- Removes three identified anomalous AAPL price records.
- Saves **349,874 cleaned records** to the `ohlc_hw_cleaned` SQLite table.

## Stock Analysis

The report template calculates daily logarithmic returns, 20-day annualized rolling volatility, cumulative returns, drawdown, annualized return, Sharpe ratio (using a 2% annual risk-free rate), and other daily performance statistics.

Papermill executes the notebook for each stock, and Scrapbook collects the calculated metrics.

## How to Run

Install the required dependencies:

```powershell
uv pip install -r requirements.txt
```

Place the course-provided SQLite database at `data/data.db`.

Run `01_data_cleaning.ipynb` first to create the cleaned database table.

Then run the automation script from the HW2 project folder:

```powershell
python 03_run_all_stocks.py
```

## Results

**Reporting period:** June 1, 2024 – November 30, 2024

- Stocks analyzed: **30**
- Successful reports: **30**
- Failed reports: **0**

The pipeline generates a dated folder under `reports/` containing individual stock-report notebooks, `portfolio_summary.csv`, `risk_return_scatter.png`, and `failed_stocks.txt`.

## Data

The SQLite database (`data.db`) and local virtual environment (`.venv`) are excluded from the GitHub submission because they are not needed in version