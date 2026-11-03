# HW2 — Yisu Wang

This homework cleans the stock data, produces a single-stock risk report, and runs that report across the available tickers.

## Run order

1. Run `01_data_cleaning.ipynb` to create `ohlc_hw_cleaned`.
2. Run `02_stock_report_template.ipynb` to inspect one stock. Its tagged `parameters` cell sets the ticker and date range.
3. From the repository root, run:
   `./.venv/bin/python Homeworks/HW2/Yisu_Wang/03_run_all_stocks.py`

Install the packages in `requirements.txt` in the Python environment used by the notebooks and script.

## Data

Download `data.db` from bCourses and place it at `Lectures/Lecture 4/data/data.db`. The cleaning notebook writes `ohlc_hw_cleaned` to this local database and leaves `ohlc_hw` unchanged. The database is not included in this submission.

## Cleaning and reports

The cleaning notebook removes exact duplicate rows, repairs missing or inconsistent adjusted closes using matched reference data, and estimates the five invalid opening prices from their same-day ranges. It reviews large returns separately from price errors and retains zero-volume observations.

Reports use adjusted close for prices and daily log returns for risk measures. The batch script requests one common six-month window, but stocks may have different actual observation dates. The CSV records each report's actual coverage.
