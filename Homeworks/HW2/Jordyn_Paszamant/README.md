# Homework 2: Building a Scalable Stock Analysis Pipeline

## Overview

This project cleans stock price data stored in SQLite, builds a reusable stock analysis notebook, and automatically generates reports for every ticker in the cleaned dataset. It also creates a portfolio summary and a risk–return scatter plot.

The completed batch processed **30 out of 30 stocks**. The latest observation in the cleaned database was **December 6, 2024**, so the six complete calendar months used for batch analysis were **June 1–November 30, 2024**.

## Project files

- `01_data_cleaning.ipynb` — examines and cleans stock price records and creates the `ohlc_hw_cleaned` table.
- `02_stock_report_template.ipynb` — parameterized report for an individual ticker, including price and risk charts and a ten-metric summary table.
- `03_run_all_stocks.py` — runs the report for every cleaned ticker using Papermill, gathers metrics with Scrapbook, and produces the portfolio summary and scatter plot.
- `requirements.txt` — Python dependencies.
- `reports/YYYY-MM-DD/` — generated reports and batch outputs (the folder name is the execution date).

The SQLite database is expected at `data/data.db`. **The database is not included in the submission**; place the course-provided database there before running the project.

## Setup

1. Use Python 3 and install the required packages:

   ```bash
   python -m pip install -r requirements.txt
   ```

2. Ensure a Jupyter kernel named `python3` is available:

   ```bash
   python -m jupyter kernelspec list
   ```

3. Place the database at `data/data.db` and run `01_data_cleaning.ipynb` to create or refresh `ohlc_hw_cleaned`.

4. Ensure that the first parameter cell in `02_stock_report_template.ipynb` is tagged `parameters`. The notebook must accept `ticker`, `start_date`, `end_date`, and `db_path`, and save the `report_metrics` dictionary using Scrapbook.

## Running the pipeline

From the project directory, run:

```bash
python 03_run_all_stocks.py
```

Alternatively, in a notebook in the project directory:

```python
%run 03_run_all_stocks.py
```

The script reads all distinct tickers from `ohlc_hw_cleaned` and finds the latest observation date. It then chooses the six **complete** calendar months preceding the month of that observation. For example, with a latest observation of December 6, 2024, the reporting window is June 1 through November 30, 2024.

Each ticker is processed independently. Errors are caught so that one failed report does not stop the batch; failed tickers are recorded in `failed_stocks.txt`.

## Analysis and metrics

Individual reports use adjusted closing prices to calculate daily logarithmic returns, 20-day annualized rolling volatility, drawdown, and the Sharpe ratio (using a 2% annual risk-free rate). Each report also includes a ten-row summary of total return, annualized return, annualized volatility, Sharpe ratio, maximum drawdown, best day, worst day, positive days, average up day, and average down day.

The portfolio summary contains these columns:

| Column | Description |
| --- | --- |
| `ticker` | Stock symbol |
| `avg_daily_return` | Mean daily log return |
| `avg_volatility` | Mean 20-day rolling annualized volatility |
| `max_drawdown` | Worst drawdown from a prior adjusted-close peak |
| `sharpe_ratio` | Annualized risk-adjusted return |
| `total_return` | Change in adjusted close over the selected observations |
| `report_path` | Path to the executed notebook |

The script prints the top and bottom ten stocks by Sharpe ratio, excluding undefined ratios from rankings. It also creates a scatter plot with average rolling volatility on the x-axis, total return on the y-axis, and color indicating the Sharpe ratio; the five highest-return stocks are annotated.

## Outputs

The script writes to `reports/YYYY-MM-DD/`:

- `<TICKER>_report.ipynb` — an executed notebook for each successfully processed ticker.
- `portfolio_summary.csv` — numerical metrics for all successful reports.
- `risk_return_scatter.png` — portfolio risk–return visualization.
- `failed_stocks.txt` — names of tickers that failed, if any; empty when all succeed.

In the completed run, **30/30 reports succeeded**. Among the reported results, IBM had the highest Sharpe ratio (approximately 3.22) and MRK had the lowest (approximately −1.93). These are historical sample results, not investment recommendations.

## Submission notes

Include the cleaning notebook, parameterized report template, automation script, `README.md`, `requirements.txt`, portfolio summary CSV, and **2–3 representative executed report notebooks**, as required by the assignment. Avoid committing the SQLite database or the full set of generated notebooks if the assignment limits the submission to samples.

## Reproducibility notes

- Run the script from the project root so its relative paths resolve correctly.
- The batch uses the latest date in the cleaned database rather than the current calendar date.
- If your database is stored elsewhere, update `db_path` in `03_run_all_stocks.py` and the notebook parameters accordingly.
- A different Python environment may require changing `kernel_name="python3"` to the name of an installed Jupyter kernel.
