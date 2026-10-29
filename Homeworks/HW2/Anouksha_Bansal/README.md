# Homework 2

## Project Overview

This project implements a stock data cleaning and automated risk analysis pipeline using Python and SQLite.

The project consists of three main components:

1. **Data Cleaning:** Identifies and handles missing values, missing trading days, outliers, and inconsistent stock prices.
2. **Single Stock Analysis:** Generates risk reports for individual stocks using historical adjusted closing prices.
3. **Automated Portfolio Analysis:** Executes individual stock reports across multiple tickers and produces a portfolio-level risk-return summary.

## Project Structure

- `01_data_cleaning.ipynb`: Data quality assessment, cleaning, validation, and creation of the cleaned SQLite table.
- `02_stock_report_template.ipynb`: Parameterized single-stock analysis and risk report.
- `03_run_all_stocks.py`: Automated execution of individual stock reports and portfolio summary generation.
- `requirements.txt`: Python dependencies.
- `data/`: Local SQLite database.
- `reports/`: Generated individual stock reports and portfolio summaries.

## Setup and Dependencies

Python 3.11 is used for this project.

Install the required dependencies using:

```bash
python -m pip install -r requirements.txt
```

The project uses a local SQLite database located at:

`data/data.db`

The database must contain the original `ohlc_hw` table supplied with the assignment.

## How to Run the Project

Run the notebooks and script in the following order.

### Step 1: Data Cleaning

Open and execute:

`01_data_cleaning.ipynb`

This notebook:
- Inspects missing values and data inconsistencies.
- Identifies missing trading days and extreme daily returns.
- Repairs eligible missing or invalid observations.
- Validates the cleaned dataset.
- Saves the cleaned data as `ohlc_hw_cleaned` in the SQLite database.

### Step 2: Single Stock Analysis

Open:

`02_stock_report_template.ipynb`

Set the desired ticker, start date, end date, and database path in the parameters cell.

Execute the notebook to generate an individual stock risk report.

### Step 3: Automated Portfolio Analysis

Run the following command from the homework directory:

```bash
./.venv/bin/python 03_run_all_stocks.py
```

The script:
- Retrieves all available tickers from `ohlc_hw_cleaned`.
- Determines a common six-calendar-month analysis window based on the latest available observation.
- Executes the single-stock report notebook for each ticker using Papermill.
- Tracks progress and logs errors without stopping the entire batch.
- Saves failed ticker symbols.
- Creates a portfolio summary CSV and risk-return visualization.

Generated outputs are saved in a date-specific folder under `reports/`.

## Key Design Decisions

### Data Cleaning and Missing Values

The original database table is preserved. Cleaning is performed on a separate working dataset, which is saved as `ohlc_hw_cleaned`.

Missing raw OHLC prices are repaired using forward-filling where appropriate. Rows with all raw OHLC prices missing are excluded before filling. Missing volume values are filled with zero and marked with repair flags.

Adjusted closing prices are reconstructed only when a trusted adjustment factor is available for the matching ticker and date.

### Outlier Detection

Daily returns are calculated separately for each ticker. Observations are flagged when their returns are more than three sample standard deviations from the ticker's mean daily return.

Both unusually high and unusually low returns are considered.

Additional validation checks identify:
- Nonpositive or nonfinite prices.
- High prices below closing prices.
- Low prices above closing prices.
- Violations of OHLC price relationships.

Extreme returns are flagged for investigation rather than automatically removed, since unusually large price movements may represent legitimate market events.

### Assumptions

- The cleaned dataset is the source of data for all subsequent analysis.
- The analysis window is common across tickers and is determined using the latest available observation in the dataset.
- Stocks may have different actual historical coverage within the requested window.
- No prices are extrapolated beyond a stock's available observations.
- Daily returns are calculated using adjusted closing prices.
- Annualized volatility uses 252 trading days.
- The Sharpe ratio uses an assumed annual risk-free rate of 2%.
- Rolling volatility is calculated over a 20-trading-day window.

## Interpreting the Reports

Each individual stock report includes:
- Daily log returns.
- Rolling annualized volatility.
- Drawdown analysis.
- Sharpe ratio.
- Total and annualized returns.
- Daily return statistics.
- Risk visualizations.

The portfolio summary includes average daily returns, average rolling volatility, maximum drawdown, Sharpe ratio, and total return for each successfully analyzed ticker.

Stocks are ranked by Sharpe ratio. Undefined Sharpe ratios are excluded from rankings.

The risk-return visualization plots average rolling volatility against total return, with point color representing the Sharpe ratio.

Comparisons should be interpreted carefully when stocks have different actual data coverage, since their metrics may reflect different numbers of observations and market conditions.

## Sample Outputs

The `reports/` directory includes:
- Two to three representative individual stock report notebooks.
- The portfolio summary CSV.
- Other generated outputs may be retained locally but are not required for submission.

