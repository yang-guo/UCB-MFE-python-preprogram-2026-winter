# HW2 — Scalable Stock Analysis Pipeline

**Ananya Das**

## Overview

An end-to-end stock analysis pipeline covering data cleaning, individual stock risk analysis, and portfolio-level aggregation.

The project consists of three stages:

1. **Data Cleaning:** Diagnose and clean raw OHLC data while maintaining a complete audit trail.
2. **Stock-Level Analysis:** Generate parameterized risk reports for individual tickers using Papermill.
3. **Portfolio Analysis:** Automate report generation across all tickers and produce a consolidated risk-return summary.

## Project Structure

| File | Description |
|---|---|
| `01_data_cleaning.ipynb` | Data cleaning, diagnosis, and validation |
| `02_stock_report_template.ipynb` | Parameterized stock-level risk report |
| `03_run_all_stocks.py` | Batch execution and portfolio summary |
| `hw2_cleaning.py`, `hw2_report.py` | Reusable data processing and analytics functions |
| `tests/` | Unit and integration tests |
| `reports/` | Generated reports and portfolio outputs |

## Part 1 — Data Cleaning

The raw dataset contains 397,711 observations across 30 tickers, with duplicate records, missing values, inconsistent labels, and corrupted prices.

Key implementation decisions:

- Standardized ticker labels before grouping and processing.
- Removed 29,460 exact duplicates while preserving an audit trail.
- Reconstructed missing and corrupted adjusted closing prices using date-matched reference adjustment factors.
- Quarantined observations that could not be reliably repaired.
- Persisted cleaning diagnostics, repaired cells, removed rows, and ticker-level audits in separate database tables.

All 397,711 source observations are accounted for, and the original source tables remain unchanged.

## Part 2 — Stock-Level Risk Reports

A Papermill-parameterized notebook generates reports for individual tickers over a specified date range.

Using adjusted closing prices, the analysis includes:

- Daily log returns and annualized returns.
- 20-day rolling annualized volatility.
- Maximum drawdown.
- Annualized Sharpe ratio (2% risk-free rate).
- A ten-metric performance summary and a 2×2 visualization of price, returns, volatility, and drawdown.

The calculations are implemented as reusable, documented functions in `hw2_report.py`. Undefined metrics are handled explicitly, and reports with insufficient history are rejected.

## Part 3 — Batch Execution and Portfolio Analysis

The batch runner executes the report template for all tickers in the cleaned dataset, using Papermill and tqdm for execution and progress tracking.

- The analysis window is the six complete calendar months preceding the month of the latest observation. For this dataset, the window is **June 1 to November 30, 2024**.
- Individual ticker failures are logged without interrupting the batch.
- Only successfully generated reports are included in the portfolio summary.
- `portfolio_summary.csv` contains average daily log return, average rolling volatility, maximum drawdown, Sharpe ratio, and total return, along with data coverage information.
- A risk-return scatter plot compares volatility against total return, with Sharpe ratio represented by color.

## Outlier Treatment and Assumptions

Statistical outlier flags are treated as review indicators rather than automatic deletion criteria. Confirmed one-day price errors are repaired; plausible market movements and unresolved observations are retained with flags.

The analysis assumes 252 trading days per year and uses a 2% annual risk-free rate. Rolling volatility is calculated over 20 trading days. Metrics are calculated only over available observations, without extrapolating missing history.

## Results and Interpretation

The batch successfully generated reports for all 30 tickers. However, 22 tickers have shorter historical coverage, ending on October 18, 2024, while eight cover the full requested window. Consequently, returns and drawdowns should be interpreted in the context of each ticker's actual coverage.

The pipeline produces individual stock reports, a consolidated portfolio summary, a risk-return visualization, and execution/failure logs.

## Reproducibility

The project uses Python 3.12, with dependencies specified in `requirements.txt`. Cleaning and analytics logic are separated into reusable modules, and the test suite covers individual functions and batch execution.

The input database is supplied separately and is not included in the submission.