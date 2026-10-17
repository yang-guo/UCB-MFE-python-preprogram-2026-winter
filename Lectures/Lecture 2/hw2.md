# Homework 2: Building a Scalable Stock Analysis Pipeline

**Due Date**: October 29, 2026 (two weeks after the final Data Analysis lecture on October 15).
**Submission**: Pull request to the course repository

## Overview

You're joining a quantitative trading desk as an analyst. Your first project is to build an automated stock analysis system that can:
1. Clean messy market data (it's always messy in the real world!)
2. Generate standardized risk reports for individual stocks
3. Scale the analysis across the entire portfolio

This homework will teach you production-level data cleaning, notebook parameterization, and automated reporting - skills you'll use daily in quant finance.

---

## Preparation and notebook map

The two data-analysis sessions are October 8 and October 15. The due date and three deliverables below are unchanged. The live notebooks provide worked examples and practice; the homework asks you to apply those patterns independently and add its specified metrics, styling, and broader batch scope.

- Pandas foundations: [October 8 module 1](2_pandas_basics.ipynb).
- Portfolio analysis: [October 8 module 2](3_pandas_to_portfolio.ipynb).
- Cleaning and plots (deferred from October 8): [October 15 module 1](../Lecture%204/1_visualize_and_clean.ipynb).
- SQL and quality checks: [October 15 module 2](../Lecture%204/2_sql_and_data_quality.ipynb).
- Report template: [October 15 module 3](../Lecture%204/3_stock_analysis_template.ipynb).
- Papermill and result collection: [October 15 module 4](../Lecture%204/4_running_analysis_at_scale.ipynb).

The course database lives at `Lectures/Lecture 4/data/data.db` (download from bCourses). In the assignment below, `data/data.db` means the database path relative to your own homework working directory; use a local working copy when writing cleaned tables. Never commit the database or credentials.

Sort by ticker/date before calculating returns or filling; fill within ticker groups, preserve repair flags, and recheck OHLC bounds afterwards. Forward filling alone can violate today's range. Quarantine unresolved leading gaps or invalid rows with a reason rather than borrowing a different ticker's value. For adjusted prices, preserve a consistent adjustment convention after any raw-price repair.

## The Data

You'll work with the `ohlc_hw` table in `data/data.db`. This table contains daily OHLC (Open, High, Low, Close) data for multiple stocks.

The columns are `ts` (observation date), `ticker`, raw `open`, `high`, `low`, `close`, vendor `adj_close`, and `volume`. The lecture's synthetic example calls its date column `date`; rename deliberately when reusing a function.

**Warning**: This data is intentionally dirty! Inspect missing values (including adjusted close), extreme values, ticker formatting, repeated records, and price inconsistencies. Missing cells and missing trading dates are different checks; a check can correctly find zero issues. Do not invent faults to match the checklist.

Normalize ticker markup, whitespace, and case **before** grouping or deduplicating. Remove exact duplicate records; investigate conflicting `(ticker, ts)` keys rather than arbitrarily keeping the first row. Preserve the original labels or an audit of their normalization.

Your first task is to clean it properly.

---

## Part 1: Data Cleaning & Validation

Create a Jupyter notebook called `01_data_cleaning.ipynb` that:

### 1.1 Load and Inspect the Data
- Load the `ohlc_hw` table from `data/data.db`
- Display basic statistics (`.info()`, `.describe()`)
- Identify the columns and their meanings
- Show the date range and list of tickers

### 1.2 Detect Data Quality Issues

Write functions to detect:

**Missing Data:**
- Identify missing values in each column
- Detect missing trading days (gaps in the time series)
- Create a summary: "Ticker X is missing Y trading days between [start] and [end]"

For this assignment, the supplied `ohlc` table's dates **for each ticker** are an acceptable reference calendar over that ticker's supplied history. Alternatively, document an exchange-calendar and active-history assumption. Business weekdays alone include exchange holidays. Report zero gaps when appropriate, and demonstrate your detector on a tiny example with one deliberately removed date. Detecting absent rows does not require inventing prices for them.

**Outliers:**
- Calculate daily returns: `(close - prev_close) / prev_close`
- Flag `abs(return - mean_return) > 3 * std_return` within each ticker, using the sample standard deviation (`ddof=1`); flag both tails. Exclude undefined returns and treat zero/undefined standard deviation explicitly. This retrospective screen is a review flag, not a deletion rule.
- Flag any day where: `high < close` or `low > close` (impossible!)
- Flag nonpositive or non-finite equity prices, including adjusted close; raw OHLC bounds do not apply to adjusted close

**Inconsistencies:**
- Verify that `low <= open <= high` and `low <= close <= high`
- Check for zero volume days (could indicate stale data)

Create a **Data Quality Report** DataFrame with columns:
```
ticker | issue_type | issue_date | description | severity (high/medium/low)
```

Allow multiple issues per observation. Include missing dates as separate issue rows, and explain your severity choices. Retain the diagnosis before repairs; rerun validation afterwards. Demonstrate the detector and repair behavior on small injected examples for required cases absent from your database.

### 1.3 Clean the Data

Implement cleaning functions:

**Handle Missing Values:**
- For missing prices: Use forward-fill (carry last known price)
- For missing volume: Fill with 0 and keep a repair flag (**unknown** volume is not evidence of no trading)
- Drop rows where ALL raw price columns (open, high, low, close) are missing **before** filling
- Missing `adj_close` needs a consistent adjustment convention. You may use matched `(ticker, ts)` rows in supplied `ohlc` to obtain a trusted date-specific adjustment factor, document the source, and reconstruct adjusted close from validated raw close. Do not forward-fill a factor across an unverified corporate action. Quarantine unresolved rows and report counts; do not replace the entire dirty dataset with the reference table.

**Handle Outliers:**
- For impossible price relationships (high < close, etc.):
  - Option 1: Replace with NaN then forward-fill
  - Option 2: Use the previous day's close
  - Option 3: Interpolate using other prices (e.g. if C is weird, use OHL)
  - Document your choice!
- For extreme returns (>3 std dev):
  - Investigate: Is it a stock split? Merger? Data error?
  - If data error: interpolate using neighboring days
  - If valid (e.g., major news): keep it but flag it

Interpolate only confirmed errors, within a sorted ticker and between valid neighbors; do not extrapolate leading/trailing gaps. Record evidence or a clearly stated uncertainty rather than claiming a split or news event without support. A documented quarantine is acceptable when repair is not defensible. Recheck all raw bounds and adjusted-price consistency after every repair.

**Validation:**
- Create a new table `ohlc_hw_cleaned` in the database
- Verify: No missing, non-finite, or nonpositive values in retained raw and adjusted price columns; unique `(ticker, ts)` keys; finite nonnegative volume
- Verify: All price relationships are valid
- Show before/after statistics

Account for retained, removed, and quarantined rows and save repair flags with the cleaned data. Keep the raw table unchanged. Store `ts` consistently as ISO dates (`YYYY-MM-DD`) so the report's SQL date filters work.

**Deliverable**:
- `01_data_cleaning.ipynb` with your analysis and code
- A saved `ohlc_hw_cleaned` table in `data/data.db`

---

## Part 2: Single Stock Analysis Notebook

Create a **parameterized** Jupyter notebook called `02_stock_report_template.ipynb` that generates a comprehensive risk report for a single stock.

### 2.1 Parameters

At the top of your notebook, define these parameters (using papermill-compatible cells):

```python
# Parameters
ticker = "AAPL"  # Stock ticker to analyze
start_date = "2024-01-01"  # Analysis start date
end_date = "2024-12-31"  # Analysis end date
```

**Tag this cell as "parameters"** in Jupyter (View → Cell Toolbar → Tags)

In JupyterLab, use the Property Inspector → Cell Tags. Verify the metadata tag is present; the `# Parameters` comment alone is insufficient. A `db_path` parameter is also useful for a portable local working copy.

### 2.2 Data Loading

- Load cleaned data for the specified ticker and date range
- Display summary: "{Ticker} from {start} to {end}: {N} trading days"
- Show the first and last few rows and actual first/last observation dates. Validate the input and report insufficient history clearly; 20 returns require at least 21 prices.

### 2.3 Calculate Risk Metrics

Implement functions to calculate:

**Daily Returns:**
```python
def calculate_returns(prices: pd.Series) -> pd.Series:
    """Calculate daily log returns"""
    # Hint: np.log(prices / prices.shift(1))
    pass
```

**Rolling Volatility (20-day window):**
```python
def calculate_volatility(returns: pd.Series, window: int = 20) -> pd.Series:
    """Calculate annualized rolling volatility"""
    # Hint: returns.rolling(window).std() * np.sqrt(252)
    pass
```

**Drawdown and Maximum Drawdown:**
```python
def calculate_max_drawdown(prices: pd.Series) -> pd.Series:
    """Return per-date drawdown from the running price peak."""
    # Hint:
    # 1. Calculate cumulative max price
    # 2. Drawdown = (price - cum_max) / cum_max
    pass
```

This function name is retained for the assignment, but the returned Series is **drawdown**, which can recover. Maximum drawdown for the summary is `drawdown.min()`; the running worst drawdown would be `drawdown.cummin()`.

**Sharpe Ratio:**
```python
def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    """Calculate annualized Sharpe ratio"""
    # Hint: (mean_return * 252 - rf) / (std_return * sqrt(252))
    pass
```

Apply these functions to your data and display the results.

Use **adjusted close** for report returns and drawdown. Use daily **log returns** for volatility, Sharpe, and daily-return summary statistics; Part 1's simple raw-close returns serve a different diagnostic purpose. The Sharpe hint defines the course approximation with a 2% annual risk-free rate. Return `NaN`/display `N/A` for zero volatility or insufficient returns. Keep fractions in calculations and CSVs; format percentages only for display.

### 2.4 Visualization

Create a **2x2 subplot grid** showing:

**Top Left - Price Chart:**
- Plot adjusted close prices
- Add a 50-day moving average
- Color the line by volatility (high vol = red, low vol = green)
  - Use mean rolling volatility over the report as the threshold; gray indicates unavailable warm-up estimates. Equal-to-mean can be green. This is retrospective coloring.
- Title: "{Ticker} Price and 50-day MA"

**Top Right - Daily Returns:**
- Bar chart of daily returns
- Color bars: green (positive), red (negative)
- Add horizontal line at 0
- Title: "{Ticker} Daily Returns"

**Bottom Left - Rolling Volatility:**
- Line chart of 20-day rolling volatility
- Add horizontal line at mean volatility
- Shade regions: high vol (>mean), low vol (<mean)
- Title: "{Ticker} Rolling Volatility (20-day)"

**Bottom Right - Drawdown:**
- Area chart of per-date drawdown (fill below 0)
- Color: red/pink gradient
- Mark the maximum drawdown point
- Title: "{Ticker} Maximum Drawdown"

**Requirements:**
- Figure size: 16x12 inches
- Add grid to all plots
- Label axes properly
- Use consistent color scheme
- Add a main title: "{Ticker} Risk Analysis Report - {start_date} to {end_date}"

Make it pretty :) 

Preserve rolling warm-up gaps; do not fill them with zero. A 50-day moving average is unavailable until 50 prices exist. If a report has fewer observations, show the available data and explain the missing average. Layered transparent red fills are sufficient for the gradient.

### 2.5 Summary Statistics Table

At the end of the notebook, create a summary table:

```
Metric                    Value
------------------------  --------
Total Return             +23.45%
Annualized Return        +18.32%
Annualized Volatility    21.45%
Sharpe Ratio             0.85
Maximum Drawdown         -15.67%
Best Day                 +5.23%
Worst Day                -4.89%
Positive Days            55.2%
Average Up Day           +1.23%
Average Down Day         -1.15%
```

Use `last_adjusted_close / first_adjusted_close - 1` for total return. With `n` nonmissing log returns, compounded annualized return is `exp(sum(log_returns) * 252 / n) - 1`; whole-period annualized volatility is `std(log_returns) * sqrt(252)`. It is distinct from the mean rolling volatility used in Part 3. Best/worst and up/down averages use log returns. Positive days are positive returns divided by **all nonmissing** returns, including zero days. An empty up/down subset displays `N/A`.

**Deliverable**:
- `02_stock_report_template.ipynb` with parameters cell tagged
- All functions documented with docstrings
- Beautiful visualizations

---

## Part 3: Automated Multi-Stock Analysis

Create a Python script called `03_run_all_stocks.py` that uses **papermill** to generate reports for all stocks.

### 3.1 Script Structure

Your script should:

1. Load the list of all tickers from `ohlc_hw_cleaned`
2. Determine the date range: **last 6 complete months** in the database
   - Anchor to the latest observation date in the database, not the computer clock. If that date is Dec 6, 2024: use June 1 - Nov 30, 2024.
   - For a reproducible rule, use the six calendar months **before the month containing the latest cleaned observation**, including when that observation falls on month-end. A final trading date alone does not certify a complete delivery.
3. Create an output directory: `reports/{YYYY-MM-DD}/`
   - This is the run date, independent of the historical analysis window.
4. For each ticker:
   - Use papermill to execute `02_stock_report_template.ipynb`
   - Pass parameters: ticker, start_date, end_date
   - Save output to: `reports/{date}/{ticker}_report.ipynb`
   - Catch and log any errors (some stocks might fail!)

### 3.2 Progress Tracking

Add a progress indicator:
```python
from tqdm import tqdm

for ticker in tqdm(tickers, desc="Generating reports"):
    # Run papermill
    ...
```

Handle errors gracefully:
- If a stock fails, log the error but continue
- At the end, print: "Successfully generated N/M reports"
- Save failed tickers to `failed_stocks.txt`

Use one common requested window for all tickers. Supplied histories end on different dates: disclose each report's actual coverage and explain comparisons with shorter histories. Do not extrapolate prices through trailing gaps. Apply error handling to every ticker and handle an all-failed run without pretending there is a summary.

### 3.3 Summary Dashboard

After generating all reports, create a **portfolio summary DataFrame**:

```python
summary_df = pd.DataFrame({
    'ticker': [...],
    'avg_daily_return': [...],  # Mean daily log return over period (fraction)
    'avg_volatility': [...],     # Mean of rolling volatility
    'max_drawdown': [...],       # Worst drawdown over entire period
    'sharpe_ratio': [...],       # Annualized Sharpe ratio
    'total_return': [...],       # last_adj_close / first_adj_close - 1
    'report_path': [...]         # Path to generated notebook
})
```

Sort by Sharpe ratio (descending) and display top 10 and bottom 10 stocks.

Collect named outputs from successful executed reports (the lecture uses Scrapbook) or explain an equivalent consistent approach. Never collect every notebook in the folder indiscriminately: failed and stale reports can exist. Exclude undefined Sharpe values from rankings; top and bottom groups may overlap with fewer than 20 successes.

Create a summary visualization:
- Scatter plot: X-axis = avg volatility, Y-axis = total return
- Color points by Sharpe ratio (colorbar)
- Annotate top 5 stocks by return
- Title: "Risk-Return Profile of Portfolio"

Save summary to: `reports/{date}/portfolio_summary.csv`

**Deliverable**:
- `03_run_all_stocks.py` with clean, documented code
- Generated reports in `reports/` directory
- Portfolio summary CSV

---

## Submission Requirements

### What to Submit

1. **Code Files:**
   - `01_data_cleaning.ipynb`
   - `02_stock_report_template.ipynb`
   - `03_run_all_stocks.py`
   - `requirements.txt` (all dependencies)

2. **Documentation:**
   - `README.md` explaining:
     - How to run your code
     - Key design decisions you made
     - How you handled outliers and why
     - Any assumptions you made
     - How to interpret the reports

3. **Sample Output:**
   - Include 2-3 example report notebooks in `reports/`
   - Include the portfolio summary CSV

Generate the full batch locally; commit only the requested 2–3 sample reports and summary CSV. Do **not** commit `data.db`, other database copies, or credentials. The cleaning notebook must recreate `ohlc_hw_cleaned` for the grader. Keep submitted code under your homework directory, with paths and run order documented in your README; do not modify the course's source data or report template.

### Evaluation Criteria

Your submission will be evaluated on:

- **Correctness**: Does the code work? Are calculations correct?
- **Code Quality**: Clean, readable, well-documented code
- **Data Cleaning**: Thorough identification and handling of issues
- **Presentation**: Clear visualizations and professional reports

This is a pass/no-pass assignment. To pass, you must demonstrate competency in all four areas above.

### Tips for Success

1. **Start Early**: Data cleaning is harder than you think!
2. **Test with One Stock First**: Get Part 2 working before scaling to all stocks
3. **Document Your Decisions**: Why did you handle outliers that way?
4. **Handle Errors**: Not all stocks will have clean data - handle failures gracefully
5. **Make it Professional**: Pretend you're presenting this to a portfolio manager

---

## Resources

### Required Libraries
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sqlite3
import papermill as pm
from datetime import datetime, timedelta
from tqdm import tqdm
```

### Papermill Basics
```python
# Execute a notebook with parameters
pm.execute_notebook(
    input_path='template.ipynb',
    output_path='output.ipynb',
    parameters={
        'ticker': 'AAPL',
        'start_date': '2023-01-01',
        'end_date': '2023-12-31'
    }
)
```

### Database Connection
```python
import sqlite3
conn = sqlite3.connect('data/data.db')
df = pd.read_sql_query("SELECT * FROM ohlc_hw", conn)
```

---

## Questions?

Post in slack. Good luck!

**Remember**: In quantitative finance, clean data is as important as clever models. Take the data cleaning seriously!
