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

- Cleaning and plots: [October 8 module 2](3_visualize_and_clean.ipynb).
- SQL and quality checks: [October 15 module 1](../Lecture%204/1_sql_and_data_quality.ipynb).
- Report template: [October 15 module 2](../Lecture%204/2_stock_analysis_template.ipynb).
- Papermill and result collection: [October 15 module 3](../Lecture%204/3_running_analysis_at_scale.ipynb).

The course database lives at `Lectures/Lecture 4/data/data.db` (download from bCourses). In the assignment below, `data/data.db` means the database path relative to your own homework working directory; use a local working copy when writing cleaned tables. Never commit the database or credentials.

Sort by ticker/date before calculating returns or filling; fill within ticker groups, preserve repair flags, and recheck OHLC bounds afterwards. Forward filling alone can violate today's range. Handle unresolved leading gaps explicitly rather than borrowing a different ticker's value. For adjusted prices, preserve a consistent adjustment convention after any raw-price repair.

## The Data

You'll work with the `ohlc_hw` table in `data/data.db`. This table contains daily OHLC (Open, High, Low, Close) data for multiple stocks.

**Warning**: This data is intentionally dirty! It contains:
- Missing values (gaps in trading days)
- Outliers (data entry errors, corporate actions not adjusted)
- Inconsistencies (you'll need to discover these!)

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

**Outliers:**
- Calculate daily returns: `(close - prev_close) / prev_close`
- Flag returns > 3 standard deviations as potential outliers
- Flag any day where: `high < close` or `low > close` (impossible!)
- Flag any day where: `close < 0` (negative prices don't exist)

**Inconsistencies:**
- Verify that `low <= open <= high` and `low <= close <= high`
- Check for zero volume days (could indicate stale data)

Create a **Data Quality Report** DataFrame with columns:
```
ticker | issue_type | issue_date | description | severity (high/medium/low)
```

### 1.3 Clean the Data

Implement cleaning functions:

**Handle Missing Values:**
- For missing prices: Use forward-fill (carry last known price)
- For missing volume: Fill with 0 (indicates no trading)
- Drop rows where ALL price columns (open, high, low, close) are missing

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

**Validation:**
- Create a new table `ohlc_hw_cleaned` in the database
- Verify: No missing values in price columns
- Verify: All price relationships are valid
- Show before/after statistics

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

### 2.2 Data Loading

- Load cleaned data for the specified ticker and date range
- Display summary: "{Ticker} from {start} to {end}: {N} trading days"
- Show the first and last few rows

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

**Maximum Drawdown (cumulative):**
```python
def calculate_max_drawdown(prices: pd.Series) -> pd.Series:
    """Calculate running maximum drawdown from peak"""
    # Hint:
    # 1. Calculate cumulative max price
    # 2. Drawdown = (price - cum_max) / cum_max
    pass
```

**Sharpe Ratio:**
```python
def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    """Calculate annualized Sharpe ratio"""
    # Hint: (mean_return * 252 - rf) / (std_return * sqrt(252))
    pass
```

Apply these functions to your data and display the results.

### 2.4 Visualization

Create a **2x2 subplot grid** showing:

**Top Left - Price Chart:**
- Plot adjusted close prices
- Add a 50-day moving average
- Color the line by volatility (high vol = red, low vol = green)
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
- Area chart of maximum drawdown (fill below 0)
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
   - Complete months only!
3. Create an output directory: `reports/{YYYY-MM-DD}/`
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

### 3.3 Summary Dashboard

After generating all reports, create a **portfolio summary DataFrame**:

```python
summary_df = pd.DataFrame({
    'ticker': [...],
    'avg_daily_return': [...],  # Mean daily return over period
    'avg_volatility': [...],     # Mean of rolling volatility
    'max_drawdown': [...],       # Worst drawdown over entire period
    'sharpe_ratio': [...],       # Annualized Sharpe ratio
    'total_return': [...],       # (last_close - first_close) / first_close
    'report_path': [...]         # Path to generated notebook
})
```

Sort by Sharpe ratio (descending) and display top 10 and bottom 10 stocks.

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
