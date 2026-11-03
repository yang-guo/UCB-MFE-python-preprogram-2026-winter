# HW2: Scalable Stock Analysis Pipeline

## How to run the code

To execute the pipeline, run the files in the following order:

1. **`01_data_cleaning.ipynb`**: Run all cells. This notebook connects to `data/data.db`, extracts the raw OHLC data from the `signals` table, performs data quality checks (missing dates, invalid prices, outliers), cleans the data, and saves the resulting table as `ohlc_hw_cleaned` in the SQLite database.
2. **`02_stock_report_template.ipynb`**: This is a Papermill-parameterized template. Do not run it manually for the whole portfolio, but you can execute it cell-by-cell for a single stock (e.g., AAPL) to test the risk metrics and visualizations.
3. **`03_run_all_stocks.py`**: Execute this script from the terminal (`python 03_run_all_stocks.py`). It will detect the last 6 complete calendar months in the dataset, use Papermill to generate individual Jupyter Notebook reports for all stocks, and compile the final `portfolio_summary.csv` and a scatter plot dashboard.

## Key Design Decisions & Assumptions

*   **Database Mapping**: The assignment mentioned an `ohlc_hw` table, but the provided database contained the `signals` table. I used a SQL query with `AS` to map `symbol` to `ticker` and `date` to `ts` directly upon loading.
*   **Execution Speed**: Generating 5,400+ reports sequentially via Papermill takes nearly 3 hours. To make the pipeline truly scalable, I implemented `concurrent.futures.ThreadPoolExecutor` with 4 workers, reducing the total batch time to roughly 15-20 minutes while still capturing expected failures.
*   **Date Window**: The script dynamically determines the trailing 6-month window based on the maximum date found in the database, avoiding hardcoded dates.

## Outlier Handling Strategy

*   **Missing Prices**: Filled using forward-fill (`ffill`), followed by backward-fill (`bfill`) for leading missing values.
*   **Invalid Prices (non-positive or bound violations)**: Set to `NaN` and repaired using the same `ffill`/`bfill` logic to ensure continuity without injecting arbitrary external data.
*   **Volume**: Missing volume was filled with `0`.
*   **Unresolvable Tickers**: Tickers with completely broken price histories (e.g., all 0s) that could not be repaired via filling were dropped to prevent downstream division-by-zero errors in return calculations.

## Interpreting the Reports

*   **Individual Reports**: Provide a 2x2 grid containing price history with a 50-day MA, daily log returns, 20-day annualized rolling volatility, and maximum drawdown. The summary table displays annualized return and Sharpe ratio (assuming a 2% risk-free rate).
*   **Portfolio Dashboard**: A scatter plot maps total return against average volatility, colored by Sharpe ratio, allowing instant visual identification of the best-performing assets on a risk-adjusted basis.
