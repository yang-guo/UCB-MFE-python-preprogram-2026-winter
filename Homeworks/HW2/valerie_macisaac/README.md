
# Homework 2: Building a Scalable Stock Analysis Pipeline README


## Overview
This project begins by implementing a data cleaning protocol on 30 equity tickets stored in data.db. It then outputs risk and return statistics as well as a summary table for all stocks through a papermill pipeline.



1. **`01_data_cleaning.ipynb`**: Detects data quality anomalies and issues in historical equity data spanning 1962-01-02 to 2024-12-06. Applies repairs to some of the data and flags any remaining issues. Writes the cleaned dataset to a new table in data.db (`ohlc_hw_cleaned`).
2. **`02_stock_report_template.ipynb`**: A report template designed to be applied on all stocks through `papermill`. It computes key statistics and makes a visualization for one stock (starter stock was AAPL). It outputs a formatted summary table, and embeds statistics for the overall csv using `scrapbook`.
3. **`03_run_all_stocks.ipynb`**: A notebook meant to run notebook 2 on all stocks. It determines the last 6-month window and executes the report template across all 30 tickers with progress tracking (`tqdm`). It logs any failures and logs an overall manifest for all stocks. Final output is an aggregated csv to display all stocks and a risk-return scatter plot.


---

## How to Run the Code


1. Ensure data.db is located in the valerie_macisaac folder, then run the notebooks in order:
- Step 1 — Data Cleaning **`01_data_cleaning.ipynb`**
Run all cells from top to bottom. This processes ohlc_hw and creates/updates the ohlc_hw_cleaned table inside data.db.
- Step 2 — Single-Stock Template **`02_stock_report_template.ipynb`**:
Run all cells from top to bottom, will demonstrate the report_template for AAPL and create calculations and visuals.
- Step 3 — Run all the Stocks in the Portfolio **`03_run_all_stocks.ipynb`**:
Run all cells from top to bottom. The script automatically creates an output directory (reports/{YYYY-MM-DD}/) containing: individual report notebooks ({TICKER}_report.ipynb), portfolio_summary.csv, risk_return.png, manifest.json and failed_stocks.txt

---


## Key Decisions Made
1. Anchoring the six month window to the six months before the last month in the database, ie. in ohlc_hw_cleaned (2024-12-06). It snaps to the first day of that month (2024-12-01) and selects the six complete preceding calendar months (2024-06-01 to 2024-11-30), ensuring every stock is evaluated over a standardized six month window.
2. I decided to get rid of the HTML strings in the ticker column in first even though that wasn't listed until later. I chose to do this because they were making a mess of all the outputs.
3. I added many not null/not na statements to ensure that any data points that were missing were not used in attempting to calculate any metrics.
4. Decided to make a manifest.json to store an overview of all the successful reports and failed ones.
5. I had some extra checks in the plots, like a mini check to exclude undefined (NaN or infinite) Sharpe values from rankings because I originally had microsoft creating a nan sharpe value until I fixed the data cleaning.
6. Determined insufficinet history by the 50 prices limit since that is the minimum needed to have a 50 day moving average as called later in the code.
7. An extra check in data cleaning to ensure there were no duplicate days. This doenst really matter for graphing purposes but repeated dates could mess up any statistics like mean, median, std...
8. Included a limit when forward filling missing adj_closed values. For this part, if the adjusted close is missing but the close is present, we can use other rows where the adjusted close and raw close are present to make a rescaling factor. (This is a logical correction to make because the adjusted close represents the close adjusted for stock splits and dividends.) It does not make sense to forward fill because stock splits having a large impact on the values. For this we will include a limit to ensure no forward filling in large time gaps. Finally, if there is no factor to adjust by, quarantine that row to ensure that data remains valid.


---
## How Outliers Were Handled
There were 2 distinct types of outliers:
1. Impossible Bounds:
I adopted a hybrid approach to address bad data points:
- Contextual Identification: Identify the specific erroneous metric using surrounding context (e.g., if High is lower than Open, Low, or Close, replace High; if Close falls below Low, replace Close).
- Primary Repair Strategy (Option 1): Replace the invalid value with 'NaN' and forward-fill within the same ticker to maintain data continuity.
- Secondary Fallback (Option 3): If forward-filling fails or produces a value that still violates daily boundaries (e.g., forward-filled Close exceeds High due to multi-day gaps or sharp market movements), recalculate using same-day boundary bounds $ {High} = \max(\text{Open}, \text{Close})$


Note: A caveat of this methodology is its dependence on Open prices being accurate. I explored alternative, less biased approaches with Gemini—such as taking the OHLC median to locate anomalies—but ultimately set them aside. Most alternative options relied heavily on averaging Open and Close, introducing similar biases and disproportionately adjusting High and Close values due to their likelihood of being outliers.


2. (For extreme returns >3 std dev)
- For the flagging process, I flagged daily returns exceeding 3 from the ticker's mean on either raw close or adj_close (12,442 rows flagged). I decided to flag both because there were lots of artifical spikes in the adj_close values, even when the raw_close was not spiking.
- Classified flagged records as confirmed data errors if a multiplicative reversion check confirmed single-day spikes, or if `adj_return` diverged by over 50% from `raw_return`. Confirmed errors were set to `NaN` and repaired with interpolation.
- Records classified as unverified extreme movements were left unaltered as genuine market movements.
- Quarantined boundary rows containing extreme events on the first or last observation.


---
## Assumptions Made
- 252 trading days per year.
- Annualized risk-free rate of 2.0%.
- The ratio between `adj_close` and `close` remains semi-stable across short gaps of up to 5 trading days for interpolation.
- Zero-volume days treated as low-severity flags.
- Relied on the Open price as the primary benchmark for OHLC impossible bounds repairs.


---
## How to Interpret the Reports and Outputs
1. Each generated notebook contains 4 visual plots and a 10-metric summary table:
   - Plots: Top-Left (Price & 50-Day MA), Top-Right (Daily Returns), Bottom-Left (20-Day Rolling Volatility), Bottom-Right (Drawdown).
   - Summary Statistics Table: Reports Total Return, Annualized Return (extrapolated for the 6-month window), Annualized Volatility, Sharpe Ratio, Maximum Drawdown, Best/Worst Day, Positive Days, and Average Up/Down Day.
2. `portfolio_summary.csv`: Contains summary metrics for each executed stock report sorted by Sharpe Ratio, including `avg_daily_return`, `avg_volatility`, `max_drawdown`, and `total_return`.
3. `risk_return.png`: Plots Average 20-Day Rolling Volatility (%) against Total Return (%) for each stock, highlighting the top 5 stocks by total return.




