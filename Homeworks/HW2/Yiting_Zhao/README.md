# Homework 2: Stock Analysis Pipeline— Yiting Zhao

## Setup

From this folder:

```bash
uv venv --python 3.12.9
source .venv/bin/activate
uv pip install -r requirements.txt
```

Download `data.db` from bCourses and place it at `data/data.db` (not committed).

## How to run

1. `01_data_cleaning.ipynb`: run top to bottom. This creates the `ohlc_hw_cleaned` table in `data/data.db`.
2. `02_stock_report_template.ipynb`: parameterized single-ticker report. You don't run this directly; step 3 runs it for every ticker.
3. `python 03_run_all_stocks.py`: generates one report per ticker in `reports/YYYY-MM-DD/`, plus `portfolio_summary.csv`, `failed_stocks.txt`, and `risk_return.png`.

Only 2–3 sample reports and the summary CSV are committed. Running step 3 regenerates the full set.

## Key design decisions

- **Output folder:** named by the run date (`reports/YYYY-MM-DD/`), independent of the analysis window.
- **Error handling:** Failed tickers are saved to `failed_stocks.txt`. 
- **Collecting metrics:** instead of Scrapbook, the script recalculates each stock's metrics from the database using the same window and formulas as the template. 

## Missing values

In `01_data_cleaning.ipynb`:

- **Missing raw prices** are forward-filled within each ticker (the last known price is carried forward). Rows that were filled are marked in the `price_filled` flag.
- **Missing volume** is filled with 0 and marked in the `volume_filled` flag. 
- **Missing adjusted close** is rebuilt as our own raw close multiplied by the same-date adjustment factor (`adj_close / close`) from the reference `ohlc` table, matched on ticker and date. Rebuilt rows are marked in `adj_from_ref`. Rows with no matching factor are quarantined.

## Outlier handling
 **Impossible price relationships** (high below close, low above close, or open outside the low-high range): replaced with the previous day's close for all four raw prices, and the previous day's adjusted close (Option 2). Setting all four to the same value guarantees the row is internally consistent without inventing new prices. Repaired rows are marked in `bounds_repaired`. A bad row on a ticker's first day has no previous day, so it is quarantined instead.
- **Wrong adjusted close**: some rows had `adj_close` 5–24 times larger than `close`. Each row's `adj_close / close` ratio is compared with the reference factor for the same date; rows more than 5% off are rebuilt from the reference factor and marked in `adj_repaired`. Rows with no reference factor and a ratio far from the ticker's usual ratio are quarantined.
- **Extreme returns** (more than 3 standard deviations from the ticker's mean): these are flagged for review, not deleted. Each one is labeled in `return_flag` based on evidence:
  - `split or corporate action`: the reference adjustment factor changed that day. Kept.
  - `data error, interpolated`: the price jumped and returned to its prior level the next day, which suggests a bad value. Replaced with the average of the previous and next day.
  - `kept, cause unverified`: no evidence either way, so the move is kept and the uncertainty is stated rather than claiming news or a merger.
- **Duplicate (ticker, date) rows**: the first copy is kept so every key is unique.
- 
## Assumptions

- 252 trading days per year for annualizing
- Risk-free rate of 2% per year for the Sharpe ratio
- 20-day rolling window for volatility
- Daily returns are log returns of adjusted close
- Prices are not filled forward through missing days; a stock whose data ends early is analyzed only over the days it has
- Sharpe is left undefined (NaN) when there are fewer than 2 returns or zero volatility, and those stocks are excluded from rankings

## Interpreting the reports

**Report notebooks** (`{ticker}_report.ipynb`): one per stock, covering the requested date window. Each report shows:

Data summary: the number of trading days and the actual first and last dates in the data. These can differ from the requested window if a stock's history starts later or ends earlier. A warning appears if there are fewer than 21 prices, and metrics that need more history show N/A.
Price and 50-day MA (top left): adjusted close over time. The line is red when 20-day volatility is above its average for the period, green when below, and gray during the first 20 days before volatility can be measured. The blue dashed line is the 50-day moving average, which starts after 50 prices.
Daily returns (top right): daily log returns as bars, green for up days and red for down days.
Rolling volatility (bottom left): 20-day volatility, annualized. Red shading marks periods above the average, green shading periods below.
Drawdown (bottom right): how far the price is below its highest point so far. The black dot marks the maximum drawdown.
Summary table: total return, annualized return and volatility, Sharpe ratio, maximum drawdown, best and worst day, share of positive days, and average up and down day.

**How to read the key numbers**

Total return: the overall gain or loss from the first to the last price in the window.
Annualized volatility: how much the stock typically swings in a year. Higher means riskier.
Sharpe ratio: return earned per unit of risk, after subtracting the 2% risk-free rate. Above 1 is generally good; below 0 means the stock did worse than the risk-free rate.
Maximum drawdown: the largest peak-to-trough drop, the worst loss an investor could have experienced in the window.

**portfolio_summary.csv** columns:

- `avg_daily_return`: mean daily log return (as a fraction)
- `avg_volatility`: average of the 20-day annualized rolling volatility
- `max_drawdown`: worst peak-to-trough drop (e.g., -0.25 = -25%)
- `sharpe_ratio`: annualized return above the risk-free rate, per unit of volatility; higher is better
- `total_return`: last adjusted close / first adjusted close - 1
- `first_date`, `last_date`, `n_days`: each stock's actual data coverage within the window
- `report_path`: path to the stock's report notebook

**risk_return.png:** each dot is a stock. Further right means more volatile, higher means more total return, and greener means a higher Sharpe ratio. The 5 highest-return stocks are labeled.





