# HW2: Stock Data Cleaning and Reporting

This homework has three main pieces:

- `01_data_cleaning.ipynb` - inspects, cleans, and validates raw OHLC data.
- `02_stock_report_template.ipynb` - a parameterized notebook that generates one stock risk report.
- `03_run_all_stocks.py` - runs the template for every ticker and aggregates portfolio-level results.

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

1. Run `01_data_cleaning.ipynb` top to bottom. It writes the cleaned data to a new SQLite table named `ohlc_hw_cleaned`.
2. For a single stock, open `02_stock_report_template.ipynb`, set `ticker`, `start_date`, `end_date`, and `db_path`, then run it.
3. To generate reports for all tickers, run:

```bash
python 03_run_all_stocks.py
```

The batch script discovers all tickers in `ohlc_hw_cleaned`, selects the last six complete calendar months before the latest observation, and uses `papermill` to execute the template for each ticker. Results are written under `reports/<run-date>/`.

## Key Design Decisions

- The cleaning notebook produces a single cleaned table; the report notebook only reads that table.
- **Use adjusted close for returns and drawdowns.** Adjusted prices incorporate corporate actions and are more appropriate for financial return calculations.
- **Parameterize the report notebook.** `ticker`, date range, and database path are injected through a tagged parameters cell, which lets the same notebook be reused by the batch runner.
- **Automate batch reporting with Papermill and Scrapbook.** The runner executes the template per ticker and collects a glued `metrics` dictionary into a portfolio summary.

## Outlier Handling

- Raw ticker symbols are extracted from HTML, stripped, and uppercased.
- Exact duplicate rows and duplicate `(ticker, ts)` rows are removed.
- Rows with all four raw prices missing are dropped.
- Missing volume is filled with `0` and flagged as repaired.
- Missing adjusted close is reconstructed from the reference table's `adj_close / close` factor when possible.
- Impossible OHLC relationships, such as `high < low`, `close` outside `[low, high]`, or non-positive/non-finite prices, are treated as corrupted. The offending price field is set to `NaN` and forward-filled within each ticker.

Forward-fill is used because it preserves the time order and keeps the rest of a partially corrupted record usable. The trade-off is that repaired days can create flat prices and mechanically distorted next-day returns, so those affected returns are tracked separately.

- Extreme returns more than 3 standard deviations from the ticker mean are not blindly removed. They are classified and either kept with a flag, interpolated when there is confirmed evidence of a bad close with valid neighbors on both sides, or quarantined when a safe repair is not possible. Quarantined rows are excluded from the final cleaned table.

## Assumptions

- The SQLite database has a raw `ohlc_hw` table and a reference `ohlc` table with matching `ticker` and `ts` keys.
- Adjusted close is preferred over raw close for report metrics.
- A year has 252 trading days and the risk-free rate is 2%.
- At least 21 prices are needed to produce a meaningful 20-day rolling volatility; shorter series show `N/A` where applicable.
- The batch window is the last six complete calendar months before the month of the latest observation.

## Interpreting the Reports

Each per-ticker notebook shows:

- adjusted-close price with a 50-day moving average,
- daily log returns,
- 20-day annualized rolling volatility compared with its own mean,
- drawdown from the running peak,
- a summary table including total return, annualized return, annualized volatility, Sharpe ratio, maximum drawdown, best/worst day, and positive-day statistics.

`portfolio_summary.csv` is sorted by Sharpe ratio in descending order and contains:

- `avg_daily_return`
- `avg_volatility`
- `max_drawdown`
- `sharpe_ratio`
- `total_return`
- `report_path`

`portfolio_summary.png` plots total return against average rolling volatility, with points colored by Sharpe ratio. Higher total return for a given volatility, or a warmer color, generally indicates a better risk-adjusted result. `run_log.txt` records per-ticker status, and `failed_stocks.txt` lists any tickers that could not be processed.
