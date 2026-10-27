# HW2: Stock Analysis Pipeline

This homework cleans stock price data, creates individual stock risk reports, and runs the analysis for all 30 stocks.

`01_data_cleaning.ipynb` cleans `ohlc_hw` and saves `ohlc_hw_cleaned`.

`02_stock_report_template.ipynb` calculates risk statistics and creates charts for one stock.

`03_run_all_stocks.py` generates reports for all stocks and saves a portfolio summary.

`reports/` contains three sample reports, `portfolio_summary.csv`, and the portfolio risk and return plot.

## How to run

1. Install the packages listed in `requirements.txt`.
2. Download `data.db` from bCourses. The cleaning notebook first reads it from the homework folder and saves the cleaned database at `data/data.db`.
3. Run all cells in `01_data_cleaning.ipynb` to create the cleaned table.
4. Run `python 03_run_all_stocks.py` from the homework folder. The reports will be saved in a folder under `reports/` named with the run date.

## Design decisions

I keep the original `ohlc_hw` table unchanged and save the cleaned data in a separate table. I standardize ticker names, check for duplicates and missing dates, and record issues in a data quality report before cleaning. Missing prices are filled using previous prices from the same stock. Rows without any raw prices are removed. Missing volume is filled with zero and marked as a repair, since unknown volume does not necessarily mean there was no trading.

I use the `ohlc` reference table to check expected trading dates and help reconstruct adjusted closing prices using matching dates. I also keep a `repair_flag` to record changes. Part 3 uses Papermill to run the same report for each stock and Scrapbook to collect the results. If a stock fails, the script records the error and continues.

## Outlier handling

I flag daily returns that are more than three standard deviations from the stock's average return. I do not automatically remove these observations because large price changes can be real. I also check for invalid prices and cases where the open or close falls outside the day's high and low. When there is enough information, I repair questionable values. Otherwise, I flag or exclude uncertain observations instead of guessing. I check the price relationships again after cleaning.

## Assumptions

I use the dates in `ohlc` as the expected trading calendar for each stock. The reports use adjusted close and daily log returns. Volatility and returns are annualized using 252 trading days, and the Sharpe ratio assumes a 2% annual risk free rate. The batch analysis uses the six complete calendar months before the latest month in the database. Some stocks have shorter histories within that period.

## Reading the reports

The price chart shows adjusted close and its 50 day moving average. Red indicates higher volatility, green indicates lower volatility, and gray indicates unavailable estimates. Daily returns show positive and negative days. The volatility chart compares rolling volatility with its average. Drawdown measures how far a stock falls below its previous peak. The summary table includes returns, volatility, Sharpe ratio, drawdown, and daily performance statistics. The portfolio plot compares risk and return across stocks. A higher Sharpe ratio indicates more historical return relative to risk under this measure.

I keep the database and full set of generated reports locally. The submission includes the code, README, requirements file, three example reports, and portfolio summary CSV.
