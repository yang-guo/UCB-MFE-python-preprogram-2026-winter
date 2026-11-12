**Mila Marsot**

# HW2: Stock Analysis Pipeline

- `01_data_cleaning.ipynb`: cleans `ohlc_hw` and saves `ohlc_hw_cleaned`
- `02_stock_report_template.ipynb`: risk report for one stock
- `03_run_all_stocks.py`: runs the report for all stocks and builds a summary
- `reports/2026-10-26/`: three example reports (AAPL, IBM, MRK), `portfolio_summary.csv` and `risk_return_profile.png`

## How to run

1. Install the packages in `requirements.txt` (Python 3.12).
2. Put `data.db` from bCourses in `data/data.db` (it is not committed).
3. Run all cells of `01_data_cleaning.ipynb`. This creates `ohlc_hw_cleaned`, which the other two files need.
4. Run `python 03_run_all_stocks.py`. The results are saved in `reports/{today's date}/`.

## Design decisions

- I never change `ohlc_hw`: I clean a copy and save it as a new table.
- I clean the ticker names first (150 labels with HTML become 30 tickers).
- I list all the problems in a quality report before repairing anything.
- Each repaired value has a `_repaired` flag in the clean table.
- I use the `ohlc` table only to know which days to expect and to repair adjusted prices.
- Some problems from the assignment are not in the data, so I add them on purpose to test my code.
- Part 3 runs the same report notebook for each stock with papermill and reads the results with scrapbook.

## Outliers

- **Open = 0 (5 rows, on a stock's first day):** replaced with the last known close if it is between the low and the high, otherwise the middle of the two. On a first day, the last known close is that day's close.
- **Missing or wrong adjusted close (about 28,800 rows):** rebuilt as `close × factor`, with the factor from `ohlc` for the same day.
- **Extreme returns:** kept, because most are real market moves. Only 4 days were errors (zero volume, flat price, a jump that comes back the next day); I replaced them with the average of the days around them. MCD on 1968-01-02 is uncertain, so I kept it with its flag.
- **Zero volume (75 old days):** kept with a flag, since the prices are valid.

## Assumptions

- The expected trading days of a stock are its days in `ohlc`.
- Part 3 uses the last 6 complete months of the database: 2024-06-01 to 2024-11-30.
- The Sharpe ratio uses a 2% risk-free rate and 252 trading days per year.

## Reading the reports

- **Price:** red when volatility is high, green when it is low, gray at the start.
- **Returns:** green bars for up days, red for down days.
- **Volatility:** how much the price moves; red areas are above average.
- **Drawdown:** how far the price is below its highest point; the black dot is the worst moment.
- **Sharpe ratio:** return per unit of risk; higher is better.

Careful when comparing stocks: 22 of the 30 stocks have data only until 2024-10-18, so their results cover a shorter period.
