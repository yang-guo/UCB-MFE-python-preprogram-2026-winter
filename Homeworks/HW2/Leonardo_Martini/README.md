# HW2 - Stock Analysis (Leonardo Martini)

## What is in this folder
- `01_data_cleaning.ipynb`: cleans the `ohlc_hw` table and saves the clean data as `ohlc_hw_cleaned` in `data/data.db`.
- `02_stock_report_template.ipynb`: makes a risk report for one stock. You can change the stock and the dates in the first cell.
- `03_run_all_stocks.py`: makes the report for every stock and a summary of all of them.
- `reports/`: 3 example reports, the summary table (`portfolio_summary.csv`), the summary chart and the list of failed stocks.
- `requirements.txt`: the Python packages I used.

## How to run it
1. Download `data.db` from bCourses and put it in the `data` folder (it is not uploaded to GitHub).
2. Install the packages with `uv sync` (or `pip install -r requirements.txt`).
3. Run all the cells in `01_data_cleaning.ipynb`.
4. Run `python 03_run_all_stocks.py` from this folder. The reports are saved in `reports/<today's date>/`.

## How I cleaned the data
- Some tickers had HTML code around the name (for example `<span>KO</span>`). I removed it, so there are 30 tickers instead of 150. I kept the original name in a separate column.
- I deleted 29,460 rows that were exact copies of other rows. I didn't find rows with the same stock and date but different values.
- I checked for missing trading days using the dates in the `ohlc` table. None were missing. I also tested the check on a small example where I removed one date on purpose.
- 5 rows had an open price of 0, all on the first day of the stock history. I replaced it with the close of the same day, because there is no day before.
- adj_close was missing in 18,374 rows and wrong in 10,417 rows (much too big). The normal close is the same as in the `ohlc` table, so I calculated adj_close again using the ratio adj_close / close from `ohlc` on the same day.
- 75 days have volume 0. I kept them but marked them, because a day with no trading is possible.
- I didn't change the original table, and I added columns that show which rows I repaired.
- Some problems are not in the data (missing prices, high lower than close, missing volume, two different rows for the same day). I tested them on a small fake example at the end of the cleaning notebook.

## Outliers
- I marked the days where the return is more than 3 standard deviations from the average of that stock. I didn't delete them, they are just marked for checking.
- On 3 days TRV's price doubles and the next day it goes back to normal. This is clearly a mistake in the data, so I replaced those prices with the average of the day before and the day after.
- The other big moves look real (market crashes, important news), so I kept them.

## Assumptions
- The reports use the adjusted close and daily log returns, with a risk-free rate of 2% and 252 trading days in a year.
- All stocks use the same period: the last 6 full months before the month of the last date in the data. The last date is 6 December 2024, so the period is 1 June to 30 November 2024.
- 22 stocks have data only until 18 October 2024, the others until 29 November. Each report shows its real first and last date, and the script prints the stocks with less data, so the comparison between them is not perfect.
- A report needs at least 21 prices. If a stock has fewer, the report stops and the stock is written in `failed_stocks.txt`.

## How to read a report
- Top left: the price, red when the stock is more volatile than usual, green when it is calmer, grey at the start when there is not enough data yet. The blue line is the 50-day average price.
- Top right: daily returns, green for up days and red for down days.
- Bottom left: the volatility over the last 20 days. Red areas are above the average, green areas below.
- Bottom right: the drawdown, so how much the price is below its previous highest point. The black dot is the biggest fall.
- The table at the end shows the main numbers: total and yearly return, volatility, Sharpe ratio (how much return you get for the risk you take), biggest fall, best and worst day, and how often the stock went up.
- The portfolio summary ranks the stocks by Sharpe ratio and shows a chart of risk against return.
