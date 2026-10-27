### How to run my code?
1. Ensure course database at `Lectures/Lecture 4/data/data.db`
2. Install dependencies  `pip install -r requirements.txt`
3. Open `Homeworks/HW2/Ada_Kam/01_data_cleaning.ipynb` and run all cells in order
4. The notebook writes new table `ohlc_hw_cleaned` to `data/data.db`. Rerunning recreates the table
5. `03_run_all_stocks.py`: runs `02_stock_report_template.ipynb` for every ticker, takings parameters ticker, start, end date, path, volatility window and produces charts and summary on risk profile
6. Outputs: `reports/{run date}/{TICKER}_report.ipynb`, `portfolio_summary.csv`, `risk_return_profile.png`, `failed_stocks.txt`

### Key Design Decisions
- Normalzied tickers by stripping HTML/whitespace
- Duplicates removed
- Severity: high = missing trading dates; medium = repairable price/adj_close errors; low = extreme returns and zero volume (often legitimate)
- Cleaning statistics shown including how many of each issue is repaired in the comparison table
- For report, validates input before analysis, missing database, start date end date sequence, volatility window < 2, fewer than 21 prices, missing data
- For part 3 batch run, errors for each ticket caught and logged to failed_stocks.txt, rankings exclude undefined Sharpe


### Handling Outliers
- For impossible price relations: use previous day's close, it assumes there is no px change on the day, anchor all corrupted OHLC fields to yday's closing px for consistency
- For adj_close, use same-day close and adj_close to cal factor from ohlc, so that no previous day data is used that may 
- For extreme returns, I categorized the moves as 'market-wide' if 5 or more stocks moved by same extreme, 'high-volume' if volume is 1.5x rolling 5-day mean, the rest are unexplained and will need manual review, if it is a data-error after review i.e. > 10 s.d. away and volume = 0, interpolate using t-1 and t+1 days
- Quarantined rows are for rows with unresolved prices

### Assumptions made
- ohlc table is the reference used for good trading days
- If trading volume is 1.5x prev 5 day average, assumed daily return of > 3 s.d. is legitimate

### How to interpret reports
- 22 tickers end on 2024-10-18, while 8 end on 11-29, different lengths of comparison periods used
- Risk Analysis Plots:
  - Price and 50 dma: px action, momentum and volatility from Jun 2024
  - Rolling-volatility: chg in x-day vol over period, with horizontal line representing mean vol over period
  - Maximum Drawdown: price change from peak with max dd labeled
  - Daily Returns: gains (positive returns) in green, losses in red, magnitude represented by bar abs bar height
- Risk-return plot: higher and further left is better