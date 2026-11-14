# HW4 Ridge stock selection and backtest

1. Code and instructions
   Run hw4_walkthrough.ipynb from top to bottom in Jupyter, or run python -B hw4.py --database /path/to/equities_top2000.duckdb from the HW4 submission folder. Install DuckDB, NumPy, pandas, scikit-learn, and Matplotlib first. 
2. Features, model, validation, and trading strategy
   I used trailing returns and 63-day volatility, calculated through the previous close, to predict roughly 90-day stock returns with Ridge regression. Expanding-window validation selected an alpha of 1,000, and the strategy bought up to 10 top-ranked stocks at the first trading close of each month, held each lot for at least 90 calendar days, and charged 5 basis points per side.
3. Performance comparison and cumulative returns
   From April 2016 through September 2026, the strategy gained 186.45% versus 340.27% for the S&P 500 Total Return Index, with maximum drawdowns of 48.33% and 33.79%, respectively. The cumulative-return chart compares both portfolios over the full test period. 
4. Accounting and timing checks
   I reconciled cash against the trade ledger and checked that daily NAV equaled cash plus the marked value of holdings. The tests cover pre-entry feature timing, exits after maturity, overlapping lots, and missing quotes. 
5. Findings and next steps
   I would consider to use some new features to rank the stocks, also non-linear models. And before I tried some other features, but it performed worse than current model. 
