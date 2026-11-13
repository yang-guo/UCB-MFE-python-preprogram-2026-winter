# Strategy, features, model, and validation
- The strategy implemented is dollar cost averaging with quarterly rebalance. Specifically, it checks at the beginning of each March, June, September, and December how many stocks are in the universe, and allocate NAV equally to all stocks for holding of 90 days.

- Reused course_data.py unchanged for features, model, and validation.
    - Features:
        - Five price features: 21-, 63- and 252-minus-21-day momentum, 63-day volatility, 252-day drawdown.
        - Six accounting ratios: net margin, operating ROA, debt/assets, cash/assets, FCF/assets, accruals/assets.
        - Filing age and sector.
        - A financial row is usable only from the day after its filing date. Anything older than 550 days is set to missing.
    - Model
        - The pipeline clips tails, median-imputes with a missing flag, scales and one-hot encodes the sector, then fits Ridge or gradient boosting. All of that is fitted on training rows only.
    - Validation
        - Three expanding-window yearly folds (2013, 2014, 2015). 
        - Every training label ends before the fold’s first signal. 
        - The buffer period exists so 2015 signals have finished targets.

# Design decisions
1. Decision dates
- Entry is the first market session of March, June, September and December, and the signal is the prior session’s close. This reuses the month-change check from MonthlyTopNStrategy.generate_orders.
2. Training rows
- To keep memory manageable, I use every 5th session plus the decision signals.
3. Walk-forward fits
- There is one refit per decision date, using only rows with label_end < decision date. This applies in the final test too, not just before the buffer.
4. Model selection
- Candidates are Ridge (several alphas, with and without winsorizing the training target) and one small gradient-boosting model. 
- The 2013, 2014 and 2015 validation folds come from date_folds, and I select by mean daily rank IC while reporting RMSE.
5. Model use
- Since the strategy buys all stocks equally, by default the model isn’t used in the portfolio.
- Added an optional top_fraction gate (buy only the model’s top X%) for testing on development data whether the model adds value.

# How to run
1. Put equities_top2000.duckdb in Lectures/modeling/data/
2. Save hw4_backtest.py and hw4_model.py in Lectures/modeling/data/, next to course_data.py and backtest.py.
3. Run python hw4_model.py. 
- It builds the panel, validates, picks the model, writes walk-forward scores and fit_audit.csv to outputs/hw4/.
4. Run python hw4_backtest.py. 
- It compares strategy variants on development data, then runs the final test once with chosen. 
- It saves the audit output, NAV and fills, and cumulative_return_vs_sp500tr.png in in Lectures/modeling/outputs/hw4/ folder.

# Findings and next steps
- In the rising market of 2016 to 2026, the dollar cost averaging strategy trails SP500 in return, but has lower volatility.
    - Sepecifically, the return for our strategy was ~$1.12mn net of transaction fees, while SP500 achieved ~$3.40mn on the same initial investment amount of $1mn.
    - Our strategy has an anualized volatility of ~13.8% while SP500 was ~17.8%.
- Next steps
    - Improve dollar cost averaging with less targets, i.e. instead of rebalancing over ~2000 tickers, choose 20~30 tickers. 
    - The 20 ~ 30 tickers will be selected based on newly-engineered factors, starting from price / volume factors. The 20 ~ 30 tickers will also be filtered using a covariance matrix gate to check correlation to market.
    - Expecting the strategy to reduce volativity further (thereby improving Sharpe ratio), and reduce transaction costs.

# HW4 submission checklist
- Your code: a notebook, Python files, or both, with instructions to run it. 

**See hw4_model.py and hw4_backtest.py** 

- A brief explanation of your features, model, validation, and trading strategy

**See Strategy, features, model, and validation section** 

- The performance comparison and cumulative-return chart

**See hw4_outputs/hw4/**

- Evidence that you checked the backtest's accounting and timing

**See hw4_outputs/hw4/ (e.g. check_signal_timing proves orders ignore post-signal prices, and fit_audit.csv shows last_label_end < fit_date for every fit.)**

- Your findings, including unsuccessful ideas or changes you would try next

**See Findings and next steps section** 

# Citations
- course_data.py, backtest.py, 1_backtesting.ipynb, 2_research_workflow.ipynb, together with python libraries used for these files were used in building the model and backtest of this homework.
- dollar cost averaging is a well-understood strategy and therefore no specific citation here. For reference, see https://www.investopedia.com/terms/d/dollarcostaveraging.asp 
- Claude was used in generating codes.