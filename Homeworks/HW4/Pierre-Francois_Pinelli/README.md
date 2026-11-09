HW4: Pierre-Francois Pinelli

The idea is to predict which stocks will perform better over the next 90 days, then use these predictions in a long-only strategy. I use an elastic net with 18 characteristics and buy the 20 stocks with the highest scores. The notebook contains the calculations, checks and results.

How to run

From the course repository root, run `uv sync --locked` and select the .venv environment as the notebook kernel. Download equities_top2000.duckdb from bCourses into Lectures/modeling/data/.

Then, from this homework folder:
- Run `python test_backtest.py` for the 9 accounting tests. They do not need the database.
- Open hw4.ipynb and run all cells in order.

If the database is somewhere else, set HW4_DB to its full path before starting Jupyter. The default path assumes the course repository structure. The panel is cached in outputs/. Delete the cached panel if you change the data or feature calculations.

Files:
- hw4_data.py: loads the data, builds the characteristics and the 90-day returns.
- hw4_model.py: ranks the characteristics, fits the model and evaluates the scores.
- hw4_backtest.py: manages cash, purchases, sales, costs and performance metrics.
- test_backtest.py: small examples with results computed by hand.
- hw4.ipynb: runs the study and saves results_summary.csv and cumulative_return.png.

Data and model

There is one row per stock and month-end. I keep the 1,000 largest stocks by the dated rank in the course top-2000 universe, with a price at the signal date and 63 complete daily returns. This leaves around 990 stocks per month.

The 18 characteristics cover momentum, volatility, drawdown, beta, dollar volume, profitability, debt, cash, accruals, asset growth, valuation and size. Financials become available at filing date + 1 day. I stop using them when the fiscal quarter is more than 550 days old.

Each month, I rank the characteristics across eligible stocks between -0.5 and 0.5. Missing values become 0, the middle. The target is the 90-day return, clipped at the 1% and 99% quantiles of each date, minus the average clipped return of that date. So the model predicts relative performance within the universe.

I chose an elastic net because it is a simple linear model with L1 and L2 penalties. L1 can remove characteristics, while L2 reduces the size of the coefficients. This is useful because several characteristics are correlated.

The model is refitted every quarter on an expanding sample starting in 2010. A training observation can only be used once its 90-day outcome has ended, strictly before the fit date.

I compared 12 combinations of alpha and l1_ratio, with portfolios of 20, 30 or 50 stocks. The choice is based on the best Sharpe ratio in the 2012-2015 development backtest. The selected parameters are alpha = 0.0003, l1_ratio = 0.8 and N = 20.

Trading rules

The initial capital is $1M. At each month-end close, I rank the stocks using the latest model. Purchases happen at the next session's close, with equal budgets across the top 20. Each monthly group of purchases receives up to one third of portfolio value, limited by available cash.

Each purchase is held for 90 calendar days, then sold at the first available close. Sale proceeds are reinvested at the next close using the latest monthly scores. Different groups of purchases can hold the same stock at the same time.

There is no borrowing, cash earns 0%, and costs are 10 bps on each purchase and sale. Positions use fractional units of adjusted prices, so dividends are included in the return series.

Some price series are problematic. I exclude stocks with a move above x5 or below /5 during the last 252 sessions. The backtester also rejects quotes more than x5 away from its reference price and logs them. These filters do not catch every data error.

When a quote is missing, the main run keeps the last price until another quote appears. I also run a stress scenario that writes the position off after more than 5 sessions without a quote. The database has no delisting proceeds, so neither treatment tells us the actual recovery value.

Checks and results

The 9 tests cover next-close execution, the holding period, costs, cash constraints, missing quotes, bad prints, reinvestment, open positions and performance metrics.

In the notebook, I also check a real trade by hand, rebuild cash from the trades and reconcile realized plus unrealized PnL with total PnL. Portfolio value computed from lagged weights and returns must match cash plus holdings. There are timing checks for filings, membership, labels and model refits.

From April 1, 2016 to September 30, 2026, the notebook reports a final value of $4.57M for the strategy and $4.40M for the S&P 500 total return index. CAGR is 15.57% versus 15.16%.

The strategy takes more risk: volatility is 23.0% versus 17.8%, maximum drawdown is -36.8% versus -33.8%, and Sharpe is 0.66 versus 0.77. The slightly higher return therefore does not mean better risk-adjusted performance.

The average test rank IC is 0.036. Its reported t-stat of 3.6 does not adjust for dependence between months. Since the 90-day targets overlap, I would need an adjusted standard error before making a significance claim.

With costs of 25 bps per side, CAGR falls to 14.24%. With the write-off scenario, it falls to 13.72%. In the main run, about 15% of final value is in positions without a recent usable quote. These scenarios are not guaranteed lower and upper bounds on the true result.

Previous attempts and limits

Before this version, I ran other methods on the same test period. The earlier results recorded during the project were:
- 12-1 momentum before the data fixes: $2.52M final value.
- 12-1 momentum after the fixes: $4.61M.
- Characteristic shrinkage: $4.04M, or $3.60M with quarterly model selection.
- Ridge regression: $3.25M.
- Factor momentum: $2.62M with a 3-month window, $2.72M with a volatility-scaled 12-month window.
- Volatility-adjusted 12-1 momentum: $3.07M.
- Elastic net, the version in these files: $4.57M.

The test period influenced the research choices, so it is not an untouched test set. The quarterly predictions use past outcomes, but that does not remove the bias from comparing several approaches on the same period.

The financial data contains the vendor's latest values rather than original filing versions. Dated membership helps with survivorship bias but does not remove all data limitations. Transaction costs are simplified and there is no market impact model.

For me, the model gives an encouraging ranking in this sample, but the small advantage over the S&P is too sensitive to costs and missing prices to claim a robust outperformance. I would first improve the treatment of acquisitions and delistings, then check how much of the result comes from beta and sector exposure.

Sources and assistance

The data functions are adapted from Lectures/modeling/course_data.py. The metric conventions and some accounting checks follow Lectures 7-8. The backtester is a separate implementation, with the course backtest.py used as a reference.

I designed the approach: an elastic net on ranked characteristics, a monthly top-20 strategy with 90-day lots, costs proportional to turnover, and computing portfolio value from yesterday's weights times today's returns. Claude wrote most of the code from these choices and ran the experiments. I then went through the whole project myself, redid it by hand, and can explain every part of it.

References used in the project: Zou and Hastie (2005), Regularization and variable selection via the elastic net; Gu, Kelly and Xiu (2020), Empirical asset pricing via machine learning; Kozak, Nagel and Santosh (2020), Shrinking the cross-section.
