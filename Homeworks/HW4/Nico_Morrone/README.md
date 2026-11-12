# HW4: Stock Return Prediction and Backtesting

## Overview

For this assignment, I developed a stock-selection strategy using historical price momentum and Ridge Regression. The model predicts stock returns over a 90-calendar-day holding period, and I use those predictions to select investments. I tested the strategy from April 1, 2016, through September 30, 2026, and compared its performance with the S&P 500 Total Return Index.

## Features and Model

I used the full historical database containing 3,849 stocks across quarterly top-2,000 selections. I checked each stock's eligibility using the membership data for its entry quarter, including companies that are no longer active.

My five features were 63-day momentum, 21-day annualized volatility, 126-day beta, 63-day maximum drawdown, and 63-day relative momentum. I chose Ridge Regression because it is a relatively simple way to predict numerical returns, and its regularization can help control the size of the model's coefficients.

I trained the initial model on 2010–2013 data and validated it on 2014–2015 data. Extreme observations caused unusually large prediction errors, so I clipped the training features and returns at the 1st and 99th percentiles. This reduced the validation MAE to 12.70 percentage points and the RMSE to 19.39 percentage points. I then trained the final model on the development data from 2010–2015, using only returns completed before April 1, 2016. The model was not retrained during the test period.

## Trading Strategy

At the beginning of each month, I first identify the top 20% of eligible stocks based on their previous 63-day momentum, excluding stocks with negative momentum. I then select the five stocks with the highest predicted 90-day returns from Ridge Regression.

The portfolio starts with $1,000,000. Each month, I invest up to one-third of the initial capital, divided equally among the five stocks, provided enough cash is available. Purchases occur at the next trading session's close after the prediction, and each purchase is held for 90 calendar days. Positions are scheduled to be sold on maturity or the next market session when maturity falls on a non-trading day.

The strategy is long-only, with no borrowing or leverage. I assume fractional shares, no transaction costs, and no interest earned on cash. I use adjusted closing prices and do not add dividends separately. Purchases without valid prices are skipped, and positions remain open if an exit price is unavailable. Stocks are not sold simply because they leave the quarterly universe. Positions still open on September 30, 2026, are included in the final valuation.

## Results

| Metric | My strategy (indicative) | S&P 500 |
| --- | ---: | ---: |
| Final value | $3,652,189.79 | $4,402,746.78 |
| CAGR | 13.13% | 15.17% |
| Maximum drawdown | -31.05% | -33.79% |
| Sharpe ratio | 0.653 | 0.773 |

My strategy increased the initial investment to approximately $3.65 million, but it finished about $750,557 below the benchmark. It had a slightly smaller maximum drawdown, although its annualized return and Sharpe ratio were lower. One possible reason is that the fixed monthly budget left some cash uninvested as the portfolio grew.

These results are indicative rather than fully verified. Five stocks (PLAN, AVLR, PACWP, SILK, and COMM) had no valid prices when their positions were scheduled to close, so their last recorded prices were used to estimate their values rather than to invent sales. Unusual adjusted-price movements may also affect the backtest. In a future version, I would investigate these missing corporate actions and unusual prices before relying on the performance figures.

## How to Run the Code

Place `equities_top2000.duckdb` in `Lectures/modeling/data/` within the course repository. From the repository root, set up the course Python environment using `uv sync --locked`, then open `HW4.ipynb` in Jupyter and select the course `.venv` kernel. Restart the kernel and run the notebook cells in order. The notebook finds the database from the repository folder and produces the regression results, backtest, performance comparison, and chart. The database is not included in the submission.

I used the course materials for the data definitions and ChatGPT to help develop, explain, and debug my code. I did not copy the lecture backtester.
