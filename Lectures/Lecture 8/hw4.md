# Homework 4: Building a Stock Return Prediction and Trading Strategy

**Due Date**: November 12, 2026 (two weeks after the final Modeling lecture on October 29).
**Submission**: Pull request to the course repository

## Overview

In this capstone assignment, you'll build a complete quantitative trading system that:
1. Trains a machine learning model to predict stock returns
2. Implements a trading strategy based on your predictions
3. Backtests the strategy against a benchmark (S&P 500)

This assignment integrates everything you've learned: data manipulation, feature engineering, machine learning, and performance evaluation. Your goal is to **beat a simple S&P 500 buy-and-hold strategy**.

---

## The Data

You'll work with the `signals` table in `data.db`. This table contains daily data for multiple stocks with various features that you can use for prediction.

**Important Date Ranges:**

| Period | Start Date | End Date | Purpose |
|--------|------------|----------|---------|
| Training | 01/01/2010 | 12/31/2015 | Train your initial model (6 years) |
| Gap | 01/01/2016 | 03/31/2016 | Buffer period (not used) |
| Testing | 04/01/2016 | 12/31/2025 | Evaluate your strategy |

---

## Part 1: Model Development

Create a model that predicts the **90-day forward return** for each stock on each day.

### 1.1 Target Variable

Your target variable is the 90-day return:
```python
# 90-day return = (price_in_90_days - price_today) / price_today
target = (close_t+90 - close_t) / close_t
```

### 1.2 Feature Engineering

Use any features available in the `signals` table. You may also create derived features. Some ideas:
- Technical indicators (momentum, volatility, moving averages)
- Relative metrics (stock vs. market, stock vs. sector)
- Lagged features

**Critical**: Never use future data to create features. All features must be known at time `t` to predict returns from `t` to `t+90`.

### 1.3 Model Selection

You may use any model:
- Linear regression, Ridge, Lasso
- Random Forest, Gradient Boosting
- Neural networks
- Or any other approach

**Deliverable**: Document your model choice and rationale.

---

## Part 2: Trading Strategy

Implement a trading strategy that uses your model's predictions to make investment decisions.

### 2.1 Trading Rules

| Rule | Description |
|------|-------------|
| **Entry** | You may enter any stock position at the **close price** on any trading day |
| **Holding Period** | All positions must be held for exactly **90 calendar days** |
| **Exit** | Sell at the **close price** on day 90 (or next trading day if market is closed) |
| **Cash Management** | Cash is consumed when entering positions, returned when exiting |

### 2.2 Initial Capital

- **Starting Cash**: $1,000,000
- **Benchmark**: $1,000,000 invested in S&P 500 on 04/01/2016 (buy and hold)

### 2.3 Position Management

Each day during the test period, you decide:
1. **Which stocks to buy** (based on your model's predictions)
2. **How much to allocate** to each position

Constraints:
- You can only invest cash you currently have available
- You cannot borrow or use leverage
- Cash returns to you only when positions are closed (after 90 days)

---

## Part 3: Performance Evaluation

Calculate the following metrics for your strategy during the **test period** (04/01/2016 - 12/31/2025):

### 3.1 Required Metrics

| Metric | Description |
|--------|-------------|
| **Final Portfolio Value** | Cash + market value of open positions on 12/31/2025 |
| **Annualized Return** | Geometric mean of yearly returns |
| **Maximum Drawdown** | Largest peak-to-trough decline |
| **Sharpe Ratio** | Risk-adjusted return (assume risk-free rate of 2%) |

### 3.2 Required Visualization

Create a plot showing:
- Your portfolio's cumulative return over time
- The S&P 500 benchmark's cumulative return over time
- Both lines on the same chart for comparison

Example structure:
```python
def calculate_annualized_return(start_value: float, end_value: float, years: float) -> float:
    """Calculate annualized return from start and end values."""
    return (end_value / start_value) ** (1 / years) - 1

def calculate_max_drawdown(portfolio_values: pd.Series) -> float:
    """Calculate maximum drawdown from a series of portfolio values."""
    cummax = portfolio_values.cummax()
    drawdown = (portfolio_values - cummax) / cummax
    return drawdown.min()

def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    """Calculate annualized Sharpe ratio."""
    excess_returns = returns.mean() * 252 - risk_free_rate
    return excess_returns / (returns.std() * np.sqrt(252))
```

---

## Part 4: Model Retraining (Optional but Recommended)

You may retrain your model during the test period using a rolling window approach:
- Only use data available up to the current date
- Never include future data in training

Example approach:
```python
# On each day t in test period:
# 1. Train on data from (t - training_window) to (t - 1)
# 2. Predict returns for all stocks on day t
# 3. Make trading decisions
```

---

## Submission Requirements

### What to Submit

1. **Code**: Jupyter notebook or Python script with your complete solution
2. **Documentation**: Brief explanation (in markdown cells or a README) covering:
   - Your model choice and why
   - Key features you engineered
   - Your trading strategy logic
   - Any interesting findings

### Required Output

Print or display the following:

```
============ PORTFOLIO PERFORMANCE ============
Final Portfolio Value:    $X,XXX,XXX.XX
Annualized Return:        XX.XX%
Maximum Drawdown:         -XX.XX%
Sharpe Ratio:             X.XX

============ BENCHMARK (S&P 500) ============
Final Value:              $X,XXX,XXX.XX
Annualized Return:        XX.XX%

============ RESULT ============
Beat Benchmark:           Yes/No
Outperformance:           +/-XX.XX%
===============================================
```

---

## Evaluation Criteria

Your submission will be evaluated on:

| Criteria | Description |
|----------|-------------|
| **Correctness** | Does the backtesting logic correctly implement the trading rules? No look-ahead bias? |
| **Code Quality** | Clean, readable, well-organized code |
| **Model Approach** | Reasonable model choice and feature engineering |
| **Documentation** | Clear explanation of your approach |

This is a pass/no-pass assignment. To pass, you must:
- Implement a working backtesting system
- Calculate all required metrics correctly
- Produce the required visualization
- Document your approach

**Bonus recognition** (not required to pass): Beat the S&P 500 benchmark!

---

## Tips for Success

1. **Start Simple**: Get a basic model working first, then iterate
2. **Feature Engineering Matters**: This is often where you'll see the most improvement
3. **Watch for Look-Ahead Bias**: Double-check that you're not accidentally using future data
4. **Mind Training Time**: Complex models can be slow; use the learning curve to estimate training time
5. **Hyperparameter Tuning**: Useful for optimization, but won't save a fundamentally weak model
6. **Test Incrementally**: Verify your backtest logic with simple examples before running the full simulation

---

## Resources

### Required Libraries
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sqlite3
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor  # or your model of choice
```

### Database Connection
```python
conn = sqlite3.connect('data/data.db')
signals = pd.read_sql_query("SELECT * FROM signals", conn)
```

### Getting S&P 500 Data
The S&P 500 can be found in the signals table with the appropriate ticker symbol.

---

## Questions?

Post in Slack. Good luck!

**Remember**: The goal is to demonstrate your ability to build an end-to-end quantitative system. A well-documented, correctly-implemented strategy that underperforms is better than a buggy one that appears to beat the market!
