# HW4 — Project Notes and Findings

## 1. Project objective

Build a stock-return prediction model and long-only trading backtester using the full `equities_top2000.duckdb` dataset. Evaluate performance from April 1, 2016, through September 30, 2026, against the S&P 500 Total Return Index.

## 2. Model development

The model predicts stock returns over a 90-calendar-day holding period.

Three historical price features were selected:

- **20-day return:** Measures short-term momentum.
- **60-day return:** Measures medium-term momentum.
- **20-day volatility:** Measures the variability of recent daily returns.

Ridge regression was selected as a regularized linear baseline. The model uses standardized features and `alpha=100`.

Extreme training observations were addressed by clipping training targets and features at the 1st and 99th percentiles. Feature clipping thresholds were calculated using training data only.

## 3. Validation

The development period covers January 2010 through December 2015.

The initial model comparison trained on observations whose holding periods finished before January 2014 and validated on observations from 2014–2015.

Three regularization values were evaluated: 0.1, 10, and 100.

Validation correlations were approximately −0.024, indicating weak predictive performance. Increasing regularization did not materially improve the results.

## 4. Trading strategy

The strategy starts with $1,000,000 and follows these rules:

1. Identify eligible stocks from historical quarterly universe membership.
2. Calculate price features using information available on the signal date.
3. Predict each eligible stock's 90-day return.
4. Select the ten stocks with the highest predictions.
5. Buy at the next trading session's closing price.
6. Allocate available cash equally across selected positions.
7. Hold each purchase for 90 calendar days.
8. Sell at the first available closing price on or after maturity.
9. Reinvest proceeds when the portfolio becomes available for another cycle.

The strategy uses no leverage, assumes zero transaction costs, and earns no interest on cash.

## 5. Final backtest results

The corrected backtest completed successfully through September 30, 2026.

| Metric | Ridge strategy | S&P 500 Total Return |
|---|---:|---:|
| Starting capital | $1,000,000 | $1,000,000 |
| Final value | $1,811,338 | $4,402,747 |
| Total return | 81.13% | 340.27% |
| CAGR | 5.82% | 15.17% |
| Maximum drawdown | −70.91% | −33.79% |
| Sharpe ratio (2% risk-free) | 0.278 | 0.773 |

The strategy underperformed the benchmark by approximately 259.13 percentage points in cumulative return.

The simulation initiated 42 investment cycles, recorded 410 completed stock trades, and included 10 open positions in the final portfolio valuation.

## 6. Accounting and timing verification

The backtester automatically checks:

- Cash never becomes negative.
- Daily portfolio value equals cash plus the value of open positions.
- Entry dates occur after the corresponding signal dates.
- Exit dates occur no earlier than 90 calendar days after entry.
- The final portfolio valuation occurs on September 30, 2026.

The final execution reported:

| Check | Result |
|---|---|
| Nonnegative cash | Passed |
| Portfolio accounting identity | Passed |
| Test end date | September 30, 2026 |
| Maximum stale-price positions | 1 |
| Stale open positions at test end | 0 |
| Missing-entry events | 0 |

### Hand-verifiable accounting example

The following is an **illustrative calculation**, not a claim about a specific historical trade.

Suppose the strategy starts with $1,000,000 and buys ten stocks, allocating $100,000 to each.

For one stock purchased at an adjusted price of $50:

- Shares purchased: $100,000 / $50 = 2,000 shares.
- Cash spent: $100,000.
- If the stock is sold after maturity at $55, proceeds are 2,000 × $55 = $110,000.
- Profit is $10,000.
- Return is 10%.

Immediately after buying all ten positions, assuming every order fills and no fees:

- Cash = $0.
- Holdings value = $1,000,000.
- Total portfolio value = $1,000,000.

This illustrates the accounting identity used in the backtester. The actual historical accounting checks are recorded in `backtest_accounting_checks.csv`.

## 7. Data quality and limitations

Several important limitations affect interpretation:

- Some historical adjusted prices contain extreme anomalies.
- Training targets require available future exit prices, potentially excluding securities that disappear.
- Missing daily stock prices may require carrying forward the last available quote for valuation.
- Stale valuations may not reflect actual recoverable proceeds after a suspension or delisting.
- The strategy is concentrated in ten stocks.
- Transaction costs and slippage are assumed to be zero.
- The model showed weak validation-period predictive performance.

These limitations mean that the reported historical performance should not be interpreted as an executable investment result without further data-quality analysis.

## 8. What did not work well

The Ridge model's validation correlations were slightly negative, and the strategy underperformed the S&P 500 benchmark while experiencing a larger maximum drawdown.

Changing Ridge regularization across the tested values did not substantially improve validation performance.

## 9. Possible improvements

Potential improvements include adding carefully timestamped fundamental features, improving treatment of delisted securities, imposing diversification limits, testing transaction costs, and evaluating alternative models during the development period.

## 10. Attribution

The project uses the course-provided financial database and assignment specifications. ChatGPT assisted with code generation, debugging, backtest design, and documentation. The reported performance figures come from the executed project scripts.
