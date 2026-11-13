\# HW4: Stock Prediction and Trading Strategy



\## Files

\- `HW4.ipynb`: Data preparation, feature engineering, Ridge Regression model, validation, and backtest.



\## Methodology

\- Used 18 fundamental and price-based features.

\- Trained a Ridge Regression model to predict 90-calendar-day stock returns relative to the cross-sectional market median.

\- Used data from 2010–2015 for model development and validation, with the required buffer period. The final test period was April 1, 2016–September 30, 2026.

\- Selected the top 1,250 stocks each month based on predicted relative returns.

\- Used equal-weighted allocations across monthly cohorts, with each position held for 90 calendar days.

\- Started with $1,000,000 and applied 5 basis points of transaction costs per trade.



\## Results



| Metric | Ridge Strategy | S\&P 500 Total Return |

|---|---:|---:|

| Final value | $2,330,297 | $4,402,747 |

| Cumulative return | 133.03% | 340.27% |

| CAGR | 8.39% | 15.17% |

| Maximum drawdown | -38.75% | -33.79% |

| Daily Sharpe ratio | 0.431 | 0.773 |



The strategy underperformed the benchmark by $2,072,450 in final portfolio value.



\## Findings and Limitations

The strategy generated a positive return but did not outperform the S\&P 500 Total Return benchmark. Its maximum drawdown was larger and its Sharpe ratio was lower.



The analysis used one Ridge Regression specification and a limited set of 18 features. Model selection relied on the 2015 validation period. Transaction costs were modeled as fixed fees and did not separately account for market impact or bid-ask spreads.



The final backtest also contained open positions with maturity dates on or before the test end date. This unresolved maturity check is a limitation of the reported results and should be investigated before treating the final valuation as fully verified.



\## Accounting and Timing Checks

The notebook includes checks for cash reconciliation, fee reconciliation, next-session execution, sales occurring on or after maturity, and buy/sell/open-lot reconciliation. The remaining-lot maturity check did not pass; see the limitation above.



\## Reused Code and AI Assistance

The implementation adapts concepts from the course lecture code, including the order-management system. ChatGPT assisted with debugging, code development, model experimentation, and report drafting. The submitted code and results were reviewed and run by the student.

