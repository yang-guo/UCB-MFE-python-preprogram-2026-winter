# HW4: Stock Prediction and Trading Strategy Backtest

## How to run the analysis
1. Ensure the `equities_top2000.duckdb` database is located in `../../../Lectures/modeling/data/` relative to this directory.
2. Select the `.venv` kernel (which includes `duckdb` and `scikit-learn`).
3. Run all cells in `hw4_trading_strategy.ipynb`.

## 1. Features, Model, and Validation
- **Features:** I engineered three trailing features at signal time (close of day *t*): 20-day momentum, 60-day momentum, and 20-day return volatility.
- **Universe Alignment:** To strictly prevent look-ahead bias, a stock is only eligible for scoring and purchase if it is confirmed to be in the `universe_membership` top 2,000 for the quarter encompassing the entry session (close of *t+1*).
- **Model:** A regularized linear regression (`Ridge` with alpha=10.0) was chosen for its robustness to collinear momentum features.
- **Validation:** The model was trained purely on development data (entries starting 2010, exits completing by December 31, 2015). Target variables were clipped during training to mitigate the impact of historical split-adjustment artifacts.

## 2. Trading Strategy and Accounting
- **Rules:** Starting with $1,000,000, the strategy allocates available cash equally among the top 20 predicted stocks. A flat 10 bps transaction cost is applied.
- **Timing Constraint:** Signals are generated at the close of session *t*. Fills occur at the close of session *t+1*. 
- **Holding Period:** Positions are held for exactly 90 calendar days. If a stock is halted or delisted and misses 5 consecutive trading sessions, it is liquidated at its last known mark.
- **Verification:** The backtester natively checks accounting integrity daily. My run confirmed 0.00 max daily accounting error, no negative cash balances, and strictly validated 90-day holding periods for 1,342 closed trades (including 11 successful stale/delisted exits).

## 3. Results and Findings
| Metric | Ridge Strategy | S&P 500 Benchmark |
| :--- | :--- | :--- |
| **Final Value** | $1,398,796.35 | $4,402,746.78 |
| **CAGR** | 3.25% | 15.17% |
| **Sharpe Ratio** | 0.20 | 0.77 |
| **Max Drawdown** | -58.70% | -33.79% |

**Analysis:**
The strategy significantly underperformed the S&P 500 benchmark. The simple momentum/volatility factors trained on a 2010–2015 regime failed to capture the highly concentrated, mega-cap tech rally that dominated the 2016–2026 test period. The strategy's lack of concentration (distributing capital equally among 20 names) and forced 90-day turnover led to missing out on prolonged single-name winners, while transaction costs and the -58.70% maximum drawdown heavily dragged down the CAGR.

**Future Improvements:** 
I would incorporate fundamental trailing features (like P/E or operating margin) available strictly before the selection date, use a non-linear tree-based model (like `HistGradientBoostingRegressor`) to capture feature interactions, and implement a stop-loss mechanism rather than a rigid 90-day hold.
