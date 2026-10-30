# Homework 4: Build a Stock Prediction and Trading Strategy

**Due: November 12, 2026** · Submit a pull request and assign `kevinramlal` and `raghuramkowdeed`.

**Your challenge: build a stock-return model and your own backtester, then try to beat the S&P 500 total-return index.**

You choose the features, model, stock-selection rules, position sizes, and trading frequency. Explain your decisions and show what happened.

## 1. Get the data

Download `equities_top2000.duckdb` from bCourses into `Lectures/modeling/data/`. It contains prices and financials from **2000 through September 30, 2026**, plus the `^SP500TR` benchmark.

The file contains **3,849 stocks** across quarterly top-2,000 selections. Use `universe_membership` to find the candidates for each entry quarter, including stocks now inactive. Keep existing positions when a stock leaves the universe. You may screen candidates further using information available at the time.

Use `equities_class.duckdb` to get your code working quickly; use the full homework database for your final experiment. See the [data guide](../modeling/README.md) for table definitions and known issues.

## 2. Build and evaluate your model

Predict the return over your strategy's **90-calendar-day holding period**. Choose and engineer features you think will help. Use any model and explain why you chose it.

| Period | Dates | Use |
|---|---|---|
| Development | Jan 1, 2010–Dec 31, 2015 | Build features, train, validate, and choose your strategy |
| Buffer | Jan 1–Mar 31, 2016 | Allow earlier return targets to finish |
| Final test | Apr 1, 2016–Sep 30, 2026 | Evaluate your chosen strategy |

Earlier data is available for lookbacks. Choose your own chronological validation approach within development, then settle on your model and strategy before running the final test.

**Avoid look-ahead:**

- Features must use information available when the prediction is made. Financial results become available after publication, not at fiscal period end.
- At every fit, use only targets whose holding periods have already ended. This still matters after the buffer.
- Fit any learned preprocessing on training data only. Do not remove stocks because you know they later disappear.

**Retraining is optional.** You may update the model during the test period using newly completed outcomes. Choose the update rule during development.

## 3. Build your backtester

Write your own implementation. You may reuse lecture data-loading and preprocessing code; cite anything you adapt. Your backtester should track purchases, sales, cash, open positions, and daily portfolio value.

| Rule | Requirement |
|---|---|
| Starting capital | $1,000,000; long only; no borrowing or leverage |
| Entries | Any trading day, at the close; decisions must be made before that fill price is known |
| Holding period | Hold each purchase for 90 calendar days; sell at the close on maturity, or the next market session if closed |
| Cash | Purchases consume available cash; sales return cash. Never spend the same cash twice |
| Test end | Include the value of open positions on Sep 30, 2026; do not use later prices |

If your features use today's close, the earliest entry is the **next session's close**. Align your prediction target with that entry and the 90-day exit.

You decide how many stocks to buy, how much to allocate, and whether to impose concentration limits. State your assumptions about transaction costs, missing prices, and cash interest. Use adjusted prices consistently; do not add dividends again if they are already reflected in returns.

Check the accounting on cases you can verify by hand before trusting the full run. Explain any data gaps that prevent a reliable result; do not invent fills.

## 4. Compare with the benchmark

Compare your strategy with **$1,000,000 invested in `^SP500TR` at the first test close**, held through the same end date.

Show these metrics for **both** portfolios:

- Final value, including cash and open positions
- Annualized compounded return (CAGR)
- Maximum drawdown
- Sharpe ratio from daily portfolio returns, using a 2% annual risk-free rate

Plot both cumulative-return curves on one chart. Report whether you beat the benchmark and by how much. Briefly explain the result, the risks taken, and the main limitations of your analysis.

## 5. Submit

Create `Homeworks/HW4/<First_Last>/` and include:

- [ ] Your code: a notebook, Python files, or both, with instructions to run it
- [ ] A brief explanation of your features, model, validation, and trading strategy
- [ ] The performance comparison and cumulative-return chart
- [ ] Evidence that you checked the backtest's accounting and timing
- [ ] Your findings, including unsuccessful ideas or changes you would try next

Keep databases and large generated files out of Git. Cite reused code and explain the work you submit, including any AI-assisted work.

**Pass/no-pass:** a working backtester, correct evaluation, readable code, and a clear explanation. Beating the benchmark earns bonus recognition and is not required to pass. The course's late-day policy applies.
