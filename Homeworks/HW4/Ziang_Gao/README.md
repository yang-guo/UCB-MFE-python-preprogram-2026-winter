# HW4: Stock return model, own backtester, and the S&P 500

Author: Ziang Gao

I predict the 90-day return of US stocks, buy the stocks with the best predictions in my own backtester, and compare the result with $1,000,000 in the S&P 500 total return index (`^SP500TR`). The main work and all outputs are in `hw4.ipynb` (round 1). A second round of changes is in `hw4_round2.ipynb` (see "Refinement round 2" below). This file explains how to run everything and what I did.

## Result in short

The strategy did **not** beat the benchmark. Test period: April 2016 to September 2026.

| Metric | Strategy | S&P 500 TR |
|---|---|---|
| Final value | $3,139,813 | $4,402,747 |
| CAGR | 11.5% | 15.2% |
| Maximum drawdown | -30.3% | -33.8% |
| Sharpe ratio (2% risk-free) | 0.58 | 0.77 |

The strategy is $1,262,934 behind. The chart is in `cumulative_return.png` and in the notebook. The largest limit is that 42 lots (21% of the final value) are in stocks that stopped trading. I value them at their last price. If they were worth zero, the CAGR would be 9.0%. The notebook has the full discussion of risks, limits, failed ideas, and next steps.

**Round 2** (two refinements, described below) did not beat the benchmark either, and it did not clearly improve on round 1. Round 2 ended with $2,922,679 (CAGR 10.8%) against $3,139,813 for round 1 and $4,402,747 for the S&P 500.

## What I did

| Step | Choice |
|---|---|
| Data | `equities_top2000.duckdb`. Stocks come from the quarterly top 2,000 list that was known on each date. Stocks that later disappear stay in the data. |
| Features | Price features (momentum over 21, 63, and 252 days, volatility, drawdown), financial ratios from the latest statement that was already filed (filing date plus one day), and two features of my own (momentum minus the sector median). |
| Target | Return from the next session's close to the first close on or after 90 calendar days. |
| Screen | Skip stocks with extreme recent price behavior (3.2% of rows), because of data errors. The rule uses only past prices. |
| Validation | Train on earlier years, validate on 2013, 2014, and 2015. Only outcomes that had ended are used. Models compared by the mean daily rank IC. |
| Model | Ridge regression on momentum and financial features. It had the best mean rank IC on the development years (0.015, which is small and noisy). The target is clipped at the 0.5% and 99.5% training quantiles. |
| Strategy | Every 5 sessions, buy the top 10 scores in equal parts. One entry spends 10% of portfolio value. One stock is at most 5%. Hold 90 days. The settings come from the development years only. |
| Final fit | One fit on signals up to December 2015 whose outcomes ended before 2016-03-31. The model is not retrained. |
| Test | One run, April 2016 to September 2026. |

## How to run

Open a terminal in the repository root. Then run:

```
uv sync --locked
```

Put `equities_top2000.duckdb` from bCourses in `Lectures/modeling/data/`. Open `Homeworks/HW4/Ziang_Gao/hw4.ipynb` and choose the `.venv` kernel. Run all cells. The first run builds the data and saves it in the `cache/` folder (about 30 seconds). The folder `cache/` is not committed to Git. The saved outputs in the notebook come from one full run.

For round 2, open `hw4_round2.ipynb` the same way and run all cells. The first run builds a second data file, `cache/full_panel_v2.pkl` (about 2 minutes, not committed to Git). The whole notebook takes about 6 minutes. It checks that it reproduces the round 1 final value ($3,139,813) before it shows the new results.

## Files

| File | Job |
|---|---|
| `hw4.ipynb` | Round 1. Data, validation, strategy choice, the final test, accounting checks, baselines, and findings. It stays unchanged as a record. |
| `hw4_round2.ipynb` | Round 2. The risk model, the position-aware rule, the comparison of four variants with the S&P 500 and round 1, accounting checks, and findings. |
| `hw4_features.py` | Builds the data in batches and the features. It reuses the lecture functions from `Lectures/modeling/course_data.py`, which are cited in the file. Round 2 adds size, liquidity and price-level features and the `stuck` outcome flag (`round2=True`). |
| `cumulative_return.png`, `results_summary.csv`, `results_baselines.csv` | Round 1 chart and result tables. |
| `cumulative_return_round2.png`, `results_round2.csv` | Round 2 chart and result table (four variants and the S&P 500). |
| `backtester.py` | My own backtester. It keeps the books, replays the market day by day, and computes the metrics. |
| `test_backtester.py` | 24 tests. Each test uses a small made-up market with numbers that can be checked by hand. |

## How to run the tests

Open a terminal in the repository root. Then run:

```
uv sync --locked
cd Homeworks/HW4/Ziang_Gao
uv run --with pytest python -m pytest -q
```

All 24 tests must pass. The project does not list `pytest` as a dependency, so `--with pytest` adds it for this one run.

## What the backtester does

A backtest replays history. It pretends that we followed a trading plan in the past and tracks what the portfolio would have been worth each day.

The work is split into three layers.

| Layer | Job |
|---|---|
| Model | Predicts returns and gives each stock a score. This part is in the notebook. |
| Strategy | Decides what to buy and how much. It only proposes dollar orders. It is the class `TopNStrategy` in `backtester.py`. |
| Backtester | Enforces the rules of the market and does the accounting. It checks every order, fills at the close, charges fees, sells after 90 days, and tracks cash and value. |

Each day the backtester gives the strategy one row of scores (from the previous session) and a small view of the portfolio (cash, value, and exposure per stock, all from the previous close). The strategy answers with a dictionary of dollar amounts. The backtester refuses an order that is negative or larger than the known cash. A strategy cannot break the rules of the market.

### Rules of the world (the backtester)

| Item | Rule |
|---|---|
| Starting capital | $1,000,000. Long only. No borrowing. No leverage. |
| Signal | Made after the close of session t-1, with data through that close only. |
| Entry | At the close of session t, the next session. |
| Holding period | 90 calendar days. |
| Exit | At the close of the first session on or after day 90 that has a quote. |
| Cash used for orders | The cash known at the previous close. Cash from a sale on day t is not used for purchases on day t. This way cash is never spent twice. Orders above that cash are refused. |
| Fees | 5 basis points (0.05%) on each purchase and each sale. The fee on a purchase is paid out of the dollar budget. |
| Cash interest | None. |
| Prices | Adjusted close prices. They already include dividends, so dividends are never added again. |
| Missing price for a held stock | The portfolio keeps the last known price for valuation. The day is counted as a stale mark in the events table. The program never invents a sale price. |
| Missing price on a sale day | The sale waits for the next session with a quote. The delay is written in the events table. |
| End of test | Open positions are valued at the last close. They are not sold, and no later price is used. |

These rules are in the class `Config` (cash, fee, holding days) and in `run_backtest`.

### Choices of the strategy (`TopNStrategy`)

| Item | Value |
|---|---|
| Number of stocks | `top_n` scores in each entry, in equal parts. The class default is 20. The notebook sets 10, chosen on the development years. |
| Size of one entry | At most `cohort_fraction` of portfolio value, and never more than the known cash. The class default is 8%. The notebook sets 10%. |
| Cap per stock | One stock may not exceed 5% of portfolio value, all its lots together. |
| Entry frequency | Every 5 sessions. |
| Rebuy wait (round 2) | `rebuy_wait_days`. When it is a number, the strategy skips a stock whose newest open lot is younger than that many calendar days, and it takes the next best score instead. `None` (the default) turns the rule off. |

These values are my assumptions. They are set in one place when the strategy object is made. I tune them on the development period only.

### One session, step by step

For each market session the function `run_backtest` does these five steps in this order.

1. **Ask the strategy.** Give it the scores of the previous session and the portfolio at the previous close. It returns dollar budgets. The backtester checks them.
2. **Mark.** Update the last known price of every stock that we hold.
3. **Sell.** Sell every lot that has reached 90 calendar days, if the stock has a quote today.
4. **Buy.** Book the approved orders at today's close. Skip a stock that has no quote today and write the skip in the events table.
5. **Record.** Save cash, holdings, total value (NAV), number of open lots, fees, and traded amount for the day.

### Main pieces of code

| Name | Meaning |
|---|---|
| `Config` | The rules of the world. These are cash, fee, and holding days. |
| `Lot` | One purchase, with its units, entry price, entry fee, and due date. |
| `Ledger` | Cash, open lots, and the trade log. Its methods `buy` and `sell` do the accounting and refuse an impossible trade. |
| `PortfolioView` | What the strategy may see. It holds cash, value, exposure, and (round 2) the days since the newest purchase of each stock, all from the previous close. |
| `Strategy` | The base class. A strategy has one method, `orders`, that returns dollar budgets. |
| `TopNStrategy` | My strategy. It buys the top scores in equal parts, with the size limits above. |
| `check_orders` | Refuses orders that are negative or larger than the cash. |
| `run_backtest` | The daily loop described above. It takes the prices, the scores, and a strategy. |
| `benchmark_curve` | Value of $1,000,000 in the S&P 500 total return index, bought at the first test close and held. |
| `performance` | Final value, CAGR, maximum drawdown, and Sharpe ratio. |
| `compare` | The same metrics for the strategy and the benchmark, side by side. |
| `audit` | Checks the books after a run. |

### Metrics

- **Final value.** Cash plus open positions at the last close.
- **CAGR.** (final value / start value) raised to the power of 1 / years, minus 1. Years are the number of calendar days divided by 365.25.
- **Maximum drawdown.** The largest fall from a previous peak, as a fraction of that peak.
- **Sharpe ratio.** Uses daily returns. The daily risk-free rate is 1.02^(1/252) - 1, which matches a 2% annual rate. Sharpe is the square root of 252, times the mean daily excess return, divided by the standard deviation of daily returns.

## How the accounting is checked

The function `audit` runs after a backtest and stops on the first failure. It checks that

1. NAV equals cash plus holdings on every day.
2. Cash equals the start cash plus all trade cash flows.
3. Cash is never negative.
4. Every purchase comes after its signal date.
5. Every sale is on or after day 90, and every lot is due exactly 90 days after its purchase.
6. Every lot is sold once, or is still open at the end.

## What the tests cover

| Test | What it proves |
|---|---|
| `test_one_lot_by_hand` | Units, fees, cash, and profit for one purchase and one sale match a hand calculation (about $9,780 profit). |
| `test_cannot_spend_cash_twice` | A second purchase without cash is refused. |
| `test_buy_needs_earlier_signal_and_sale_needs_maturity` | A same-day signal is refused. A sale on day 30 is refused. |
| `test_entry_next_session_exit_at_90_days_and_cash` | The buy happens the session after the signal. The sale happens on the first session on or after day 90. Final cash is exactly $1,100,000 in the example. |
| `test_no_lookahead_in_signals` | A stock that scores best only on the entry day is not bought. |
| `test_open_positions_are_valued_at_the_end_and_not_sold` | Open lots count in NAV and are not sold. |
| `test_missing_quote_on_due_date_delays_the_sale_without_inventing_a_fill` | The sale moves to the next quote and the delay is logged. |
| `test_stale_mark_keeps_last_price_for_nav` | A missing quote keeps the last mark and is counted. |
| `test_entering_every_day_never_overspends_and_audit_passes` | Daily entries with full cash use never make cash negative. |
| `test_name_cap_limits_one_stock` | One stock gets at most 5% of NAV (in the example, $50,000). |
| `test_backtester_refuses_orders_larger_than_cash` | A strategy that asks for $2,000,000 with $1,000,000 in cash is refused. |
| `test_backtester_refuses_negative_orders` | A strategy cannot short sell. A negative order is refused. |
| `test_strategy_sees_only_the_previous_session_row` | On each day the strategy receives only the score row of the previous session. |
| `test_any_strategy_plugs_into_the_same_backtester` | A different strategy runs through the same backtester and passes the audit. |
| `test_max_drawdown_by_hand` | A series 100, 120, 90, 110 gives a drawdown of -25%. |
| `test_cagr_by_hand` | CAGR matches the formula for a doubling over 366 days. |
| `test_sharpe_by_hand` | Sharpe matches a direct calculation from known returns. |
| `test_benchmark_buys_at_first_test_close` | The benchmark stays at $1,000,000 until the first test close, then follows the index. |
| `test_default_strategy_ignores_position_age` | Without the wait rule, the old behavior is unchanged. |
| `test_stock_bought_yesterday_is_skipped_and_the_next_best_takes_its_place` | With the wait rule, a stock bought yesterday is skipped, the next best score takes its place, and the whole budget is still spent. |
| `test_stock_can_be_bought_again_when_its_newest_lot_has_aged` | A stock with a newest lot of 44 days is skipped. At 45 days and later it can be bought again. |
| `test_ledger_reports_the_age_of_the_newest_lot` | The ledger reports the age of the newest open lot of each stock. |
| `test_full_run_does_not_stack_a_stock_bought_five_sessions_ago` | In a full run, three stocks are bought one after the other and none is stacked. |
| `test_default_run_is_unchanged_by_the_new_field` | The new field in `PortfolioView` does not change an old result. |

I also ran the backtester once on the class database for 2015 with a simple momentum score. All six audit checks passed. That run only tests that the code works on real data. Its return is not a result of this homework.

## Limits

- The tests use made-up prices. They test the logic, not the performance of a strategy.
- Fills happen at the close price. The model has no bid-ask spread, no market impact, and no limit on trading volume. The 5 basis point fee is only a rough cost.
- Units are fractional.
- If a held stock stops trading, the program keeps its last price and flags it. The data has no reliable delisting payout, so the real loss or gain on such a stock is unknown. Round 2 shows that many of these stocks may be takeover targets, where the real payout is probably close to the last price.
- `TopNStrategy` uses equal weights. There is no optimization of position sizes.

## Refinement round 2

**Why.** In round 1, 42 lots (21% of the final value) were still open after their 90-day due date because the stock had no more quotes. I also saw that the strategy could buy the same stock again a few days after it had bought it. I made two changes. `hw4_round2.ipynb` shows the work and `hw4.ipynb` stays as it was.

**A warning that applies to every number in this section.** Both changes came from looking at the round 1 test result. So the round 2 test is **not** a clean out-of-sample test any more. I limited the damage in three ways. I chose every setting that can be tuned on the development years only (2010 to 2015). I fixed the settings that cannot be tuned (the risk cut-off and the retraining dates) before the test and did not change them. I ran each variant once. Read the result with this in mind.

### Change 1: avoid stocks that stop trading

| Item | What I did |
|---|---|
| New features | `log_mcap` (market cap at the quarterly selection date), `log_dvol_63` (63-day average dollar volume), `turnover_63` (dollar volume divided by market cap), and `log_price` (log of the raw closing price). All are known at the signal date. |
| New outcome flag | `stuck` is True when a stock has an entry quote but no exit quote within 5 sessions after the due date. It is an outcome. It is used only as a training label, and only for signals whose outcome was known before the fit date. |
| Risk model | A logistic regression (`C=0.1`) with size, liquidity, price level, volatility, drawdown, momentum and balance sheet features. It predicts the chance that a signal ends up stuck. |
| Update rule | Fit on 2016-03-31 and again on January 1 of each year, using only outcomes known before that date. Fixed in advance. |
| Gate | On each signal date, the riskiest 25% of eligible stocks get no score and cannot be bought. The 25% was fixed in advance and not tuned. |

Three things I found.

1. **The development years hold almost no stuck events.** About 0.01% of signals in 2010 to 2015, against 0.7% to 1.7% a year from 2021. Only 45 of the 3,849 stocks have a last quote before 2021. The data keeps price history for recent delistings only (the data guide says delisted coverage is incomplete). So the risk model cannot learn from the development years alone, and I retrain it every year.
2. **Market cap did not improve the return model.** Size and liquidity features did not raise the validation rank IC. The best accepted set is still round 1 (mean IC 0.015). Sets that include the price level reached 0.045, but they buy penny stocks (the median price of their 2015 picks was $2.60 against $33.77 for all stocks). Penny stocks that vanished before 2021 are missing from the data, so the survivors look good. I excluded the price level from the return model. I made this decision after I saw the numbers, and I list it here as a judgment call. The size and liquidity features are used in the risk model.
3. **The risk model has some skill.** Out of sample, from 2021 on, its AUC is 0.62 to 0.76, and the riskiest 25% holds about 50% to 64% of the stuck events that came later. In the early years the numbers are noisy because there are few events.

### Change 2: position-aware rebuy rule

The strategy now sees how many days ago it bought the newest lot of each stock. With `rebuy_wait_days=30` it skips a stock whose newest lot is younger than 30 days and takes the next best score. A stock can be bought again once its newest lot has aged, which is close to the end of the holding period. The 5% cap per stock still applies to the total exposure.

I chose 30 days (out of 30, 45 and 60) on the development years by the best Sharpe ratio. I decided before the run that the rule is on, because I wanted it. The development years did not support it: the best setting without the rule had a Sharpe ratio of 0.78, and the best setting with the rule had 0.60.

### Result (test period April 2016 to September 2026, one run per variant)

| | Final value | CAGR | Max drawdown | Sharpe | Lots past due and still open | Final value if those lots were worth zero |
|---|---|---|---|---|---|---|
| Round 1 (as submitted) | $3,139,813 | 11.5% | -30.3% | 0.58 | 42 (21.2% of value) | $2,475,381 |
| Rebuy rule only | $3,336,906 | 12.2% | -35.0% | 0.59 | 26 (16.2%) | $2,795,326 |
| Risk gate only | $2,876,717 | 10.6% | -30.8% | 0.54 | 30 (14.7%) | $2,452,474 |
| **Round 2 (both)** | **$2,922,679** | **10.8%** | **-35.1%** | **0.53** | **22 (15.4%)** | **$2,472,412** |
| S&P 500 TR | $4,402,747 | 15.2% | -33.8% | 0.77 | | |

Round 2 is $1,480,068 behind the S&P 500 (CAGR 4.4 points lower) and $217,134 behind round 1. It was above round 1 in 6 of 11 calendar years and above the S&P 500 in 2 of 11. The chart is `cumulative_return_round2.png`.

### How to read the result

- **The two changes did not give a clear improvement.** The risk gate reduced the stuck lots (42 to 30) but lowered the final value. The rebuy rule alone raised it by $197,093, but the development years did not support it and its drawdown is larger.
- **Most of the rebuy gain probably comes from less idle cash, not from better stock choice.** In round 1 the 5% cap often cut an order, and the unused money stayed in cash (average cash share 12.4%). With the rebuy rule the budget goes to other stocks, so the cash share falls to 5.5%, and the strategy buys 839 different stocks instead of 425. More money in a rising market raises the return and the drawdown.
- **Stuck stocks are probably not mostly failures.** Several names in the stuck list (for example HES and SWN) are, as far as I know, takeover targets, and the data guide says an inactive flag can also mean a rename. A takeover usually pays close to the last price, so valuing those lots at the last price is probably not far off, and the zero-value case is too harsh. The real cost is idle money, because a lot with no quote cannot be sold. On average 7.3% of the round 1 portfolio sat in such lots (4.1% in round 2). That is a drag, but it probably explains only a small part of the gap to the S&P 500.
- **The gap to the S&P 500 is still the weak return signal.** The return model is the same as in round 1. Round 1 already showed that its signal is weak.
- **The differences between variants are small.** In round 1, random scores earned $2.68M and the model $3.14M. Differences of $200,000 between variants are in the same range as that noise. I did not measure the noise for round 2.

### Limits of round 2

- The test is not clean (see the warning above).
- The data holds survivors in the development years (see Change 1, finding 1). This affects every model that trains on those years, including the return model.
- Excluding the price level from the return model was a judgment call made after I saw the validation table.
- The cut-off of 25% and the risk model settings were not tuned. Other values may do better or worse.
- The zero-value case is a bound. The data has no payout table, so I did not invent a sale price for a stuck lot.
- The share of frozen money uses the last known price from the due date on, which is an approximation.

### Checks of round 2

The audit (six accounting checks), an independent NAV rebuild on three days, a check that no repeat purchase breaks the 30-day rule (3,532 repeat purchases, smallest gap 30 days), and a check that no purchase has a risk rank above the limit (4,420 purchases) all pass. The notebook also reproduces the round 1 final value to the dollar before it shows round 2.

## Credits and AI assistance

The data and model steps reuse the lecture functions in `Lectures/modeling/course_data.py` (loading, price features, financial joins, membership, labels, folds, and the preprocessing pipeline). `hw4_features.py` lists each one.

The backtester design uses ideas from Lecture 8.1: a strategy that is separate from the backtester, a ledger of separate lots, fee-inclusive dollar budgets, a daily loop, and an audit of the books. I did not copy code from `Lectures/modeling/backtest.py`.

Claude, an AI assistant from Anthropic, helped write `backtester.py`, `test_backtester.py`, `hw4_features.py`, `hw4.ipynb`, and `hw4_round2.ipynb`. The two round 2 changes were my requests. Claude proposed the design and the code for them and checked the results. It read Lecture 8.1 and the lecture backtester as design references and did not copy the lecture backtester.
