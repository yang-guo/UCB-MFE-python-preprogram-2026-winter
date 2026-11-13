# HW4: 90-day return model and backtester

The model is a histogram gradient-boosting regressor. It is fit once, on April 1, 2016, and then held through the final test. On completed test holds it has a mean daily Spearman rank IC of **0.041**. The 30 names with the highest predicted return beat the same-day eligible universe by **1.6 percentage points** over the 90-day hold, before costs. That spread is an overlapping signal diagnostic, not a portfolio NAV.

## Run

From this directory, with the full database at `Lectures/modeling/data/equities_top2000.duckdb`:

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python model.py
.venv/bin/python backtest.py
```

`model.py` builds and scores the return model. `backtest.py` reruns the hand accounting checks, then the test-period book. The cached panel stays in `outputs/` (gitignored). Tables and charts are in `results/`. To rerun only the hand checks, use `.venv/bin/python test_backtest.py`. To rebuild the accounting table from the saved book, use `.venv/bin/python backtest.py --evidence`.

## Target

A score uses the close on signal date `t`. The position enters at the next market session’s close and exits at the first quoted close on or after 90 calendar days, provided that quote is within five market sessions of the due date. The target is the adjusted-close return over that hold. Adjusted prices already include dividends, so the label does not add them again.

The class-database check matches `course_data.prepare_panel` on 59,257 rows (largest feature gap about `5e-14`). For AAPL on August 31, 2012 the entry is September 4 and the exit is December 3, a return of −12.8%. See `results/label_timing_check.csv`.

## Features

All inputs are known at the signal close.

| Feature | What it measures |
|---|---|
| `momentum_5`, `momentum_21`, `momentum_63`, `momentum_126` | Trailing adjusted-price returns over 5, 21, 63, and 126 market sessions |
| `momentum_252_21` | Return from 252 sessions ago to 21 sessions ago (12-minus-1 momentum) |
| `volatility_21`, `volatility_63` | Annualized volatility of daily returns |
| `drawdown_252` | Distance from the 252-session high |
| `volume_ratio_21` | Today’s volume relative to its own 21-session average |
| `large_cap_score` | Capitalization rank inside the entry quarter’s membership list (1 = largest) |
| `net_margin`, `operating_roa`, `debt_to_assets`, `cash_to_assets`, `fcf_to_assets`, `accruals_to_assets` | Latest quarterly filing, usable from the day after `filing_date` |
| `filing_age_days` | Days since that filing became usable |
| `sector_relative_momentum` | 63-session return minus the same-day sector median |
| `sector_profitability_rank` | Percentile rank of operating ROA inside the sector that day |
| `market_momentum_63`, `market_volatility_63` | Same-day trailing return and volatility of `^SP500TR` |
| `sector` | Snapshot sector, one-hot encoded with unknown sectors ignored |

Financial ratios are blank once the fiscal period is more than 550 days old. Peer stats use only names that are eligible that day, and they are computed before any row is dropped for an unfinished label. A name stays in the panel if it later stops trading. Membership is the list for the entry quarter, and the selection date is always on or before the signal.

## Why this model

`HistGradientBoostingRegressor` fits a sequence of small trees, so a split can let momentum behave differently at high volatility, or let accruals matter only in some sectors. A linear model has one weight per input. Missing filings stay missing: the tree sends them down their own branch. Median-filling them would make an unreported ratio look like a typical firm. Early stopping is off, because sklearn’s internal holdout is a random slice of rows rather than a later period. The year folds choose the tree size instead.

The frozen settings are 120 iterations, at most 63 leaves, at least 100 rows per leaf, learning rate 0.05, and L2 regularization 0.1 (`random_state=42`). Each fit clips numeric features and training targets at the training 1st and 99th percentiles. Reported errors use the raw targets. Training rows are the first market session of each month. Daily rows would repeat the same 90-day outcome. Each date receives equal weight.

On the class of models, three sizes were compared by mean daily rank IC in 2013 and 2014, with a new fit before each year:

| Size | Mean rank IC | Mean RMSE |
|---|---:|---:|
| Shallow | −0.058 | 0.266 |
| Medium | −0.041 | 0.277 |
| Deep | −0.035 | 0.276 |

Deep was the best of that screen. Every annually refit tree had a negative rank IC, so the next choice was whether to refit at all. For the deep tree, one fit before 2013 scored **+0.027** average rank IC across 2013–2014. Refitting before 2014 scored **−0.035**. The test therefore uses a single fit and keeps the prediction as the score (higher means more attractive). A static Ridge (`alpha=100`, same features, lecture preprocessing) scored **+0.007** on those two years. The tree’s advantage is the ranking, not the level: its RMSE is higher than Ridge and higher than a clipped training-mean forecast. Squared error is a weak selection metric here because a few recorded returns are not economic. AAM’s 2012 holding return is above 100,000 in the file. Clipping the training labels keeps that point from setting the forecast, and the stock is not deleted.

## Validation and the frozen rule

| Period | How it is used |
|---|---|
| 2013–2014 | Choose tree size, then choose one fit versus annual refits |
| 2015, exits through March 31, 2016 | Confirmation only. It does not change the choice |
| January–March 2016 | Buffer. Labels that finish here may train the April 1, 2016 fit. No test trades |
| April 1, 2016–September 30, 2026 | Final test of the frozen model |

The April 1, 2016 fit uses 141,435 month-start rows from January 4, 2010 through December 1, 2015. The latest training exit is March 1, 2016, before the first test signal. The model is not updated during the test.

| Year | Role | Rank IC | Top-30 excess | RMSE |
|---|---|---:|---:|---:|
| 2013 | Selection | −0.019 | +1.2 pp | 0.365 |
| 2014 | Selection | +0.074 | +0.9 pp | 0.218 |
| 2015 | Confirmation | +0.064 | +1.5 pp | 0.217 |

## Final test

Eligible names are scored every session from April 1, 2016 through September 30, 2026. Metrics use holds that had already finished by the sample end. 158,759 eligible rows are left unscored because the exit quote is missing or still in the future. Across 2,577 dates and 5,071,756 completed rows:

| Metric | Value |
|---|---:|
| Mean daily Spearman IC | 0.041 |
| Top-30 excess 90-day return | +1.58 pp |
| Top-minus-bottom quintile spread | +1.48 pp |
| RMSE | 0.246 |
| MAE | 0.157 |
| R² | −0.035 |

R² below zero means the level forecast loses to predicting the test sample’s own average return. The rank correlation is positive in 9 of 11 test years (the exceptions are 2023 and 2024, both near zero). The top-30 excess is negative in 2016, 2019, 2021, and 2023, and large in 2020 (+9.4 pp) and 2025 (+5.8 pp).

## Backtester

`backtest.py` keeps its own cash balance, one lot per purchase, and a daily value. It does not call the lecture backtester. The fee is reserved inside the buy budget the same way as `Lectures/modeling/backtest.py`.

The book starts at $1,000,000, long only. Cash earns nothing. Each buy and each sale pays 5 basis points. Prices are adjusted closes, so dividends are not added again. A decision uses the prior session’s close and the model score from that close. The fill, if the stock has a positive adjusted close that day, is the next session’s close. The lot is due 90 calendar days later and is sold on the first later session that has a real quote. A missing quote is never replaced with a price.

Once a month, on the first session of the month, the book buys up to 30 names with the highest scores, skipping anything already held. The cohort spends at most one third of portfolio value, and at most the cash on hand, split equally. A name’s purchase is also capped at 4% of portfolio value. The first fill is April 1, 2016, decided from the March 31, 2016 close. Sale proceeds are available for the next order, not for a buy planned at the same close.

## Accounting and timing

`test_backtest.py` checks nine cases by hand before the full run:

- The 5 basis point fee is reserved inside the buy budget, so cash ends at zero when the budget is the whole balance.
- A sale returns cash at the sale price.
- Two orders sized from the same cash cannot spend more than that cash.
- A lot that matures on an entry day does not fund a buy decided at that same close.
- A missing entry quote is rejected. The budget stays in cash, and no price is invented.
- A due date that is not a session sells at the next session that has a quote.
- An open lot at the sample end is marked at the last close inside the sample.
- A 4% name cap leaves the unused budget in cash.
- A two-name path keeps NAV equal to cash plus holdings and ending cash equal to the fill log.

The saved test book passes the same identities. `results/accounting_check.csv` records them: the book starts at $1,000,000, cash never goes negative, the largest NAV break is numerical noise, ending cash matches the fill log, every buy is signaled on an earlier session, and every due date is the entry date plus 90 calendar days. The first signal is March 31, 2016 and the first fill is April 1, 2016. History stops on September 30, 2026. Buys minus sales equals the 87 open lots.

Label timing is checked against the lecture panel on the class database: 59,257 rows, largest feature gap about `5e-14`. AAPL signaled on August 31, 2012 enters September 4 and exits December 3. That check is `results/label_timing_check.csv`. The April 1, 2016 fit’s latest training exit is March 1, 2016, recorded in `results/test_fit_audit.csv`.

| Item | Value |
|---|---:|
| Final value on Sep 30, 2026 | $2,291,208 |
| Cash | $713,929 |
| Open holdings | $1,577,279 |
| Open lots | 87 |
| Buys / sales | 3,658 / 3,571 |
| Fees paid | $53,817 |

The 87 unsold lots are the 87 buys that have not yet matured or have no exit quote. Sixty are inside their 90-day hold. The other 27 stopped printing for more than five sessions before a sale could be booked. They stay in the book at their last real close, about **$465,000**, which is 29% of ending holdings. No exit was invented for them. WSTC has been marked at its October 2017 close ever since.

## Benchmark

Both portfolios start from $1,000,000. The index is bought at the April 1, 2016 close of `^SP500TR` and held through September 30, 2026, with no trading cost. The strategy series includes cash and open lots. CAGR uses the actual span from the March 31, 2016 mark, when both values are still $1,000,000, divided by 365.25. The Sharpe ratio uses daily simple returns and a 2% annual risk-free rate converted with 252 sessions. These definitions follow `Lectures/modeling/backtest.py`.

| Metric | Strategy | S&P 500 total return |
|---|---:|---:|
| Final value | $2,291,208 | $4,402,747 |
| CAGR | 8.2% | 15.2% |
| Maximum drawdown | −42.3% | −33.8% |
| Sharpe ratio | 0.39 | 0.77 |

The strategy finishes **$2,111,539 behind** the index, about 6.9 percentage points a year of CAGR. It does not beat the benchmark. The curves are in `results/cumulative_returns.png`.

The index's gain over this decade was concentrated in the largest companies. This book instead buys 30 names a month in equal dollars, skips anything already held, and pays 5 basis points to enter and exit. A rank IC of 0.04 is too small to replace that market exposure. The path is also rougher: the worst decline is 42%, against 34% for the index. About $465,000 of the strategy's ending value is a last close for a stock that stopped trading, so the reported result is the more generous of the two marks and it still trails.

The main limits are that unresolved last prices stay in the strategy value, the index pays no fee while the book does, and the model was chosen for rank IC rather than for this wealth comparison. One test path is not a distribution of outcomes.

## Findings

The frozen book grows $1,000,000 to $2,291,208. The same money in `^SP500TR` grows to $4,402,747. The strategy trails by $2,111,539, about 6.9 percentage points of CAGR, and its worst decline is 42% against 34% for the index. A positive rank IC of 0.04 did not become a better portfolio than owning the market.

Several choices were tried and then dropped:

- Refitting the deep tree before each year. That rule’s 2013–2014 rank IC was −0.035. One fit before 2013 scored +0.027, so the test uses a single fit.
- Picking the model by squared error. The tree’s RMSE is worse than a static Ridge and worse than the mean of the clipped training labels. The portfolio buys a ranking, so the selection metric is rank IC. A raw training mean is unusable: AAM’s recorded 2012 hold is above 100,000.
- Taking the sign from the annual-refit IC. That screen was negative and would have bought the lowest scores. The frozen rule’s own IC is positive, so high scores are the buys.
- Crossing the one-fit rule with the shallow and medium trees. Depth was chosen on the annual screen, then the refit rule was chosen only for the deep tree.

What I would try next is to choose depth and the refit rule together on development-window portfolio value, not on overlapping rank IC. I would also predict the excess return over `^SP500TR`, or keep the book closer to the large-cap end of the membership list, because the index’s gain came from the largest names. A delisting return, once that table exists, should replace the last close for a name that stops trading. About $465,000 of the ending value is still a stale print. Fewer overlapping 90-day cohorts would cut the 5 basis point fee.

## Limits

- Sector labels and financial values are the vendor’s current snapshot. A filing date blocks a not-yet-published report; it does not recover a number that was later restated.
- `large_cap_score` uses the course’s reconstructed capitalization rank, not a certified historical market cap.
- `volume_ratio_21` can spike when a split falls inside the 21-session window.
- Rank IC and the top-30 spread reuse overlapping 90-day outcomes, so they are not a Sharpe ratio and they are not compounded NAV.
- Tree depth was chosen on the annual-refit screen. The one-fit rule was chosen afterward for that depth, and the other depths were not searched again under the one-fit rule.
- The 2013 fold had a negative rank IC. The positive selection average is mostly 2014, and 2015 then agreed.
- Twenty-seven holdings have no quote through their due date. Their last close remains in the September 30, 2026 value. That is a mark, not a fill.

## Sources

Feature timing, the filing join, membership, `QuantileClipper`, and the Ridge pipeline follow `Lectures/modeling/course_data.py`. The idea of reserving the entry fee inside the dollar budget follows `Lectures/modeling/backtest.py`; the ledger in `backtest.py` is separate. The panel is built in DuckDB so the full universe is practical. `model.py --stage verify` checks that the SQL features match the lecture code. This work was implemented with AI assistance in Cursor.
