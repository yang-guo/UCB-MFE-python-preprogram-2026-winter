# HW4

## Table of Contents

- [Run Instructions](#run-instructions)
- [Citations](#citations)
- [Findings and Takeaways](#findings-and-takeaways)
- [Files](#files)
- [Program Structure](#program-structure)
  - [Load Data](#load-data)
  - [Create Signal Developer](#create-signal-developer)
  - [Create Ideal Positioning Scheme](#create-ideal-positioning-scheme)
  - [Model Assessor](#model-assessor)
  - [Create Portfolio Class](#create-portfolio-class)
  - [Create Orderbook/History Class](#create-orderbookhistory-class)
  - [Create Trading module](#create-trading-module)
  - [Create backtester](#create-backtester)
  - [Create backtest Assesor](#create-backtest-assesor)
- [Core Assumptions](#core-assumptions)
- [Tests and Checks](#tests-and-checks)
- [Not Built / Future Work](#not-built--future-work)

---

## Run Instructions

### Requirements

- Python 3.12.15
- pandas
- numpy
- notebook
- matplotlib
- scikit-learn
- duckdb
- pathlib (`Path`) - part of the Python standard library, no install needed

`equities_top2000.duckdb` must be in this folder (distributed via bCourse, not committed).

### How to Run

From a terminal at the repository root (with the virtual environment activated):

```bash
cd Homeworks/HW4/SCOTT_WESTVOLD
python fama_backtester.py
```

This calls `main()`, which runs the whole pipeline (load → universe membership → clean → signals → ideal weights → signal assessment → trading backtest) and saves two PDF reports to the working directory:

- `Fama French Signal Assessment.pdf` — train and test total return charts and tables, quarterly predicted vs actual returns, and predicted vs actual scatters per signal.
- `Fama-French 5 Factor Sector Neutral Backtest.pdf` — backtest summary table, final value table, total return and active return charts, and the worst bad day.

To run it from Python / Spyder and keep the results:

```python
from fama_backtester import main
main_backtest = main()
```

---

## Citations

1. Data loader sources: Berkeley MFE Preprogram Python Lecture 8 Dataloading Noteook
2. Universe membership logic adapted from MFE Python Lecture 8 / modelling lecture material.
3. Claude Code (Anthropic) was used for implementation: the trading engine (`trading.py`), report functions (`report.py`), and marked functions in `fama_backtester.py`, and some functions in `fama_backtester.py` don't include the specific marker. The ideas and rules are mine; the implementation was Claude's.

---

## Findings and Takeaways

### What were my main choices?

My main choice was to pursue a factor model which actually led to trouble in the regressions. I thought I'd build sector neutral fama french and then just see which factor performed best but then I realized I should use them predictively to some extent. I was glad to see profitability performed well as it rhymes with my understanding of what happened in the 2010s with the success of big tech.

### Which factors worked?

The best factors were large cap - small cap, and profitability. There is some general hindsight bias with using large cap - small cap as we know what happened since 2006 with the S&P. However its an interesting food for thought if to some extent there was overindexing on the factor in the 2010s.

### Did it work better for security selection or sector selection?

I did not end up testing sector rotation but I am very happy I isolated the metrics to be intrasector and benchmark aware weights (using the market caps as a baseline) as fundamental measures can be very different accross sector such as book value. I dont think it wouldve been wise to do cross sector rotation with some of the core fundamentals at least

### Was TC a big drag?

TC was a much smaller drag than I expected. My turnover however was also higher than I expected. Its an interesting balance to hold but I would guess but dont know that the minimum holding periods reduced turnover meaningfully

### Did the umbrella show signs of diversification benefits?

The umbrella looked like an average of the 5 models. It did not show an IR boost above the 5 individually meaning there wasnt the magic of uncorrelated positive alpha signals. But I am glad I looked at a multifactor model

### Any other considerations worth mentioning?

I do think its a good thing to note that we were using a 2000 stock field against the largest 500 benchmark at a time when the largest 500 had outstanding market leading returns. This mightve increased the challenge but its an exercise worth doing because in the real world we wouldn't know that SP500 was about to have an excellent relative run.

---

## Files

| File | Contents |
|---|---|
| `fama_backtester.py` | Main script (`main()`): data loading, cleaning, signals, ideal weights, assessors, backtest, PDF reports |
| `trading.py` | `Portfolio`, `OrderBook`, `Trader` and `Backtest` classes (the trading engine) |
| `report.py` | Backtest report tables and charts used in the backtest PDF |

---

## Program Structure

### Load Data

#### Load Stocks

`load_data` reads prices, fundamentals, sectors, universe candidates (market cap proxy) and the S&P 500 benchmark from `equities_top2000.duckdb`.

#### Load Benchmark

S&P 500 `adj_close` from the `benchmark` table; daily `sp_return` is computed from it.

#### Universe Membership

`load_membership` / `apply_membership` restrict the price panel to the quarterly top-2000 universe. A signal at date t is traded at the next session, so membership is taken from the quarter of the next trading date. Asserts check for duplicate membership keys, look-ahead (selection date after the signal date) and duplicate date-symbol pairs.

### Create Signal Developer

#### Process Data Into As Of Report Date

Fundamentals are available 2 days after their filing date and market caps 2 days after their shares-available date (no look-ahead). Fundamentals are forward-filled for up to 252 trading days. Duplicate date-symbol rows are collapsed to the first.

#### Clean Data

- Market cap: values more than 50x the symbol's expanding mean, or above $10 trillion, are set to NA and forward-filled.
- Prices: a symbol's history is cut once it has 20 NA/zero prices; zeros are forward-filled; one-day returns of 70% or more are dropped as bad prints / unadjusted splits.
- Known bad tickers (RCA, LTCHW) are removed, and dates with fewer than 10 stocks are dropped.

#### Manufacture Signals

FAMA French 5-Factor style signals:

- Book value: (total assets − total liabilities) / market cap
- Small − large: market cap (runs long large caps)
- Profitability: operating income / revenue
- Investment aggression: 252-day rolling capex / total assets
- Beta: rolling 3-year beta to the S&P, inverted (buy low beta, sell high beta)

#### Manufacture Aggregate Signals

- Stocks below 10% of their sector's median market cap are dropped.
- Sector cap-weighted averages of each factor are created.

### Create Ideal Positioning Scheme

#### FAMA French Factor Portfolios

- Position = sector market-cap weight ± a tilt from the signal's rank within its sector, scaled to ±300bps.
- Sector weights are set to each sector's share of total market cap, so the portfolios are **sector neutral** (security selection only).
- Weights are smoothed over 10 days, lagged 3 trading days, positions under 10bps are removed and the rest rounded to 10bp increments, then rescaled to sum to 1.
- Asserts: every portfolio sums to 1 on every date, and there are no duplicate date-symbol pairs.

#### Umbrella Portfolio Accross All Factors

Equal-weight average of the five factor portfolios.

### Model Assessor

- `model_assessor` computes the frictionless total return of every portfolio's ideal weights and the S&P, with a table of CAGR, Vol, IR, Sharpe (2% rf), SP CAGR and SP Sharpe.
- Training period: Jan 2010 – Dec 2015. Final assessment (test): April 2016 – Sept 2026, run once after everything was built.
- `signal_assessor` tests each raw signal's power to predict the next 90-day return with a **Lasso** regression (`LassoCV`, 5-fold `TimeSeriesSplit`) on the training period. Predictions are clipped to ±20%.
- Predicted portfolio return = Σ weight × predicted return; actual = Σ weight × realised 90-day return, compared every quarter.
- Predicted vs actual scatters per signal use a random sample of 50,000 points for plotting, with the line of best fit on all points.

### Create Portfolio Class

`Portfolio` (in `trading.py`) holds cash and share lots. Each lot keeps its buy date so the holding period can be enforced; sells are FIFO. It stores the starting NAV (`start_cash`).

### Create Orderbook/History Class

`OrderBook` keeps three histories:

- `trades` — every fill: date, symbol, side, shares, price, value, TC.
- `unfilled` — wanted vs executed shares with the reason (`hold_90d`, `buy_cash_limit`, `sell_cash_cap`).
- `daily` — NAV, cash, cash in bps, TC, buy/sell value, scale factors and number of blocked sells.

### Create Trading module

`Trader.rebalance` applies the rules each day:

- Wanted trade = target shares − current shares, kept only if > 5bps of NAV **and** > 1 share of value.
- 90-day holding period: only lots held ≥ 90 calendar days can be sold. Blocked sells are logged.
- If cash after trading (net of TC) would be above 20bps of NAV → sells are scaled down.
- If buys + TC would cost more than cash + sell proceeds after TC → buys are scaled down (this is what scales buys when sells are blocked).
- Shares are rounded down after scaling, so the cash limits can never be broken.
- TC of 2bps: a sell returns 0.9998 of its value in cash; a buy costs 1.0002 of its value in cash. Positions always move at full value, so NAV falls by exactly the TC.
- TC is summed every day.
- No trade is made in a security whose price is missing that day.

### Create backtester

`run_backtest` runs one portfolio (the Umbrella for the final run) through the `Trader` and returns a `Backtest` object with:

- History of NAV, cash, TC, etc. (`daily`)
- The orderbook history of buys and sells (`trades`, `unfilled`)
- Current open positions (`holdings`) and lots
- Total return index of the portfolio and the S&P (`total_return`)
- A bad days screen: days where NAV moved more than 10%, with the stocks that drove the move

**Bad day (definition):** a day where the portfolio's NAV moves more than 10%. A move that large in a diversified, near market-cap-weighted portfolio suggests odd stock data may have been involved (e.g. a bad price print), so these days are flagged for review along with the stocks that drove the move.

Note: the one bad day found in the final backtest was the most volatile day of the COVID-19 pandemic sell-off. While it is good to flag, it reflects a real market move rather than bad data, so it is not an issue.

### Create backtest Assesor

`create_backtest_plots_and_summary` saves a PDF with:

- Summary table (portfolio & S&P rows): CAGR, Vol, IR, Sharpe (2% rf), Max Drawdown, Annualized Turnover, Annual TC
- Final value table: starting value ($1,000,000), final value including cash and open positions, number of positions, total return. The S&P is grown from the same starting value.
- Total return of the portfolio vs the S&P
- Cumulative active return vs the S&P
- The worst bad day, with the total number of bad days

---

## Core Assumptions

- If prices are missing a security is not tradeable and position value if help is previous day's
- TC = 2bps
- Cash interest is 0. I am hoping to hold 20bps at most. 2% of 20bps is <1/2 bps per year so for now its fine.
- Returns use day-over-day `adj_close` (total return including dividends). Share counts are also based on `adj_close`.
- Trades execute at the day's close with no slippage beyond TC.
- The 90-day holding period is in calendar days.
- Weights are lagged 3 days when built and the backtest adds 1 more day, so trades use signals 4 trading days old.
- The backtest starts all in cash, so nothing can be sold for the first 90 days.

---

## Tests and Checks

All checks are asserts that run inside the program, so a run stops with a message on the first failure.

### Universe membership (`apply_membership`)

- The input data has `date` and `symbol` columns.
- No duplicate `(symbol, quarter_start)` keys in the membership table.
- The membership merge is many-to-one (`validate='many_to_one'`), so it cannot duplicate rows.
- No look-ahead: every member's `selection_date` is on or before the signal date.
- No duplicate date-symbol pairs after filtering.

### Ideal weights (`create_model_ideal_positions`)

- Each portfolio's weights (bv, sm_lg, inv_agg, prof, bta and Umbrella) sum to 1 on every date.
- No duplicate date-symbol pairs.

### Trading backtest — after every trading day (`Trader._check_day`)

- No trade in a security whose price is missing that day.
- Every holding has a price.
- Portfolio weights + cash weight = 1.
- Change in cash = sells − buys − TC.
- Each position's change in value = its traded value (buys add their full value, sells remove it).
- NAV after trading = NAV before trading − TC.

---

## Not Built / Future Work

- Ridge regression refit once a year (a Lasso per signal on the training period was used instead).
- The sector-capped, sector-rotation and combined variations — only the sector-neutral version was built.
- Intrasector and sector position tiering.
- Using the leftover cash after rounding shares down.
