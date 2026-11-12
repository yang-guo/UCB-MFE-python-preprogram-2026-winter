# HW4 — Sector-Aware ML Strategy

**Vanditha Toleti** · UCB MFE Python Preprogram, Homework 4

A machine-learning model predicts each stock's 90-calendar-day return. A sector-balanced, long-only portfolio buys the best-predicted names every month, holds each purchase for 90 days, and is tested with a daily backtester written for this homework against `^SP500TR` from April 1, 2016 to September 30, 2026.

## Final-test result

| Apr 1, 2016 – Sep 30, 2026 | Strategy | `^SP500TR` |
|---|---|---|
| Final value (cash + open positions) | **$2,087,737** | **$4,402,747** |
| CAGR | 7.26% | 15.17% |
| Maximum drawdown | −42.70% | −33.79% |
| Daily Sharpe (2% risk-free rate) | 0.33 | 0.77 |

**The strategy did not beat the benchmark.** It finished $2,315,010 (52.6%) below the benchmark's final value, with a deeper drawdown and a lower Sharpe ratio. The base case includes 10 bps transaction costs per trade; with zero costs the CAGR would be 8.05%.

![Cumulative returns](cumulative_returns.png)

## Files

| File | What it contains |
|---|---|
| `HW4_Strategy2_Sector_Aware_ML.ipynb` | Notebook 1. Data exploration and data-quality audit, 90-day targets, chronological validation of three models (2013–2015), the fixed portfolio rules, and development conclusions. Uses `equities_class.duckdb` only. |
| `HW4_Strategy2_Final_Backtest.ipynb` | Notebook 2. Full-universe walk-forward model fits, monthly stock picks, the daily backtester and its accounting tests, the benchmark comparison, charts and final findings. Uses `equities_top2000.duckdb`. |
| `cumulative_returns.png` | Strategy vs `^SP500TR` growth of $1 over the final test. |
| `final_performance_comparison.csv` | The final metrics table. |
| `requirements.txt` | Package versions used for the submitted run. |

## How to run

1. **Environment.** From the repository root, either use the course environment (`uv sync`) or create one with `python -m venv .venv` and `pip install -r Homeworks/HW4/Vanditha_Toleti/requirements.txt`. Python 3.12 or newer.
2. **Data.** Download `equities_class.duckdb` and `equities_top2000.duckdb` from bCourses into `Lectures/modeling/data/`. The notebooks also need the course helper `Lectures/modeling/course_data.py`, which is already in the repository.
3. **Notebook 1.** Open `HW4_Strategy2_Sector_Aware_ML.ipynb` in VS Code, select the `.venv` kernel and use **Run All** (a few minutes). It writes the frozen strategy configuration to `Lectures/modeling/outputs/hw4_development/strategy_config.json`.
4. **Notebook 2.** Open `HW4_Strategy2_Final_Backtest.ipynb` and use **Run All**. The first run builds a month-end panel for all 3,849 stocks and caches it as `Lectures/modeling/outputs/hw4_final/full_monthly_panel.parquet` (the whole notebook took about 22 minutes on a 2-core machine, mostly the eleven Random Forest fits). Later runs reuse the cache; set `REBUILD_FULL_MONTHLY_PANEL = True` to rebuild it. Notebook 2 stops with an error if notebook 1 has not been run.
5. **Outputs.** The daily NAV, trade ledger, open lots, predictions and model-fit audit are saved under `Lectures/modeling/outputs/hw4_final/`, which is Git-ignored. Databases and the cache are not committed.

## Approach

### Features
The twelve course features from `course_data.py`, all computed only from information available at the signal-date close:
- **Price:** 21-day and 63-day momentum, 12-1 month momentum, 63-day realized volatility and 252-day drawdown, computed on the market calendar without filling missing prices.
- **Fundamentals:** net margin, operating ROA, debt/assets, cash/assets, free cash flow/assets, accruals/assets and filing age. A filing becomes usable only from the calendar day after its filing date, and data older than 550 days is treated as missing.

Sector is **not** a model feature, because the database only has a current snapshot of each company's sector. It is used only to group the portfolio picks.

### Target
The return from the next session's close (the earliest possible entry) to the first observed close on or after 90 calendar days later, using adjusted prices. Exits may be delayed by at most five market sessions; otherwise the target is missing. A notebook check confirms that every label obeys this rule.

### Model and validation
- Candidates: Ridge regression, Random Forest and Histogram Gradient Boosting, all inside the course's preprocessing pipeline (feature clipping, median imputation and scaling fitted on training rows only).
- Validation: expanding-window folds for 2013, 2014 and 2015. Each fold is fitted once, before the year's first signal, on labels whose 90-day window **ended before** that signal. Asserts confirm this for every fold.
- Training targets are winsorized at the 1st and 99th percentiles of each training set, because a few stocks (APDN, BESS) had 90-day moves of several hundred percent. Validation is scored on raw targets.
- Selection metric: mean daily Spearman rank IC (rank correlation between predicted and realized returns on the same date). **Random Forest** was chosen: it had the highest mean rank IC (about 0.06) and was the only model with a positive rank IC in every validation year.

### Trading strategy
- Monthly signal at the last market session of each month; buy at the next session's close.
- **Growth sleeve (50%):** in each sector, the stock with the highest positive predicted return.
- **Defensive sleeve (50%):** in each sector, among the lowest-volatility 40% of eligible stocks, the one with the highest positive predicted return, never the same stock as the growth pick.
- Equal weight within each sleeve, capped at 10% of portfolio value per stock and 20% per sector. Capital not allocated stays in cash.
- Candidates for each entry come from that entry quarter's `universe_membership`, including stocks that are inactive today. Held positions are not sold early when a stock leaves the universe.
- Final test: the model is first fitted at the March 31, 2016 signal, then refitted every December. Each fit uses only outcomes that ended before the fit date, with the same target winsorization. The final fits train on month-end rows only, to keep the 3,849-stock panel small enough to build and fit.

### Backtester (written for this homework)
- $1,000,000 start, long only, no leverage, fractional shares.
- Each lot is sold at the close of the first market session on or after its 90-day maturity that has a real quote, within five sessions. No fill is ever invented. A lot that cannot be sold in that window stays open at its last observed price and is reported.
- Order size is the sleeve weight × portfolio value at the signal close, reduced by the name and sector caps (measured at signal-date prices) and by the cash actually available. Sales are processed before purchases each day, so the same cash is never spent twice. Orders that cannot be filled are logged with a reason.
- **Assumptions:** 10 bps transaction cost on every buy and sell (0 and 20 bps shown as sensitivity); cash earns 0%; orders below $100 are skipped; missing prices are never filled forward for trading, only for daily valuation; adjusted closes are used throughout and dividends are not added again.
- Benchmark: $1,000,000 in `^SP500TR` at the April 1, 2016 close, held to September 30, 2026. Sharpe uses daily returns and a 2% annual risk-free rate.

## Evidence that accounting and timing were checked

Notebook 2, sections 15 (hand tests), 19 (full-run reconciliation) and 20 (open and unsold lots):
- **Hand-checkable tests on small examples:** a one-share purchase and sale with fees; two orders asking for more cash than exists (the second is cut to the remaining cash); a 10% name cap cutting an oversized order; a missing quote on the maturity day moving the sale to the next session; and an order with no cash left being logged as skipped.
- **Checks on the full run:** ending cash equals starting capital plus every ledger cash flow; cash is never negative; every buy and sell uses a real quote; sold quantities match their purchase lots; daily portfolio value equals cash plus holdings; every buy fills on the session after its signal and inside the test window; the first buy is on April 1, 2016; every sale is on or after its maturity session and within five sessions; 25 randomly sampled fills match the database's adjusted closes; and no price after September 30, 2026 is read.
- **Point-in-time checks:** every model fit's latest training label ends before the fit date; financial data is used only after its availability date; quarterly membership is known by the signal close.

On the final run: 940 buys and 918 sells were filled, 1,832 planned orders were skipped for lack of cash (logged with a reason), total fees were about $99,000, and 22 lots were open at the end. Three of those lots had matured but could never be sold because their prices stopped (LPI, HOUS, RCB). They are carried at their last observed price, $362,559 or 17% of the final value. Valuing them at zero gives $1,725,178 (CAGR 5.33%); at cost, $1,916,338 (CAGR 6.39%).

## Findings

- **Why it lagged.** Every sector receives equal capital, so Technology and the largest companies, which drove the index over 2016–2026, are heavily underweighted, while small sectors are overweighted. The growth sleeve's top-predicted names tend to be volatile stocks, so the portfolio had 24.6% annualized volatility against 17.8% for the index. The strategy beat the index in 2016, 2025 and 2026 (to September), and lagged in the other eight years, with larger losses in 2018 and 2022.
- **The signal was weak from the start.** Validation rank IC was about 0.06. In development the selected names earned less than an equal-weighted portfolio of all eligible stocks (2.3% vs 2.6% per 90 days), and the final test is consistent with that.
- **How capital actually rotated.** A monthly cohort can target close to 100% of the portfolio and lots are held 90 days, so the portfolio mostly held one cohort of about 22–24 stocks at a time and rotated about every three months. Most intermediate monthly orders were skipped for lack of cash.
- **Both sleeves made money** on closed lots (defensive about $603,000, growth about $377,000), so the shortfall comes from portfolio construction and risk, not from losses on most picks.
- **Unsuccessful ideas.** The two-sleeve, sector-balanced design was meant to reduce risk but produced higher volatility and a deeper drawdown than the index. Requiring a positive predicted return rarely moved money to cash, because predictions cluster around +3–4%.
- **Changes we would try next.**
  - Weight sectors by market capitalization, or rank across the whole universe.
  - Size each monthly cohort at about one-third of the portfolio, so the monthly signals are actually used.
  - Add a point-in-time liquidity screen to avoid stocks whose prices stop or jump.
  - Rank stocks against each other instead of requiring a positive prediction.
  - Predict returns relative to the market or sector.

## Limitations
- Sector labels are a current snapshot, not historical classifications.
- Fundamentals are the vendor's latest values, not the numbers originally filed, so restatements can leak in.
- The data is not a certified survivorship-free feed, and there are no delisting proceeds. A stock that stops trading while held is carried at its last price.
- Development validation used a 30-stock sample, so model selection rests on a small and possibly unrepresentative universe.
- No liquidity screen was applied; trades are assumed to fill at the close at the stated cost.
