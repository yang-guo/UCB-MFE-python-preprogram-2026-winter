# HW4: Stock-return model and trading strategy

## Question

Can a model that ranks stocks by expected next-90-calendar-day adjusted-price return build a long-only portfolio that performs better than investing the same initial capital in the S&P 500 total-return index? Beating the index is not required to pass; correct timing, portfolio accounting, and a candid explanation of the result are the goal.

## Run the experiment

1. Download `equities_top2000.duckdb` from bCourses into `Lectures/modeling/data/`.
2. From the repository root, install the locked course environment with `uv sync --locked` and activate/select `.venv`.
3. Run `python Homeworks/HW4/Mrityunjay_Balkrishnan/hw4.py`.

The script reads the database without modifying it. It processes securities in batches and writes compact result tables and the cumulative-return chart into this folder's `results/` directory. The database and generated intermediate files are not submission files.

## Research design

- **Features:** 21-, 63-, and 252-to-21-session momentum, 63-session volatility, 252-session drawdown, filing-aware financial ratios, filing age, and sector. Feature construction and the conservative filing-date join use the course's documented helpers in `Lectures/modeling/course_data.py`; that reused code is cited here.
- **Model:** Ridge regression predicts the adjusted-price return from the next market-session close to the first available close at or after 90 calendar days. Numeric clipping, imputation, scaling, and sector encoding are fitted inside the pipeline on training data only.
- **Development:** Monthly signal dates from 2010 through 2015. Select Ridge `alpha` from `[1, 100, 1000]` using chronological, whole-date validation folds for 2012–2015. Each fold trains only on 90-day targets that matured before that validation period. Fit the final development model using outcomes completed before March 31, 2016.
- **Final evaluation:** April 1, 2016–September 30, 2026. Refit quarterly with an expanding training window and the selected alpha. Every fit uses only targets whose exit dates are strictly before the fit date. New-entry scores use the dated quarterly membership of the next-session entry; current membership is not used to remove existing holdings.
- **Portfolio:** Start with $1,000,000. On the first market session of each month, buy up to the 20 highest-scoring eligible names using one third of prior-close NAV, subject to 5% per-name and 25% per-sector caps. Each purchase is a separate lot held for 90 calendar days and sold at the first quoted close on or after maturity, unless a verified corporate action extinguishes the old shares first. Decisions and budgets are fixed from the previous close; the fill price is the current close. No leverage, short sales, or cash interest.
- **Price quality policy:** Rows with zero/missing volume are not treated as executable observations. A change over 50% in `adj_close / close` or a return after more than five missing market sessions is flagged as a price-continuity break. The row is quarantined, labels crossing it are censored, and no new signals are generated for that security after the first observed break. A later quote is never applied to an older held lot across a long gap. No guessed split correction is applied. This rule uses only data observed by each signal date; it does not filter on future survival or label completeness.
- **Accounting assumptions:** 5 basis points per side; fees are included in purchase budgets. Fractional adjusted-price units represent total-return exposure, so dividends are not added again. A missing entry quote skips that purchase. A due lot waits for its next observed quote. If an open holding has no quote for more than five sessions, the run freezes it at the last observed quote for NAV valuation only, ignores later quotes across that gap, and never books a sale. Any affected final value is conditional, not a confirmed liquidation value.
- **Corporate actions:** `corporate_actions.csv` records source links and terms for 24 cash acquisitions/redemptions affecting held lots. These settled 34 lots on the event dates at stated consideration, using each lot's entry-date adjusted/raw price ratio to convert adjusted units to per-share proceeds. A cash merger overrides the original 90-day sale date because the old shares cease to exist; the cash becomes available to the strategy. `reorganization_exits.csv` maps documented share distributions and ticker aliases to successor closes at the lot's 90-day maturity. `supplemental_exit_quotes.csv` contains the one externally sourced daily quote that is absent from the database and needed for a BODY-to-BODI lot. BWNB (a senior note) and SOFIW (a warrant) are excluded because the strategy is for common stocks.
- **Benchmark and metrics:** Invest $1,000,000 in `^SP500TR` at the first test close. Report ending value, CAGR, maximum drawdown, and daily Sharpe with a 2% annual risk-free rate. The chart plots both cumulative-return series through the same final date.

## Timing and accounting checks

The monthly signal panel uses the last supplied market session of each month. The next market session is the entry date, so its fill price is not available to the signal or budget calculation. The 90-day target begins at that next-session close. A quarterly model fit excludes all labels ending on or after its fit date. The backtester retains each purchase as its own dated lot, returns sale proceeds to cash, and does not liquidate a position merely because it leaves the quarterly candidate universe.

Before the full run, a hand-check round trip verifies lot sizing, 90-day maturity, entry and exit fees, cash, and realized PnL with a $1,000 example. During the run, the script checks that cash equals starting capital plus recorded fill cash flows, order budgets do not exceed prior-close cash, ending NAV equals cash plus marked holdings, and final PnL equals realized plus unrealized PnL. It also checks that fees reconcile to the fill ledger. The included `results/fit_audit.csv`, `results/label_censoring.csv`, `results/price_quality_audit.csv`, daily NAV, fills, open lots, and data diagnostics let a reviewer inspect the schedule and books.

## Results and findings

The supplied database contains 3,849 securities and 6,726 market sessions; the model excludes BWNB (a debt security) and SOFIW (a warrant) and uses 3,847 common-stock candidates. Cross-validation selected Ridge `alpha=1000` (mean fold RMSE `0.21099`). At the September 30, 2026 close:

| Metric | Strategy: last-quote marks | Strategy: zero-recovery stress | S&P 500 total-return index |
|---|---:|---:|---:|
| Final value | $1,208,094 | $1,172,825 | $4,402,747 |
| CAGR | 1.82% | 1.53% | 15.17% |
| Maximum drawdown | -67.92% | -67.92% | -33.79% |
| Daily Sharpe (2% risk-free rate) | 0.131 | 0.123 | 0.773 |

In both reported cases, the strategy did not beat the benchmark. Under last-quote marks, final wealth was about **$3.19 million lower**, and CAGR was **13.35 percentage points lower**. The strategy also had a deeper drawdown and a much lower Sharpe ratio. Fees are recorded in `results/fills.csv`.

**The source limitation is reduced but not eliminated.** The supplied data guide says there is no reliable delisting-proceeds table. I researched individual events and added documented cash settlements for 24 tickers, plus successor/ticker mapping exits where a dated price could be established. At test end, 2 lots across 2 tickers still have no verified maturity quote: BODY's May 30, 2024 maturity and DISH's February 29, 2024 maturity. Their $35,268 of last-quote marks is 2.9% of reported strategy NAV; the oldest mark is 689 market sessions stale. These lots remain open and are not represented as confirmed sale proceeds. `results/unresolved_positions.csv` identifies them.

To show the remaining uncertainty without pretending to know recovery values, the code also reports a **zero-recovery stress**: after six consecutive market sessions without a valid quote, the position's mark is set to zero for this sensitivity only. This is not treated as a sale, does not change cash, and does not claim that either stale security became worthless. It reduces final value to $1,172,825, with 1.53% CAGR, -67.92% maximum drawdown, and 0.123 Sharpe. The plot shows last-quote valuation, zero-recovery stress, and benchmark together.

The price audit found 352,403 zero/missing-volume rows, 794 abrupt adjusted-price-basis breaks, and 10,331 quote returns after gaps longer than five sessions across 678 tickers; exact batch-level counts are in `results/price_quality_audit.csv`. The analysis removes untraded rows, stops producing new signals after an observed break, and censors 181 labels that cross one. It retains 5,080 eligible signals without a valid exit label and 147 without a valid entry quote; test scoring does not require a future label. Known cash deals and four successor paths (BHVN, RDFN, SCS, FI) are now settled; the ticker-only BODY-to-BODI path is partially resolved using a sourced April 1 close. The remaining BODY and DISH maturity prices are not in the course database, and I have not guessed them. BID's long source gap is resolved by the Sotheby's cash merger record rather than by applying its later post-gap quotes. The dataset also has incomplete historical security metadata, approximate recorded-date universe reconstruction, vendor restatements, and no reliable delisting-proceeds table. These limitations are described in `Lectures/modeling/README.md`; this experiment does not claim a fully point-in-time or survivorship-free market history.

The strategy underperformed in both reported cases. The remaining stale positions prevent a fully precise realized-return conclusion, but the uncertainty is now quantified at 2.9% of NAV, down from 46.9% before the sourced cash-action work. A final exact number requires authoritative May 30 BODY/BODI and February 29 DISH/SATS closes. Neither the last-quote marks nor the zero-recovery stress is a uniquely correct realized value.

## AI assistance

AI assistance was used to explain the assignment and draft substantial parts of the implementation. The student should inspect and understand the submitted code, reproduce the run, and be able to explain the feature timing, model validation, order budget, fills, and reported results.
