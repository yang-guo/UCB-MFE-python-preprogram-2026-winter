# CLAUDE.md: HW4 rules for Ananya Das

This file records the binding rules for my HW4 submission. They come from
`Lectures/Lecture 8/hw4.md` (the assignment) and `Lectures/modeling/README.md`
(the data guide), and they override convenience. If a rule here seems to
conflict with those sources, stop and ask. Do not guess.

## Hard stop: the final test period

**Never run, load, inspect, plot, or summarize anything from the final test
period (Apr 1, 2016 through Sep 30, 2026) until I explicitly say so in the
conversation.**

- "Anything" includes stock prices, the `^SP500TR` benchmark, fundamentals,
  membership, model predictions, backtests, and quick sanity checks or
  `.describe()` calls. Peeking counts as running.
- While development is in progress, every data load must use a cutoff of
  `end="2016-03-31"` or earlier. Prices through the buffer are allowed only so
  that targets for late-2015 signals can mature.
- Code that can reach the test period must sit behind an explicit guard that
  defaults to off (for example, `RUN_FINAL_TEST = False` plus an assertion on
  the maximum loaded date). Never flip the guard unless I ask.
- Before the test run, freeze and write down the features, model,
  hyperparameters, validation result, strategy, sizing, costs, and retraining
  rule. After the test run, do not tune anything based on the test results. If
  something must change, report both the original and the changed results and
  say why.
- An earlier approval does not carry forward. Approving one test run does not
  approve another.

## Paths and data

- Submission folder: `Homeworks/HW4/Ananya_Das/`. Put all my code, notebooks,
  and write-ups here.
- Databases live in `Lectures/modeling/data/`:
  - `equities_class.duckdb` (30 stocks, 2008–2015) is for getting code working
    quickly.
  - `equities_top2000.duckdb` (3,849 stocks, 2000 to Sep 30, 2026) is for the
    final experiment. `^SP500TR` is in its `benchmark` table.
- Never commit `.duckdb` files, `.env`, or large generated outputs. Write
  outputs to an ignored folder or keep them small.
- Environment: run `uv sync --locked` from the repo root and use the `.venv`
  kernel.
- I may reuse lecture data-loading and preprocessing code from
  `Lectures/modeling/course_data.py`, but I must **cite anything adapted**
  (file and function). `Lectures/modeling/backtest.py` is a reference only. The
  backtester must be **my own implementation**.
- AI-assisted work must be disclosed and explained in the write-up.

## Date splits

| Period | Dates | Use |
|---|---|---|
| Lookback / warm-up | Jan 3, 2000 to Dec 31, 2009 | Feature lookbacks only (trailing windows, filings) |
| Development | Jan 1, 2010 to Dec 31, 2015 | Build features, train, validate, choose the strategy |
| Buffer | Jan 1 to Mar 31, 2016 | Only lets targets from development signals finish. No signals or trades. |
| **Final test** | **Apr 1, 2016 to Sep 30, 2026** | Evaluate the frozen strategy, **only when I say so** |

- Validation within development must be chronological (walk-forward or
  expanding window), never shuffled. Keep every stock on a given date in the
  same fold. Purge any training row whose target ends on or after the first
  validation signal (see `date_folds` in `course_data.py`).
- The holding period is **90 calendar days** in every target, fold, and trade.

## Look-ahead rules

**Signal timing**
- Features and scores are computed after the close on day `t`. The earliest
  entry is the **next market session's close** (`t+1`).
- Target = adjusted close at the first quoted close on or after
  `entry_date + 90 calendar days`, divided by the adjusted close on
  `entry_date`, minus 1. The target must line up exactly with the backtester's
  entry and exit.
- Compute price features per ticker on the supplied market calendar, **with no
  forward-filling of prices**. Gaps stay missing.

**Fundamentals**
- A filing is usable only from **filing_date + 1 calendar day**, never from
  fiscal period end.
- Exclude records with a missing or nonpositive filing lag
  (`filing_date <= fiscal_date`).
- A late filing for an older fiscal period must not replace a newer known
  period.
- Blank financial features more than 550 days past fiscal end. Never backfill
  from future reports.
- Remember that vendor values are latest restated values, not original
  vintages. Treat this as a stated limitation.

**Universe and survivorship**
- Use `universe_membership` by the **quarter of the next-session entry**, and
  require `selection_date <= signal date`. A signal on the last session of a
  quarter for an entry in the next quarter uses the new quarter's list.
- Never filter or use as features: `is_active_snapshot`,
  `delisted_at_snapshot`, first or last selected quarter, selected-quarter
  count, future price coverage, or whether a label is complete. **Do not remove
  stocks because they later disappear.**
- Training rows keep their own historical membership, not the fit date's list.
- Same-date peer or cross-sectional features use only that date's eligible
  members.
- A stock leaving the universe means no new buys. **Existing positions are kept**
  and exit normally at maturity.
- Sector and other universe fields are current snapshot labels. Note this as a
  limitation if I use them.

**Fitting**
- At every fit, use only targets whose holding period has already ended:
  `label_end < fit/signal date`. This still applies after the buffer and on
  every retrain during the test period.
- Fit all learned preprocessing (clipping, imputation, scaling, encoding,
  feature selection) on training rows only. Keep it inside an sklearn
  `Pipeline`.
- Retraining during the test period is optional. If I retrain, the schedule
  and rule must be chosen during development and frozen.
- Censored or missing targets stay missing. Never fill them in.

## Backtester rules

| Rule | Requirement |
|---|---|
| Starting capital | $1,000,000. Long only. No borrowing, leverage, or negative cash. |
| Entries | Any trading day, at the close. Decisions use only information through the prior close; the fill price is unknown when deciding. |
| Holding period | Each purchase is held 90 calendar days, then sold at the close on maturity, or at the next market session if the market is closed |
| Cash | Purchases consume available cash; sales return cash. Never spend the same cash twice. Order budgets must be covered by cash at the decision time. |
| Test end | Value open positions at the Sep 30, 2026 close. Never use later prices. |

- Track purchases, sales, cash, open positions (per lot), and **daily
  portfolio value**, with a fill log.
- Use adjusted closes consistently. These are total-return units, so **do not
  add dividends again**.
- State assumptions explicitly in the write-up: transaction costs, missing
  prices, and cash interest. The class baseline is 5 bps per side with fees
  included in budgets, 0% cash interest, and stale marks for at most 5 sessions
  before stopping to investigate.
- **Never invent fills.** If there is no quote on the entry day, reject the
  order and log it. If there is no quote at maturity, exit at the next observed
  quote and flag it. Never treat a final price row or an inactive flag as a
  cash payout or a zero recovery. Report any data gaps that make a result
  unreliable.
- Before trusting a full run, verify the accounting on small cases that can be
  checked by hand (one buy and sell, fees, cash reuse, maturity on a holiday or
  weekend, a missing quote). Keep these checks in the submission as evidence.
- The strategy's choices (number of stocks, sizing, concentration caps, trading
  frequency) are mine, but they must be fixed during development.

## Benchmark comparison

- Benchmark: $1,000,000 in `^SP500TR` at the **first test close** (Apr 1,
  2016), held through Sep 30, 2026. The strategy's first investment is on the
  same close.
- Report for **both** portfolios: final value (cash plus open positions), CAGR
  (actual days / 365.25), maximum drawdown, and Sharpe ratio from daily returns
  with a 2% annual risk-free rate converted to a daily rate (annualized with
  252).
- Plot both cumulative-return curves on one chart. Say whether the strategy
  beat the benchmark and by how much, and explain the risks and limitations.

## Submission checklist (due Nov 12, 2026; PR to `kevinramlal` and `raghuramkowdeed`)

- [ ] Code (a notebook and/or `.py` files) with run instructions
- [ ] A brief explanation of the features, model, validation, and strategy
- [ ] The performance table and cumulative-return chart
- [ ] Evidence of accounting and timing checks
- [ ] Findings, including failed ideas and next steps
- [ ] Citations for reused code and a disclosure of AI assistance
- [ ] Residual data limitations (from the data guide's "what the source cannot
      establish" section)

## Agreed design (decided 2026-11-16; changes need my approval)

- **Layout:** `config.py`, `data.py`, `features.py`, `model.py`,
  `backtester.py` (my own), `tests/test_backtester.py`, `hw4.ipynb`,
  `README.md`, and `.gitignore` (ignores `outputs/`).
- **Features (8):** 1-month return, 12-month-minus-1-month momentum, 63-day
  volatility, drawdown from the 252-day high, operating ROA, accruals/assets,
  debt/assets, and log capitalization proxy. Each is converted to a
  cross-sectional percentile rank among that date's eligible stocks.
- **Screen:** raw close >= $5 on the signal date.
- **Target:** 90-day return minus that date's cross-sectional median.
- **Model:** Ridge regression. Only the penalty is tuned.
- **Validation:** expanding-window walk-forward with validation years 2012,
  2013, 2014, and 2015. Fit before each year's first signal with
  `label_end < signal`. Use mean daily rank IC and the decile spread, then one
  development backtest for 2012–2015. Decide the small set of variants in
  advance.
- **Refit:** every January, plus once before Apr 2016. Expanding window,
  matured targets only.
- **Trading:** monthly cohorts. Signal at the month-end close, buy at the next
  close, top 20 names, cohort budget = min(cash, NAV/3), equal weights. Caps
  are 5% of NAV per name and 25% per sector.
- **Costs:** 10 bps per side, with the fee included in the budget. Report
  sensitivity at 5 and 20 bps. Cash earns 0%.
- **Missing quotes:** reject an entry if there is no quote. If there is no
  quote at maturity, exit at the next quote and flag it. Marks can be stale
  for at most 5 sessions.
- **Delistings:** if a held stock has no quote for more than 5 sessions, freeze
  the lot at its last mark, never sell it while frozen, and flag it as
  unresolved. **Unfreeze rule (approved 2026-11-16):** if real quotes resume,
  unfreeze the lot and treat it normally. It is marked at the new quote and
  sold at maturity, or at the next real quote if maturity has passed, flagged
  as delayed. Report a sensitivity that marks frozen lots at zero.

## Coding style

The code is for a teaching repo, so keep it clear, commented, and
educational. Prefer small, testable functions. Add assertions that enforce the
timing rules above (for example, max loaded date, `label_end < fit date`, and
`available_date <= date`).
