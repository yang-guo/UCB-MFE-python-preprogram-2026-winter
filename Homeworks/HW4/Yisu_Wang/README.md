# HW4 — Stock signals and portfolio backtest

## Files and run order

1. `01_data_and_model.ipynb` prepares the feature and target panel from `Lectures/modeling/data/equities_top2000.duckdb`.
2. `02_model.ipynb` fits the rolling LightGBM models and writes out-of-sample scores.
3. `03_backtest.ipynb` trades those scores, checks cash and positions, and compares daily NAV with `^SP500TR`.

Use the repository's Python environment with `duckdb`, `lightgbm`, `numpy`, `pandas`, `matplotlib`, and `ipykernel` installed. Run the notebooks in order from the repository or its HW4 directory. The original database belongs in `Lectures/modeling/data/`; the prepared database is `Lectures/modeling/data/hw4_yisu_60_sessions.duckdb`. The yearly models, logs, and scores are written under `Homeworks/HW4/Yisu_Wang/outputs/rolling_lgbm_ic_v5_top4_60_sessions/`. These local databases and generated outputs are not part of the submission.

## Data and model

An end-of-day signal can enter only at the **next market session's close**. Candidates must be in the known top-2,000 membership for that entry quarter. Quarterly financial statements are attached only after their filing date; the sector descriptor is from a snapshot and may be inaccurate historically. The feature panel contains price/volume rolling statistics, dimensionless financial ratios and filing ages, and sector indicators. Missing inputs stay missing for LightGBM. No future entry/exit price, target, or label-end date is a model feature.

For this experiment, each lot has a planned exit **60 market sessions after entry**, and the target is its adjusted-close return over that period. This differs from the assignment's 90-calendar-day specification. A missing entry quote prevents a fill; an exit label uses the next observed quote within five sessions, otherwise the label is missing. Unlabeled signals may still receive model scores; labels used for fitting must have finished before the fit date.

Each annual rolling fit uses five earlier signal years for training and the next year for validation. The search samples 32 of 192 parameter combinations; the four models with the best mean daily cross-sectional validation IC are retained. The objective is negative daily Pearson IC with a positive diagonal Hessian approximation. Within each signal-date cross-section, each model's scores are clipped at the 0.5th/99.5th percentiles and standardized, then the four scores are averaged. The separate yearly model artifacts, selected settings, and score files support the backtest. The 2007–2026 run is a rolling out-of-sample **research period**, rather than the assignment's separate 2016–2026 final-test split.

## Portfolio rules

Start with $1,000,000 at the first scored signal in 2007 and stop on September 30, 2026. Four cohorts begin 15 market sessions apart. At each opportunity a cohort selects up to 50 highest-scored stocks using the **previous close's** scores, and invests at most one quarter of previous-close NAV, capped by available cash. Purchases are equal-dollar within the cohort; there is no borrowing or cash interest. Each lot is scheduled for sale after 60 market sessions. A cohort normally re-enters on the next session; if a complete entry attempt lacks scores, cash, or executable quotes, it skips that round rather than synchronizing with other cohorts' next-day retries. As a result the initial 15-session spacing is not a promise of an entry on every later 15th session.

Buys and sells use observed adjusted closes and cost **5 basis points of traded notional on each side**. Missing entry quotes cancel those individual orders. A matured position without an observed exit quote stays open until a valid quote returns; its unavailable proceeds are never credited to cash. Daily NAV is cash plus open lots marked at their latest observed adjusted close. Marks older than five sessions are explicitly reported as unresolved and make the result provisional. Open lots remain in end-date NAV; there is no forced liquidation or invented corporate-action cash flow. The benchmark is $1,000,000 invested at the first test close in `^SP500TR` over the same dates.

## Results and limitations

The notebook displays final value, CAGR, maximum drawdown, daily-return Sharpe (2% annual risk-free rate), cumulative-return curves, trade and missing-quote diagnostics, and checks for cash, NAV, unique lots, and execution timing. A no-signal gap previously made several cohorts retry together; the submitted notebook skips a missed round to keep their schedules separate. It must be rerun after that change before its results can be quoted.

<!-- BACKTEST_RESULTS_START -->
- Dates: **2007-01-03–2026-09-30** (4,967 market sessions); starting capital: **$1,000,000**.
- Ending NAV (cash plus open positions): **$12,141,621** versus **$7,857,618** in `^SP500TR`; difference **$4,284,003**.
- CAGR: **13.48%** versus **11.01%** (difference **+2.47 percentage points**).
- Maximum drawdown: **-67.77%** versus **-55.25%**; Sharpe: **0.508** versus **0.530**.
- Accounting and timing: **9/9 checks passed**. Entry sessions shared by multiple cohorts: **0**; skipped rounds (no score / no cash): **8 / 0**.
- Ending cash: **$9,894,871** (81.5% of NAV); open lots: **52**. Unresolved lots older than five sessions: **52**, marked at **$2,246,750** (18.5% of NAV). Their actual recovery value is unknown; these are not executed sale proceeds.

The strategy finished **above** the benchmark on marked NAV. This comparison is provisional because old marks may not be realizable, and the observed drawdowns and cash allocation also matter when judging the result.
<!-- BACKTEST_RESULTS_END -->

An earlier backtest stopped at the first long quote gap and could not report the complete period. The current lot accounting retains the position and reports the data gap instead. An old quote may still be a poor estimate of its realizable value after a halt, delisting, merger, or acquisition. The data do not establish the amount or date of a cash settlement. Further work would use dated corporate-action records and separately test the assignment's prescribed evaluation split and holding convention.

The data-joining and lot-accounting patterns were adapted from `Lectures/modeling/` course notebooks and `backtest.py`. ChatGPT assisted with code design, review, and documentation; the user selected the horizon, features, model search, and trading rules and ran the local experiment.
