# HW2 — Scalable Stock Analysis Pipeline (Dhruv Chopra)

Cleans the dirty `ohlc_hw` table, builds a parameterized single-stock risk report, and runs that report
for every stock with papermill to produce a portfolio summary.

| File | Purpose |
|---|---|
| `01_data_cleaning.ipynb` | Diagnose, repair, validate; writes `ohlc_hw_cleaned` (plus audit tables) to the database |
| `02_stock_report_template.ipynb` | Parameterized risk report for one ticker and date window (`parameters` cell tagged) |
| `03_run_all_stocks.py` | Runs the template for all tickers, collects results, writes the summary CSV and plot |
| `reports/2026-10-29/` | Sample reports (AAPL, IBM, MRK), `portfolio_summary.csv`, `risk_return.png`, `failed_stocks.txt` |

## How to run

1. Download `data.db` from bCourses to `Homeworks/HW2/data/data.db` (it is git-ignored; never commit it).
2. Create an environment and install dependencies (Python 3.12 or 3.13):
   ```bash
   uv venv --python 3.13 .venv && source .venv/bin/activate
   uv pip install -r requirements.txt
   ```
3. From this folder, run in order:
   ```bash
   jupyter nbconvert --to notebook --execute --inplace 01_data_cleaning.ipynb   # creates ohlc_hw_cleaned
   python 03_run_all_stocks.py                                                 # all 30 reports + summary
   ```
   The template can also be run alone, e.g.
   `papermill 02_stock_report_template.ipynb out.ipynb -p ticker MSFT -p start_date 2024-01-01 -p end_date 2024-06-30`.
   `03_run_all_stocks.py --tickers AAPL MSFT` runs a subset; `--help` lists all options.

The raw `ohlc_hw` and reference `ohlc` tables are only read, never modified. Part 1 adds
`ohlc_hw_cleaned`, `ohlc_hw_quarantine` and `ohlc_hw_quality_report` (diagnosis before repairs), plus lineage
tables so a run can be explained without re-executing it: `ohlc_hw_final_report` (after repairs),
`ohlc_hw_stage_log`, `ohlc_hw_approved_exceptions`, `ohlc_hw_ticker_audit` and `ohlc_hw_cleaning_runs`
(run id and thresholds; every cleaned row carries its `cleaning_run_id`).

**Intended use: retrospective analysis.** 3σ thresholds use each ticker's full history, the factor screen
uses a centred window and interpolation uses the next day's price. That suits after-the-fact risk reports,
but it would be look-ahead in a backtest, which would need trailing-only statistics, no interpolation from
future prices and point-in-time vendor factors.

## Key design decisions

* **Validate, normalize, then deduplicate.** Required columns and numeric types are checked first. Ticker
  labels contain HTML (150 labels → 30 tickers); an empty ticker or a date that is not a real `YYYY-MM-DD`
  date would be quarantined as an invalid key, and dates are rewritten in one canonical form. Normalizing
  first exposes 29,460 exact duplicates, which are removed. Conflicting `(ticker, ts)` rows would be
  quarantined rather than "keep first"; there are none.
* **Diagnose before repairing.** Every detector returns rows in one Data Quality Report
  (`ticker | issue_type | issue_date | description | severity`), saved before any repair and re-run after.
  Severity: *high* = definitely wrong (impossible bar, nonpositive or contradicted price); *medium* = needs
  investigation (3σ return, missing `adj_close`, missing date, factor jump); *low* = informational (zero
  volume, weekend date, exact duplicate).
* **`ohlc` is a reference, not a replacement, and it is validated before use.** It supplies each ticker's
  expected calendar and a same-date adjustment factor `adj_close / close`, but only where its keys are unique
  and both prices are positive and finite. A row without a usable factor is reported as
  `reference_unavailable` rather than silently passing the comparison. Missing `adj_close` (18,374 cells)
  and inflated values (10,417 cells, 4–24× too high) are rebuilt as `close × factor`, apart from the 14 that
  fall in quarantined rows. The rest of the dirty data is kept as is.
* **Quarantine when a repair is not defensible.** Five tickers start with `open = 0` on their first day;
  with no earlier price to carry forward, those rows are quarantined. After rebuilding, 65 days still show
  an adjustment factor that jumps away from its neighbours because the *reference* is inconsistent there.
  An isolated jump quarantines that day; clustered jumps quarantine the whole span, since inside it the
  right factor cannot be determined (two DIS spans in 1974–75). The quarantine records which rows had their
  own jump (65) and which were excluded only because they sit inside a span (82). A sensitivity table shows
  that at the 5% tolerance the result is the same 147 rows for every cluster size and gap tested.
  Row accounting balances: 397,711 raw = 29,460 duplicates + 152 quarantined + 368,099 retained.
* **Recheck after every repair.** `stage_check` recounts missing and invalid prices, bound violations,
  missing `adj_close`, reference-factor mismatches and factor jumps after each of the eight cleaning stages;
  the pipeline asserts that the final stage is clean. The injected example shows why: forward-filling a
  bad close can push it outside that day's range, which the next stage repairs.
* **No unlimited filling.** Prices are carried forward across at most 5 consecutive sessions; longer gaps
  are quarantined. Invalid prices count as missing *before* the "all prices missing" rule, and confirmed bad
  prints are the only rows ever interpolated (a test caught an earlier version that also interpolated other
  gaps).
* **Impossible OHLC bars → Option 1**: the whole bar is replaced by the most recent fully valid bar of the
  same ticker, at most 5 sessions back, so prices from different days are never mixed; the donor date is
  saved in `bar_donor_ts`. None occur in the real data; injected `high < close` and `low > close` bars,
  non-finite and invalid adjusted prices, a 7-session gap and a day with no reference factor show each repair
  or quarantine decision.
* **Report metrics** use adjusted close and daily log returns, 252-day annualization and a 2% risk-free rate.
  Undefined metrics are `NaN` (shown as N/A). Rolling warm-up gaps are left as gaps, not zeros.
* **Gap-aware returns.** A return that would span a missing session (per the reference calendar) is left
  undefined, so every return used for volatility and Sharpe is a genuine one-day move. The 2024 windows have
  no gaps; a DIS 1974 report shows the guard excluding the return across its quarantined span.
* **Flags match the final data.** `extreme_return_flag_raw` is the 3σ screen on the delivered data;
  `extreme_return_flag` is re-computed on the cleaned data and asserted equal to the final report. The only
  waived validation issue is `adj_close_mismatch` on the four interpolated days, checked by count and by
  verifying the bridged factor.
* **Batch results** come from named scrapbook outputs of *this run's successful* reports only, so stale or
  failed notebooks in the folder are never summarized. A failed ticker is logged and the run continues.

## How outliers were handled, and why

The 3σ screen (simple raw-close returns, sample std, both tails) flags 5,056 days over 60+ years on the
delivered data (5,173 on the cleaned data, where the statistics are re-estimated). A flag is a
review prompt, not a deletion rule, so each large move was checked for evidence:

* **Confirmed errors (4 rows), interpolated:** TRV 1976-10-28, 1978-12-20, 1981-08-10 and AXP 1972-12-14.
  Each is a flat `O = H = L = C` bar with **zero volume** that jumps far from both neighbours and reverses the
  next day. Two independent signs are required: a reversal alone also matches real moves (e.g. JPM
  2009-01-20, March 2020). The adjustment factor for these days is bridged from the neighbouring days,
  and only when both neighbours agree.
* **Kept and flagged:** AAPL 2000-09-29 (−52%) and JPM 2009-01-20 (−21%, reversed next day) show heavy
  volume and real intraday ranges, which points to genuine trading rather than bad prints; the database has
  no news data, so their causes are left unverified. MCD 1968-01-02 (−75%) looks like an unadjusted split,
  but there is no corporate-action data to confirm it, so it is left alone and documented as uncertain.
  All other 3σ days keep `extreme_return_flag = 1`.

## Assumptions

* Each ticker's dates in `ohlc` are its trading calendar (no missing dates were found; the detector is
  demonstrated on an injected gap). No prices are invented for absent dates.
* 384 rows are dated on weekends (24 dates, 1974–75, 16 tickers). They have distinct prices and appear in the
  reference too, so they are kept with `weekend_flag` (362 remain; 22 fall inside the quarantined DIS spans).
  They are decades before any report window, which the notebook checks.
* The batch window is the six calendar months **before the month of the latest cleaned observation**
  (2024-12-06 → 2024-06-01 to 2024-11-30), anchored to the data, not the clock. `reports/<date>/` is the run date.

## How to interpret the reports

Each report shows the price coloured by volatility regime (green at or below the window's mean rolling
volatility, red above, gray during the 20-day warm-up), daily log returns, 20-day annualized volatility
against its mean, and drawdown from the running peak with the deepest point marked. The summary table gives
total and annualized return, whole-window volatility, Sharpe ratio, maximum drawdown and daily statistics.

**Coverage matters when comparing stocks.** Only 8 of the 30 tickers have data through November 2024; the
other 22 end on 2024-10-18, so their "six-month" figures cover about 4.5 months (97 instead of ~126 trading
days). Every report and the CSV (`first_date`, `last_date`, `trading_days`, `coverage_note`) states the
actual coverage, and the scatter plot uses triangles for the shorter histories. Sharpe ratios are
annualized, but they still describe different periods, so treat cross-stock rankings with care.

**Results for 2024-06-01 to 2024-11-30:** 30/30 reports succeeded. Highest Sharpe: IBM 3.22 (+39.9%),
HD 3.07, SHW 2.99. Lowest: MRK −1.93 (−19.9%), BA −1.41, NKE −0.90. NVDA had the highest volatility
(61% average rolling volatility).
