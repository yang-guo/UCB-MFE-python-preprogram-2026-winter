# Homework 2 — Stock analysis pipeline

Prabhutosh Kumar · Data Analysis · October 29, 2026

## Setup and run order

Use Python 3.12. Run these commands from this submission directory:

```sh
python3 -m venv .venv
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m ipykernel install --user --name hw2 --display-name "HW2"
```

Download the course `data.db` from bCourses into `Lectures/Lecture 4/data/`
(relative to the repository root). Alternatively put your working copy directly
in this directory's `data/data.db`. Never commit either database.

1. Open `01_data_cleaning.ipynb`, select the **HW2** kernel, and run all cells.
   It copies the lecture database locally if necessary, diagnoses `ohlc_hw`,
   and recreates only `ohlc_hw_cleaned`. It also saves local audit CSVs under
   `data/audit/` and displays accounting and before/after statistics.
2. Run `02_stock_report_template.ipynb` with that kernel. Its tagged parameters
   default to AAPL in 2024; change them if the database history differs.
3. Execute the complete universe:

   ```sh
   python 03_run_all_stocks.py --kernel-name hw2
   ```

   Optional `--db-path` and `--output-root` arguments accept other local paths.
   Paths default to this submission directory regardless of the shell directory.
4. Run the deterministic injected checks with `python checks.py`.

The cleaning notebook expects the supplied reference table `ohlc`, accepting
`date` as an explicitly renamed reference date column. The homework table uses
`ts`. Source tables are never overwritten.

## Design and cleaning decisions

Reusable, documented functions live in `analysis.py`, imported by both notebooks;
`checks.py` contains small independent injected examples. No network market data
or credentials are required.

Ticker HTML tags, surrounding whitespace and case are normalized before grouping
and deduplication, with original labels retained. Exact normalized duplicates
are removed. All members of ambiguous conflicting keys are quarantined. Rows
with all four raw prices missing are removed before filling. Missing raw prices
use only the preceding **validated** row in the same sorted ticker. Repaired
rows must satisfy today's OHLC bounds. Leading gaps and indefensible repairs
are quarantined, not extrapolated. Missing volume is zero with an explicit
unknown-volume flag; it does not establish that no trading occurred.

Only matched, unambiguous reference dates supply adjustment factors. Missing or
inconsistent adjusted close is reconstructed as cleaned close times that date's
trusted factor. Factors are never carried across a possible corporate action.
Unresolved adjustment repairs are quarantined. Reference prices do not replace
the homework raw history. Final validation checks positive finite prices, raw
bounds, finite nonnegative volume and unique keys. Dates are stored as ISO dates.

Before/after issue reports permit multiple issues per row. Invalid keys, prices,
bounds and ambiguous duplicates are high severity; missing observations/cells
and statistical flags are medium; zero volume is low. Missing-date checks use
each ticker's supplied `ohlc` calendar across its supplied history, so holidays
are not counted as missing weekdays. Missing sessions are reported, never filled.

Extreme-return screening uses simple raw-close returns, both tails, and sample
standard deviation (`ddof=1`). Undefined returns are excluded and zero/undefined
standard deviations produce no flags. Without external evidence, an extreme
return is **uncertain**, retained with `extreme_return_review`, and never claimed
to be a split or news event. Impossible raw prices are quarantined rather than
interpolated speculatively. The notebook shows every quarantined/removed row and
reconciles their counts with the input.

## Reading reports and batch results

Reports use adjusted close and daily log returns. Twenty-return volatility needs
21 prices; the 50-day average needs 50 prices. Warm-up estimates remain missing.
Sharpe uses a 2% annual risk-free rate and the assignment's approximation;
undefined results display N/A. Price colors compare volatility to its own
retrospective mean, rather than a forecast. Drawdown can recover to zero.
Whole-period volatility in the summary differs from mean rolling volatility
used in the portfolio scatter. Numeric CSV values are fractions.

The common window is the six calendar months before the month containing the
latest cleaned observation, even if that observation is month-end. Reports show
requested and actual dates: shorter histories are not extrapolated and are not
strictly comparable. Missing/quarantined sessions can make an observed return
span multiple sessions and affect daily-risk estimates.

The runner catches each ticker failure, continues, logs tracebacks and saves
`failed_stocks.txt`. It collects a named JSON output only from notebooks that
successfully execute in the current run; stale and failed notebooks are ignored.
The CSV includes actual coverage, observation count and report paths. Nonfinite
Sharpe values are excluded from top/bottom rankings, which may overlap. Undefined
scatter colors appear gray; unavailable rolling volatility cannot be plotted.
An all-failed run writes a header-only CSV and explicitly reports no comparison.

## Submission artifacts

Full batches and databases are ignored locally. After running against the course
database, commit only 2–3 successful example report notebooks plus
`portfolio_summary.csv` (and optionally the scatter PNG/run metadata), using
explicit `git add -f` paths. Do not add the entire reports folder.

**Verified cleaning results (October 28, 2026):** the supplied homework table
contains 397,711 rows. Cleaning retained 368,246 rows across 30 tickers, removed
29,460 exact normalized duplicates, and quarantined 5 unresolved rows. There
were no absent reference-calendar dates before cleaning. Post-cleaning validation
finds 5 missing dates (the quarantined observations), 5,057 extreme-return review
flags and 74 zero-volume observations. Retained prices, bounds, volumes and keys
pass validation. The executed cleaning notebook includes the full accounting.

**Pending submission outputs:** the full Papermill batch and required 2–3 sample
report notebooks/portfolio CSV have not been generated. Permission to start the
batch Jupyter kernels was declined during this session. Run step 3 above before
submitting; no synthetic output is presented as course data.
