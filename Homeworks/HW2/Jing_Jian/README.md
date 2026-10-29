# Homework 2 — Jing Jian

A reproducible pipeline that diagnoses and cleans stock data, generates a parameterized single-stock risk report, and executes reports for every cleaned ticker.

## Setup

Use Python 3.12.9. From this personal homework directory:

```bash
uv venv --python 3.12.9 HW2
source HW2/bin/activate
uv pip install -r requirements.txt
python -m ipykernel install --user --name hw2-jing-jian --display-name "Python (HW2)"
```

On Windows, activate with `HW2\Scripts\activate` instead. The requirements file records the tested environment, including indirect dependencies. `sqlite3`, `pathlib`, `datetime`, `logging`, `argparse`, `re`, and `zoneinfo` come with Python. Cursor also needs Python/Jupyter notebook support.

Download `data.db` from bCourses and place a local working copy at `data/data.db`. The database must contain `ohlc_hw` and `ohlc`. Never commit any database or credentials.

## Run order

1. Open `01_data_cleaning.ipynb`, select **Python (HW2)**, and run all cells. The working directory must be this personal homework directory. This recreates `ohlc_hw_cleaned` and audit tables in the local database.
2. Open `02_stock_report_template.ipynb` with the same kernel. Set ticker, dates, and database path in its cell tagged `parameters`, then run all cells. Defaults produce an AAPL report for the requested 2024 window.
3. Run the full batch:

```bash
python 03_run_all_stocks.py
```

The batch script resolves default paths relative to its own location, so it can also be invoked from elsewhere. An alternate database can be provided with `--db-path /absolute/path/to/data.db`.

Papermill inserts the supplied parameter values after the tagged cell, executes the template in a fresh kernel, and saves each executed notebook. Scrapbook stores named numeric results inside successful reports; standard text fallbacks allow Cursor to render those outputs. The batch reads only the current run's successful reports, never every file found in the output directory.

## Cleaning decisions and audit

The observed input had 397,711 rows. After ticker normalization, 29,460 exact duplicate copies were removed; no conflicting ticker/date keys remained. Original labels and normalization flags are retained. Duplicate diagnosis and counts remain in the notebook before removal.

Missing trading dates are checked against each ticker's full supplied `ohlc` history, rather than business weekdays. There were no original gaps. Five leading observations with zero opening prices were quarantined: CVX, JNJ, MRK, and PG on 1962-01-02, and MCD on 1966-07-05. The reference shares these invalid opens and no preceding price exists. Their absence is explicitly explained in the after-cleaning calendar report.

The pipeline removes rows with all four raw prices missing before filling. Missing raw prices are forward-filled only within sorted ticker groups and validated against today's bounds afterward. Unresolved prices or bounds are quarantined. Missing volume becomes zero with a flag identifying unknown original volume. No actual raw-price or volume cells were missing; injected examples verify these behaviors and ticker isolation.

Date-specific adjustment factors are `ohlc.adj_close / ohlc.close`, joined by normalized ticker and date. Missing or inconsistent adjusted closes are reconstructed from validated homework raw close using that same-date factor. Factors are never carried across dates. There were 18,374 missing and 10,417 inconsistent adjusted closes, totaling 28,791 repairs. The comparison tolerance is `rtol=1e-6, atol=1e-8`. Original adjusted prices, repair flags, factors, and the source are retained. The dirty dataset is not replaced by the reference table.

The raw-close simple-return screen flags both tails beyond three within-ticker sample standard deviations (`ddof=1`). Zero or undefined standard deviation generates no flags. All 5,056 original flagged moves matched current and previous reference closes. They are retained with flags, without claiming verified splits or news. Shared-source errors remain possible, especially suspicious reversal pairs. No raw-close error was confirmed for interpolation. Recomputing after quarantine yields 5,057 flags; original and updated flags are retained separately. Recorded zero volume is flagged for review, not automatically removed; 74 such rows remain after quarantine.

Validation confirms positive finite retained prices, valid raw OHLC bounds, consistent adjusted close, finite nonnegative volume, valid dates/tickers, unique keys, and balanced row accounting. Final retained count: **368,246**, with **5 quarantined** and **0 all-price-missing removals**. Database dates are ISO `YYYY-MM-DD` strings. The raw table is not written to.

Audit tables are `hw2_quality_before`, `hw2_quality_after`, `hw2_quarantined`, and `hw2_removed_all_prices_missing`. Severity is high for unusable/missing prices and invalid values, medium for return review flags and missing volume, and low for recorded zero volume. Multiple issues can describe one observation. Injected examples exercise missing dates, missing prices/volume, invalid bounds, non-finite/negative prices, adjustment repairs, and both return tails and degenerate histories.

## Report interpretation

All report returns and drawdowns use adjusted close. Daily returns are log returns. Volatility uses sample standard deviation and annualizes by `sqrt(252)`. The 20-return rolling volatility needs 21 prices; the 50-price average retains its warm-up gaps. Reports require at least 21 prices, and explain an unavailable moving average below 50 prices.

Drawdown is price divided by its running peak minus one; maximum drawdown is the minimum of that series. Sharpe uses `(mean_log_return * 252 - 0.02) / (std_log_return * sqrt(252))`; zero volatility or insufficient returns produces NaN/N/A. Total return is last adjusted price divided by first minus one. Compounded annualized return is `exp(sum_log_returns * 252 / n_returns) - 1`. Best/worst and up/down averages use log returns. Positive-day frequency includes zero-return days in its denominator; empty subsets show N/A.

Price coloring compares rolling volatility against its report-wide mean: red above, green at/below, and gray during warm-up. This is retrospective. Summary annualized volatility is whole-period volatility; batch `avg_volatility` is the mean of rolling annualized volatility. Fractions remain numeric in CSVs; percentages are display formatting.

## Batch outputs and coverage

The latest cleaned date is 2024-12-06. The six calendar months before that month are **2024-06-01 through 2024-11-30**. This rule also excludes the latest observation's month when that observation is month-end. One requested window is applied to all stocks. Actual histories differ; many stop on October 18 while others extend through November 29. No trailing prices are extrapolated, so ranks are not comparisons over identical realized histories. Inspect `actual_start`, `actual_end`, and `n_prices` in the CSV.

Outputs go to `reports/YYYY-MM-DD/`, using the run date in America/New_York:

- `{ticker}_report.ipynb`: executed stock report.
- `portfolio_summary.csv`: successful reports' metrics and actual coverage, sorted by Sharpe.
- `portfolio_summary.png`: risk-return scatter, colored by Sharpe, labeling the top five returns.
- `failed_stocks.txt` and `batch.log`: failure tracking and diagnostics.

Failures do not stop other tickers. Undefined Sharpes are excluded from top/bottom rankings; groups can overlap with fewer than 20 valid values. An all-failed run returns a failure status and creates no misleading summary. The verified run on October 29, 2026 generated **30/30 reports**, with no failures. Date-window edge cases and simulated per-ticker/all-failed behavior were also tested.

## Submission scope

Submit the two notebooks, batch script, requirements, README, and local `.gitignore`. Include only three executed samples—**AAPL, IBM, and MRK**—plus the full portfolio CSV (and its chart). These show shorter coverage, full coverage, and a negative-return case. Other generated reports remain available locally and are ignored by Git. Do not commit the `HW2` environment, databases, logs, or credentials.

The cleaning notebook recreates the cleaned table from the locally supplied database. The sample CSV references the complete locally generated batch; only three corresponding notebooks are included in the submission. No commit, push, or pull request has been made as part of preparation.
