# HW2: Scalable Stock Analysis Pipeline

This folder contains the HW2 submission: a cleaning/validation notebook, a parameterized single-stock risk-report notebook, and a Papermill batch runner.

## Local setup and run order

Download the course database from bCourses (it is normally stored at `Lectures/Lecture 4/data/data.db` in a local course checkout). For the default workflow, put a source copy at the repository root as `data.db`; alternatively, put a working copy directly at `Homeworks/HW2/Mrityunjay_Balkrishnan/data/data.db`. If the working copy is absent, the cleaning notebook copies the root database there and writes only to that copy. The source database and its raw `ohlc_hw` table remain read-only. Never commit either database, credentials, or the virtual environment.

From this directory, create/activate a dedicated environment and install dependencies:

1. `uv venv HW2 --python 3.12.9`
2. `source HW2/bin/activate` (macOS/Linux; on Windows use `HW2\\Scripts\\activate`)
3. `uv pip install -r requirements.txt`
4. Open and run `01_data_cleaning.ipynb` top to bottom. It inspects the raw table, uses the clean `ohlc` table as the ticker/date reference and adjustment-factor source, builds the quality report, runs small fixture checks, and saves `ohlc_hw_cleaned` plus pre/post quality and quarantine audit tables to your working copy.
5. Open `02_stock_report_template.ipynb`; set `db_path` if your working copy is elsewhere, then run it or execute it through Papermill.
6. Run `python 03_run_all_stocks.py --db-path data/data.db` to create the six-complete-month batch, per-ticker reports, failure log, manifest, CSV, and risk/return chart under `reports/YYYY-MM-DD/`.

The default notebook and script paths are relative to this submission directory. No API keys or network data are required.

## Data decisions and audit trail

- Tickers are stripped of markup and whitespace and uppercased. Original ticker text is retained. Exact duplicate rows are dropped; conflicting rows with the same normalized `(ticker, ts)` are quarantined rather than arbitrarily selected.
- Ticker-specific dates in the supplied clean `ohlc` table are used as the reference calendar, restricted to each `ohlc_hw` ticker's own first and last observed dates. This detects absent rows without assuming weekdays are exchange sessions. The detector also has a tiny intentionally gapped fixture. A missing row is reported; no price is fabricated for it.
- Diagnostics are created before repairs. OHLC price anomalies are reviewed and repaired only when the affected field can be identified: invalid/out-of-range fields become missing, within-ticker forward fill is used only from a prior observation, and high/low are re-bounded around repaired open/close. Leading unresolved values and key conflicts are quarantined. Outlier returns are review flags, not automatic deletions: without corporate-action/news evidence, the notebook records the uncertainty and preserves otherwise valid observations.
- Missing volume is set to zero only with a `volume_repaired` flag (zero is not claimed as observed volume). Invalid negative/non-finite volume is similarly flagged and replaced with a nonnegative placeholder.
- Adjusted close is kept on the vendor convention when valid. If it needs reconstruction, a date-matched valid adjustment factor from the supplied `ohlc` reference table is preferred. A row-local vendor factor is allowed only when its raw close is valid. Factors are never forward-filled across dates; unresolved adjusted prices are quarantined.
- The cleaned table stores repair flags and original ticker labels, uses ISO `YYYY-MM-DD` dates, and is validated for unique keys, finite positive prices, nonnegative finite volume, and OHLC bounds. The source table is not changed.

## Interpreting reports

Reports use adjusted close. Returns, daily-return summaries, and Sharpe inputs are daily log returns; total return uses the first and last adjusted close. Annualization uses 252 trading days. Rolling estimates keep their warm-up gaps. Each report states its actual first and last observation dates and sample count; the batch applies one common six-complete-calendar-month request window, so tickers with shorter histories are not extrapolated and are not directly comparable without considering coverage. Sharpe is `N/A` when not estimable. The portfolio CSV retains fractions (not formatted percentages); plots format values for display.

The batch runner creates a metrics JSON sidecar only after each Papermill report succeeds. Only those successful reports from the current run are included in `portfolio_summary.csv`; partial, stale, and failed notebooks are not swept from the output directory. If every report fails, the CSV has headers but no rows and no misleading dashboard is drawn. `failed_stocks.txt` records failed ticker/error pairs, and `manifest.json` records the exact current run.

## Current data run

The cleaning notebook was run against the supplied root database using the personal working copy. It found 397,711 dirty rows across 30 normalized tickers (1962-01-02 through 2024-12-06), zero missing ticker-dates versus the ticker-specific clean `ohlc` calendar, 19,855 missing adjusted-close cells, 29,460 exact duplicate rows, and 11,265 date-matched adjusted-close factor mismatches. It retained 368,246 valid rows, quarantined 5 unresolved invalid rows, repaired 28,791 cells (including 10,417 retained adjusted-close mismatches), and saved the cleaned and audit tables to `data/data.db` under this homework folder. All retained adjusted closes now agree with the matching clean-reference factor. The root `data.db` was not written. The template's default AAPL example requests calendar year 2024; the database provides 202 observations through 2024-10-18, with +27.07% total return, 1.187 Sharpe, and −15.35% maximum drawdown.

The all-stock batch completed on 2026-10-25 with 30 of 30 reports succeeding and no failed tickers. Using the latest cleaned observation (2024-12-06), the runner selected the six complete months from 2024-06-01 through 2024-11-30. The run produced one report and metrics sidecar per ticker, a 30-row `portfolio_summary.csv`, the risk-return chart, an empty `failed_stocks.txt`, and a manifest. The AAPL batch report contains 97 observations through 2024-10-18; this shorter coverage is present in the source data and is reported rather than extrapolated. The submission includes AAPL, IBM, and MRK sample report notebooks, the summary CSV and chart, and the empty failure log. The local database and remaining generated reports are excluded from version control.

## Sample outputs

The assignment asks for 2–3 real example reports and a portfolio CSV. They are generated by the batch script from the local cleaned table. Never fabricate sample market prices or metrics.
