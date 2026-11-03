# Homework 2 — Stock analysis pipeline (Shashank Mishra)

`ohlc_hw` (dirty vendor data) → **cleaned, validated table** → **parameterised risk report** → **batch of 30 reports + portfolio summary**.

| File | Purpose |
|---|---|
| `01_data_cleaning.ipynb` | Inspect, diagnose, repair, validate, and write `ohlc_hw_cleaned` (executed, outputs included) |
| `02_stock_report_template.ipynb` | Papermill template: one ticker, one window, 2×2 risk figure, summary table |
| `03_run_all_stocks.py` | Runs the template for every cleaned ticker and builds the portfolio summary |
| `requirements.txt` | Dependencies |
| `reports/<run date>/` | 3 sample reports, `portfolio_summary.csv`, coverage table, risk/return chart |

## How to run

```bash
pip install -r requirements.txt          # or: uv pip install -r requirements.txt
cd Homeworks/HW2/Shashank_Mishra

# 1. Clean the data (creates data/data.db with ohlc_hw_cleaned). Use Jupyter "Run All", or headless:
python -m papermill 01_data_cleaning.ipynb 01_data_cleaning.ipynb --cwd .

# 2. Generate the reports for every ticker (uses this same interpreter for every report)
python 03_run_all_stocks.py
```

**Database.** `01_data_cleaning.ipynb` looks for a SQLite file that really contains both `ohlc` and `ohlc_hw` (it checks the table names — the repository root may also hold an unrelated `data.db` with a `signals` table) in this order: `$HW2_SOURCE_DB`, `data/data.db`, `Lectures/Lecture 4/data/data.db`, `<repo root>/data-1.db`, `<repo root>/data.db`. It copies the match to `data/data.db` (git-ignored) and only ever writes to that **working copy**; the source file and the `ohlc_hw` table are never modified (the notebook re-checks a row/volume/close fingerprint after writing). **No database or credentials are committed.**

Step 1 must run before step 2, and both must use the same interpreter. `03_run_all_stocks.py` registers a throw-away kernel that launches `sys.executable`, so no user-level Jupyter configuration is touched. Optional flags: `--tickers IBM NVDA`, `--demo-failure` (adds a bogus ticker to show failure handling), `--run-date`, `--db`, `--out-root`.

The notebook can also be run on its own for any stock: set `ticker`, `start_date`, `end_date` in the cell tagged `parameters` (and `db_path` if the database is elsewhere).

## What the data actually contained

Findings from the real `ohlc_hw`, checked against the supplied `ohlc` reference (numbers from the executed notebook):

| Finding | Count | Treatment |
|---|---|---|
| Raw ticker labels that are really 30 stocks (HTML wrapped, e.g. `<a …><span …>NVDA</span></a>`) | 150 labels → 30 tickers | Normalised first; audit table `ohlc_hw_ticker_audit` |
| Exact duplicate rows (differ only in label markup) | 29,460 | Dropped; conflicting `(ticker, ts)` keys: **0** |
| Missing trading sessions vs each ticker's reference calendar | **0** | Detector proven on an injected 1-day gap |
| `open = 0` (nonpositive price, violates `low ≤ open ≤ high`) | 5 (first rows of PG, CVX, JNJ, MRK, MCD) | Rebuilt, flagged `fix_open` |
| `adj_close` missing | 18,374 (after de-dup) | Rebuilt from reference factor |
| `adj_close` present but wrong (4×–24× too large) | 10,417 | Rebuilt from reference factor |
| Zero-volume days | 75 (after de-dup; also present in the reference) | Kept, flagged `flag_zero_volume` |
| 3σ close-to-close return days | 5,056 of 368,221 returns (1.4 %) | Kept, flagged — see below |
| Rows quarantined | **0** | – |

Row accounting: `397,711 raw − 29,460 exact duplicates − 0 quarantined = 368,251 retained` (asserted in the notebook).

## Key design decisions

* **Evidence first.** The reference table `ohlc` is allowed by the assignment, so I use it to *confirm* a fault before changing a value. In this delivery the raw OHLC and volume equal the reference exactly (except the five zero opens), so the raw `close` is validated; the dirt is concentrated in `adj_close`.
* **Checks as a registry.** Detectors are `Check(issue_type, severity, mask, detail)` objects; the quality report loops over them, so one observation can raise several issues. The report has the required schema `ticker | issue_type | issue_date | description | severity`, is computed **before** repairs, and is saved as `ohlc_hw_quality_report`. Severity: *high* = would corrupt returns/ordering (invalid or impossible prices, wrong `adj_close`, bad volume); *medium* = needs a repair or a look (missing `adj_close`/volume, zero volume, missing session); *low* = review flag only (3σ returns).
* **Two independent `adj_close` tests.** (1) vs the reference: `adj_close` must equal `close × (ref_adj / ref_close)` on the **same date** (10,417 hits). (2) A self-contained *spike* test that needs no reference — the `adj/close` factor jumps and returns within a day (9,816 hits, i.e. 94 % of the corrupted values). Six isolated factor jumps survive the repair; they equal the reference's own values, so the heuristic loses to the trusted source and they are kept.
* **Adjusted close.** Rebuilt as `close × same-day reference factor`. No factor is forward-filled or carried across dates, so no unverified corporate action is bridged, and raw OHLC is never replaced by the reference.
* **Volume.** Unknown (missing/negative/non-finite) → 0 **with** `fix_volume`; an *observed* zero stays 0 and is flagged separately. Unknown volume is not evidence of no trading.
* **Impossible OHLC.** If a close that the reference validates sits outside `[low, high]`, the contradicted `high`/`low` is rebuilt; otherwise the open/close is the suspect field. Missing `open`/`close` → previous close **if it fits today's range**, else the midpoint of high and low (plain forward-fill can leave an impossible row). Documented as Options 2/3 of the assignment. The five zero opens are leading rows with no previous close, so they use the range midpoint.
* **Everything is flagged and traceable.** `ohlc_hw_cleaned` carries `source_rowid`, `ticker_raw`, `fix_open/high/low/close/adj_close/volume`, `flag_zero_volume`, `flag_extreme_return`, `extreme_return_class`, and `ticker_relabelled`. Rows that cannot be defended go to `ohlc_hw_quarantine` with a reason (none in this delivery; all branches are exercised on the injected example).
* **Re-validation.** After repair the notebook asserts finite positive prices, valid OHLC, finite non-negative volume, unique `(ticker, ts)`, ISO dates, and `adj_close = close × reference factor`, then re-runs the full quality report and shows only review flags remain.

## How I handled outliers (and why)

The screen is exactly as specified: simple close-to-close returns within each ticker, `abs(r − mean) > 3·std` with `ddof = 1`, both tails; undefined returns are excluded and a zero/undefined `std` produces no flags. It is a **review flag, not a deletion rule**.

To decide what to do with the 5,056 flagged days I asked two questions: *does the raw close agree with the reference?* (yes, on **all** of them → nothing is a confirmed data error) and *is the move also visible in the adjusted series?* (4,850 days: yes → genuine or unconfirmed market moves; 206 days: no → splits/dividends). With no evidence of error, **nothing was interpolated** — interpolating a real gap would invent prices. Interpolation is implemented, restricted to *confirmed* errors (a close that contradicts the reference), interior-only, within a sorted ticker, clipped to the day's range, and demonstrated on an injected spike (168.87 → 93.96 vs a true 93.81).

## Assumptions

* The per-ticker dates in `ohlc` are the reference calendar over that ticker's supplied history (weekdays alone would include exchange holidays).
* `ohlc.adj_close / ohlc.close` on the same date is a trusted adjustment factor.
* Annualisation uses 252 days; the risk-free rate is the course's 2 %.
* The six-month window is the six calendar months **before** the month of the latest cleaned observation (anchored to the data, not the clock): latest = 2024-12-06 → **2024-06-01 to 2024-11-30**.

## How to read the reports

* Performance and risk use **adjusted close**; returns are **daily log returns**. Total return = `last_adj / first_adj − 1`; annualised return = `exp(Σ log returns × 252 / n) − 1`; annualised volatility (whole period) = `std × √252`. *Mean rolling volatility* (Part 3's `avg_volatility`) is a different number — the average of the 20-day estimates.
* Top-left: the price line is red where the rolling volatility is above the report's own mean, green otherwise, grey while the 20-day window is still filling; the dashed line is the 50-day average (blank until 50 prices exist). The colouring is retrospective.
* Maximum drawdown is `drawdown.min()`; the plotted series is the per-date drawdown from the running peak and can recover.
* Sharpe = `(mean × 252 − 0.02) / (std × √252)` from daily log returns; undefined (NaN / `N/A`) with fewer than two returns or zero volatility, and then excluded from rankings.
* **Coverage differs by ticker.** All reports use one common requested window, but only 8 tickers (AMZN, CVX, DIS, IBM, KO, MRK, NKE, PG) have data through November; the other 22 stop on 2024-10-18. Each report prints and footnotes its *actual* first/last date and observation count. `coverage_summary.csv` lists them, and the risk/return chart marks short histories with diamonds. Do not treat a shorter history's total return as a full-window return.
* CSV metrics are fractions (0.05 = 5 %); percentages appear only in labels.

## Submitted sample output

`reports/<run date>/` contains `IBM_report.ipynb` (full window), `NVDA_report.ipynb` and `KO_report.ipynb`, plus `portfolio_summary.csv` (the seven required columns), `coverage_summary.csv`, `risk_return_profile.png`, and `failed_stocks.txt`. The full local batch generated all 30 reports; only these samples are committed.
