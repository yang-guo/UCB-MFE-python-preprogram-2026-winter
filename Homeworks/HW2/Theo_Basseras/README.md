# HW2 – Scalable stock analysis pipeline (Theo Basseras)

Three steps: clean the dirty `ohlc_hw` table, produce a parameterized risk report for one stock, then run that report for every stock with papermill and summarize the portfolio.

## Files

| File | What it does |
|---|---|
| `01_data_cleaning.ipynb` | Diagnoses `ohlc_hw`, repairs it, validates the result and writes `ohlc_hw_cleaned` (plus `ohlc_hw_quarantine` and `ohlc_hw_ticker_audit`) to `data/data.db`. The raw tables are never modified. |
| `02_stock_report_template.ipynb` | Risk report for one ticker and date range. Its first code cell is tagged `parameters` (`ticker`, `start_date`, `end_date`, `db_path`). |
| `03_run_all_stocks.py` | Runs the template for every ticker with papermill, logs failures, collects each report's recorded metrics and writes the portfolio summary. |
| `reports/2026-10-26/` | Sample output of one full run: 3 example reports (IBM, best Sharpe; NVDA, highest volatility; MRK, worst Sharpe), `portfolio_summary.csv`, `risk_return.png` and `failed_stocks.txt` (empty: 30/30 succeeded). The full run produced 30 reports; only 3 are committed, as requested. The run's `run.log` stays local (`*.log` is git-ignored). |
| `requirements.txt` | Python packages (tested with Python 3.12.9). |

## How to run

1. **Data.** Copy the course database (bCourses, also used in Lecture 4) to `data/data.db` inside this folder. It is git-ignored and must not be committed.
2. **Environment.** Either the course environment (`uv sync --locked` at the repository root, then use `.venv`), or `pip install -r requirements.txt`. The notebooks run on the standard `python3` kernel.
3. **Run, from this folder, in this order:**
   ```bash
   papermill 01_data_cleaning.ipynb 01_data_cleaning.ipynb      # 1-2 min; creates ohlc_hw_cleaned
   python 03_run_all_stocks.py                                  # 5-10 min for 30 tickers
   ```
   To try the report on one stock: `papermill 02_stock_report_template.ipynb out.ipynb -p ticker IBM -p start_date 2024-06-01 -p end_date 2024-11-30`.
   For a quick test of the batch: `python 03_run_all_stocks.py --limit 3`.

## What the cleaning found and did

| Issue in `ohlc_hw` | Rows | Decision |
|---|---|---|
| Ticker labels wrapped in HTML (`<a ...><span style="color: blue">KO</span></a>`) | 39,936 (120 labels) | Normalized (HTML stripped, entities decoded, whitespace removed, upper case) **before** any grouping. `ohlc_hw_ticker_audit` maps every original label to its ticker. |
| Exact duplicate records | 29,460 | Removed. |
| Conflicting `(ticker, ts)` keys | 0 | The code would quarantine both rows (no "keep first"). |
| Missing `adj_close` | 18,374 | Rebuilt (see below). |
| Corrupted `adj_close` (4 to 24 times too large) | 10,417 | Detected without the reference, then rebuilt (see below). |
| `open = 0` on a ticker's first observation (PG, CVX, JNJ, MRK 1962-01-02; MCD 1966-07-05) | 5 | Quarantined with a reason: there is no earlier price to repair from, and another ticker's value must not be borrowed. |
| One-day bad prints (TRV 1976-10-28, 1978-12-20, 1981-08-10) | 3 | Confirmed errors, interpolated (see outliers). |
| 3σ return outliers | 5,056 in the raw data, 5,176 on the cleaned prices | Kept and flagged (`flag_return_outlier` comes from the screen on the cleaned prices). |
| Zero-volume days | 75 (74 kept) | Kept and flagged (possible stale data, mostly WMT and JNJ before 1983). |
| Missing raw prices or volume, rows with all prices missing, missing trading days, impossible OHLC ranges, non-finite prices, conflicting keys | 0 | Detectors and repairs run and report zero; their behaviour is demonstrated on small injected examples in the notebook. |

Row accounting: 397,711 raw rows = 368,246 cleaned + 29,460 duplicates removed + 5 quarantined.

### Key design decisions

- **Diagnosis before repair.** The Data Quality Report (`ticker | issue_type | issue_date | description | severity`) is built on the untouched data, then every detector is re-run on the cleaned table. One observation can carry several issues.
- **Severity.** *High*: the value cannot be used (impossible, nonpositive or non-finite price, missing or corrupted adjusted close, conflicting keys). *Medium*: the data would be distorted but values are intact (duplicates, label markup, missing trading days). *Low*: review flags (3σ returns, zero volume).
- **Adjusted close.** The adjustment factor `adj_close / close` is a slowly moving step function that can never exceed 1. A value is suspicious if its factor is above 1, or above 1.5× the median factor of its neighbours (7-row window, same ticker). This catches all 10,417 corrupted values with one false alarm. Missing and suspicious values are then rebuilt as **validated raw close × the date-specific factor from the supplied `ohlc` table**, matched on `(ticker, ts)`, and only where the raw close equals the reference close. The false alarm already equalled the reference value and was kept unchanged. No factor is carried across dates, and the dirty dataset is not replaced by the reference: only the factor is borrowed.
- **Repairs stay visible.** Each repair sets a flag saved with the data: `flag_adj_filled`, `flag_adj_spike_repaired`, `flag_adj_recomputed`, `flag_interpolated`, `flag_ohlc_repaired`, `flag_price_filled`, `flag_volume_filled`, plus the review flags `flag_return_outlier` and `flag_zero_volume`, which describe the cleaned values.
- **Price repairs** (none needed in this data, behaviour shown on injected examples): impossible, nonpositive or non-finite OHLC rows are blanked and forward-filled from the same ticker's previous valid day (Option 1); missing volume becomes 0 with a flag (unknown volume is not evidence of no trading); rows with all four raw prices missing are dropped before any filling; leading gaps are quarantined, never extrapolated. Forward-filling one price can break that day's range, so the bounds are rechecked after the fill and a row that breaks them takes the previous day's whole bar. The adjusted close of every repaired row is recomputed as repaired close × that date's reference factor, so one adjustment convention holds.
- **Rechecks after every step.** After each repair step the notebook recounts missing and invalid prices, broken OHLC bounds, missing adjusted closes and adjustment-factor spikes; all reach 0 at the end (table in section 1.3, assertions in the validation section).
- **Conflicting keys** (none in this data, shown on an injected example): every row of a `(ticker, ts)` key with different values is quarantined; no version is preferred.
- **Storage.** `ts` is kept as ISO `YYYY-MM-DD` text so the report's SQL date filters work. The raw `ohlc_hw` and `ohlc` tables are left unchanged (checked in the notebook).

### How outliers were handled, and why

The 3σ screen (per ticker, sample std `ddof=1`, both tails, undefined returns excluded) flags 5,056 daily moves. None is changed just for being large:

1. **Cross-check.** The raw prices of every flagged row are identical to the supplied `ohlc` table, so the homework copy did not alter them. Both tables come from the same vendor, so this is not proof that the prices are right.
2. **Pattern check for bad prints.** A genuine move persists; a bad print jumps and comes straight back. Three TRV days double all four prices with zero volume and return to the previous level the next day. They are treated as confirmed data errors and interpolated between their neighbours (prices and adjusted close). Interpolation is only done inside a ticker's history, between valid days.
3. **Everything else is kept and flagged.** Example: MCD drops 75% on 1968-01-02 and stays at the new level, adjusted close included. It may be an unadjusted split in the vendor history, but nothing in the data proves it, so it is kept, flagged and recorded as an uncertainty. No split or news event is claimed without evidence.

The number of outliers rises slightly after cleaning (5,056 to 5,176): TRV's three ±100% bad prints inflated its standard deviation, so once they are fixed, more ordinary TRV moves cross 3σ. `flag_return_outlier` marks the 5,176 rows flagged on the cleaned prices.

## Assumptions

- **Trading calendar:** the supplied `ohlc` dates for each ticker, over that ticker's own history, are the reference calendar. Business weekdays alone would include exchange holidays.
- **Prices:** the vendor `close` is split-adjusted but not dividend-adjusted; `adj_close` adjusts for both. Reports therefore use `adj_close` for returns and drawdown.
- **Report conventions** (from the brief): daily log returns; 252 trading days a year; 2% annual risk-free rate in the Sharpe ratio; 20-day rolling volatility; 50-day moving average. Undefined values (zero volatility, too few returns, empty up/down subsets) show as `N/A`, never as 0. A report needs at least 21 prices (20 returns) or it fails with a clear message.
- **Batch window:** the six complete calendar months before the month of the latest cleaned observation, anchored to the data, not the clock. Latest observation 2024-12-06 gives **2024-06-01 to 2024-11-30**. The output folder is named after the run date, independently of that window.
- **Coverage differences:** 22 of the 30 tickers have no data after 2024-10-18, while the other 8 run to 2024-11-29. Each report prints its actual first and last observation dates and says so when its ticker stops before the window's last market day; the summary CSV keeps `first_date`, `last_date` and `n_days`, and `risk_return.png` draws these tickers as squares. Nothing is extrapolated, so short histories cover about 4.5 months instead of 6: compare them with care.

## How to read the reports

- **Top left, price and 50-day MA:** adjusted close, coloured red where the 20-day volatility is above its mean over the report, green at or below, and gray during the volatility warm-up. The colouring is retrospective (the mean uses the whole window). The dashed line is the 50-day moving average, which starts once 50 prices exist.
- **Top right, daily returns:** daily log returns, green up and red down.
- **Bottom left, rolling volatility:** annualized 20-day volatility with its mean (dashed), shaded red above the mean and green below.
- **Bottom right, drawdown:** distance from the running peak. The black dot marks the maximum drawdown, the worst peak-to-trough fall over the window.
- **Summary table:** total return = last/first adjusted close − 1; annualized return = exp(Σ log returns × 252 / n) − 1; annualized volatility = std of log returns × √252 (whole period, unlike the mean rolling volatility used in the portfolio summary); best/worst day and average up/down day use log returns; positive days = positive returns / all non-missing returns.
- **Portfolio summary** (`portfolio_summary.csv`): one row per successful report with `avg_daily_return` (mean daily log return), `avg_volatility` (mean of the 20-day rolling volatility), `max_drawdown`, `sharpe_ratio`, `total_return` (all fractions), the coverage dates and the report path. It is sorted by Sharpe ratio; undefined Sharpe ratios are listed last and not ranked. Values are read from each report's recorded outputs (scrapbook), only for reports that succeeded, never by scanning the folder.
- **`risk_return.png`:** average volatility (x) against total return (y), coloured by Sharpe ratio, with the five best returns labelled. Circles have data over the whole window; squares are the tickers whose data stops on 2024-10-18 (shorter period).

## Tools

Written with the help of Claude Code (an AI coding assistant), as the course encourages. The design decisions above were reviewed and are explained in the notebooks.
