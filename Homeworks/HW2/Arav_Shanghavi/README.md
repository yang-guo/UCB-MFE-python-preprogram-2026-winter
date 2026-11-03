# HW2 – Stock Analysis Pipeline (Arav Shanghavi)

| File | What it does |
|---|---|
| `01_data_cleaning.ipynb` | Diagnoses `ohlc_hw`, repairs it with flags, validates, writes `ohlc_hw_cleaned` |
| `02_stock_report_template.ipynb` | Parameterized single-stock risk report (first code cell tagged `parameters`) |
| `03_run_all_stocks.py` | Runs the template for every ticker with papermill and builds the portfolio summary |
| `reports/2026-10-31/` | Sample reports (IBM, NKE, NVDA), `portfolio_summary.csv`, `portfolio_summary.png`, `failed_stocks.txt` |
| `requirements.txt` | Dependencies |

## How to run

1. `uv sync --locked` from the repo root (or `pip install -r requirements.txt`), and use that environment as the notebook kernel.
2. Put the bCourses database at `Lectures/Lecture 4/data/data.db`.
3. Run `01_data_cleaning.ipynb` top to bottom with this folder as the working directory (about a minute). It copies the course database to `data/data.db` on the first run and only ever writes to that copy. It recreates `ohlc_hw_cleaned` plus three audit tables: `ohlc_hw_quality_report`, `ohlc_hw_quarantine` and `ohlc_hw_ticker_audit`.
4. Optional single report:
   ```bash
   papermill 02_stock_report_template.ipynb out.ipynb -p ticker NVDA -p start_date 2024-06-01 -p end_date 2024-11-30
   ```
5. Batch:
   ```bash
   python 03_run_all_stocks.py                          # all tickers, ~2 min
   python 03_run_all_stocks.py --tickers AAPL FAKE      # shows the failure handling
   ```
   Output goes to `reports/<run date>/`. Paths are resolved from the script's location, so it can be launched from anywhere.

`data/` and all but three reports are git-ignored. Nothing writes to the course database or to the raw `ohlc_hw` table; the notebook checks the raw table's hash before and after.

## What the data looked like

| Issue | Count | Handling |
|---|---|---|
| HTML markup / case in ticker labels | 150 labels → 30 tickers | normalized before anything else, mapping saved in `ohlc_hw_ticker_audit` |
| Exact duplicate rows | 29,460 | removed (`n_raw_copies` kept) |
| Conflicting `(ticker, ts)` keys | 0 | would be quarantined, all versions |
| Missing `adj_close` | 18,374 | rebuilt from the same-date reference factor |
| `adj_close` inconsistent with its factor (4–24x too large) | 10,417 | rebuilt from the same-date reference factor |
| `open = 0` on a ticker's first day (CVX, JNJ, MRK, PG 1962; MCD 1966) | 5 | quarantined: leading gap, nothing to fill from |
| Confirmed bad ticks (TRV x3, AXP x1) | 4 | AXP interpolated; TRV quarantined (see below) |
| Single-day jumps in the *reference* factor | 7 | flagged for review only, not changed |
| 3σ return flags | 5,056 | kept, flagged |
| Zero volume | 75 | kept, flagged (71 after quarantine) |
| Missing raw prices / volume, `high < close`, missing trading days | 0 | detectors shown on injected examples |

Row accounting: 397,711 raw = 368,243 retained + 29,460 duplicates + 8 quarantined.

## Key design decisions

- **Diagnose first, then repair.** Every detector writes to one quality report (`ticker | issue_type | issue_date | description | severity`, several issues per row allowed) before anything changes. The same detectors are rerun on the cleaned table.
- **Severity.** *High* means unusable or impossible values that would corrupt metrics. *Medium* means missing information or something that needs review. *Low* means harmless (duplicate copies, zero volume). Details are in the notebook.
- **Trading calendar.** Each ticker's dates in the supplied `ohlc` table, within that ticker's own first/last date. Business days would count exchange holidays as gaps.
- **Adjusted close.** One convention: `adj_close = close x factor`, with the factor (`adj_close / close`) taken from the reference row for the *same* date. It's never carried forward across dates. The reference is only used for factors and the calendar; raw prices come from `ohlc_hw`.
- **The reference factor is treated as trusted**, as the assignment says. A few reference factors jump for a single day (e.g. AAPL 1990-10-19 is exactly 1.0 between two days of 0.796; DIS alternates between two values in 1974–75). Those rows are kept as delivered with `adj_factor_review_flag = 1`; I don't invent a replacement factor.
- **Missing values (none in this delivery).** Rows with all four raw prices missing are dropped first. A missing or invalid price is forward-filled within the ticker. If that leaves no earlier price, or the filled bar breaks `low <= open, close <= high`, the row is quarantined instead. Missing volume becomes 0 with `volume_filled = 1`.

## Outliers

- **Impossible price relationships:** Option 1. The whole bar is set to NaN and forward-filled from the previous day. The data can't tell which of the four prices is wrong, and a copied bar is always internally consistent. None occur in the real data; it's demonstrated on an injected example.
- **3σ returns:** simple raw-close returns within each ticker, sample std, both tails, undefined returns excluded. This is a review flag. Most of the 5,056 flags are real moves: Black Monday 1987, NVDA in 2000, AAPL 2000-09-29 and so on, on heavy volume with persistent new levels.
- **Confirmed errors.** TRV 1976-10-28, 1978-12-20, 1981-08-10 and AXP 1972-12-14 are flat bars (O=H=L=C) with zero volume, and the next close returns to within 2% of the previous one. The TRV closes are exactly 2x their neighbours.
  - **AXP** is linearly interpolated, and its adjusted close is rebuilt with the same-date reference factor.
  - **TRV** is quarantined. On those three days the reference adjusted close spikes too, so the interpolated close times the reference factor would leave a fake spike in the adjusted series, and there is no other trusted factor. The spec allows a documented quarantine when repair isn't defensible. The three dates then show up as missing trading days after cleaning, and are reported as such.
- **MCD 1968-01-02 (−75%)** persists and looks like an unadjusted split, but nothing in the data confirms it, so it's kept and flagged rather than "fixed".
- After cleaning, the 3σ count goes up to 5,174. Removing TRV's ±100% ticks lowers its std, so more ordinary TRV days cross 3σ.

## Report and batch

- Returns and drawdowns use adjusted close and daily log returns. Volatility is the 20-day sample std x √252. Sharpe is `(mean x 252 − 2%) / (std x √252)`, shown as N/A when volatility is zero. Maximum drawdown is the minimum of the per-date drawdown. Warm-up periods are left blank, never zero-filled.
- The report refuses an unknown ticker, a reversed window, or fewer than 21 prices.
- **Window:** the six calendar months before the month of the latest cleaned date. That date is 2024-12-06, so the window is 2024-06-01 to 2024-11-30. It's the same for every ticker and doesn't depend on the computer clock.
- Each ticker runs in its own `try/except`. Failures go to `failed_stocks.txt` and the batch continues. If every ticker fails, the script exits with an error and builds no summary.
- Metrics are passed back with scrapbook, read only from the notebooks that succeeded in the current run.

## Assumptions

- 252 trading days a year, a 2% risk-free rate, and the course's Sharpe approximation.
- **Coverage differs.** 22 of the 30 tickers stop on 2024-10-18 (about 4.5 months of the window); only AMZN, CVX, DIS, IBM, KO, MRK, NKE and PG run to 2024-11-29. Each report states its actual dates, the CSV includes `actual_start`, `actual_end` and `n_prices`, and the scatter plot draws short-coverage tickers as squares. Their total returns cover a shorter period, so they aren't directly comparable with the full-window group. Nothing is extrapolated.

## Reading the reports

- **Price and 50-day MA:** red where 20-day volatility is above its average over the report, green at or below it, gray before there are 20 returns. This is retrospective coloring.
- **Daily returns:** green up days, red down days.
- **Rolling volatility:** shaded red above the mean and green below.
- **Drawdown:** distance below the running peak, with the worst point marked.
- **Summary table:** total and annualized return, whole-period volatility, Sharpe, max drawdown, best/worst day, share of positive days, average up/down day. Annualizing a 4–6 month window is a rate, not a forecast.

**This run (2024-06-01 to 2024-11-30, 30/30 succeeded):** IBM had the best Sharpe (3.22, +39.9%), followed by HD, SHW, WMT and MCD. MRK, BA and NKE were the only losers. NVDA's average volatility (61%) was about three times its peers'. Six months of data is too short for the Sharpe rankings to mean much beyond this period.
