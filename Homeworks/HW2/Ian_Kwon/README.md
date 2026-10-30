# HW2: Scalable Stock Analysis Pipeline (Ian Kwon)

Three deliverables, run in order:

| Step | File | Output |
|---|---|---|
| 1 | `01_data_cleaning.ipynb` | `ohlc_hw_cleaned` (plus quarantine, quality-report and ticker-audit tables) in `data/data.db` |
| 2 | `02_stock_report_template.ipynb` | Parameterized single-stock risk report (`parameters` cell tagged for Papermill) |
| 3 | `03_run_all_stocks.py` | `reports/{run_date}/`: one report per ticker, `portfolio_summary.csv`, `risk_return_profile.png`, `failed_stocks.txt`, `run_log.txt` |

## How to run

Use the course environment (`uv sync` at the repository root creates `.venv/` with every package in `requirements.txt`).

1. **Database.** Put the bCourses `data.db` at `Lectures/Lecture 4/data/data.db` (the course location). The cleaning notebook copies it to this folder's `data/data.db` the first time it runs; you can also place a copy there yourself. Only the working copy is written to, and the raw `ohlc_hw` table is never modified (the notebook checks this).
2. **Clean.** Open `01_data_cleaning.ipynb` with the `.venv` kernel and choose *Restart & Run All*. This recreates `ohlc_hw_cleaned` from scratch.
3. **Single report (optional).** Open `02_stock_report_template.ipynb`, edit `ticker`, `start_date` and `end_date` in the tagged parameters cell, and run all cells.
4. **All stocks.** From this folder, run:

   ```bash
   python 03_run_all_stocks.py                      # all 30 tickers, about 3 minutes
   python 03_run_all_stocks.py --tickers AAPL MSFT  # a subset
   ```

   The script works from any directory, since paths come from the script's own location. It registers a temporary Jupyter kernel that runs the *same* Python interpreter. This matters because Papermill's default `python3` kernel can point at a different Python installation without pandas. The script exits with code 1 if every report fails.

Committed sample output: `reports/2026-10-30/` holds three reports: **IBM** (best Sharpe, full window), **MRK** (worst Sharpe, full window) and **AAPL** (data ends 2024-10-18, a shorter-coverage example). It also holds the full `portfolio_summary.csv` for all 30 tickers, the risk-return plot, the run log and an empty `failed_stocks.txt`. The other 27 reports are generated locally and ignored by `.gitignore`, as is the database.

## What the data contained

Found by comparing `ohlc_hw` with itself and, for verification, with the supplied `ohlc` table:

| Issue | Count | Handling |
|---|---|---|
| HTML-wrapped ticker labels (`<a ...><span ...>AAPL</span></a>`) | 39,936 rows; 150 labels → 30 tickers | Normalized first (strip tags, trim, upper-case). The original label is kept in `raw_ticker`, plus an audit table |
| Exact duplicate rows | 29,460 | Removed |
| Conflicting `(ticker, ts)` keys | 0 | Would be quarantined |
| Missing `adj_close` | 18,374 | Rebuilt as `close ×` the reference adjustment factor for that date |
| Corrupted `adj_close` (4× to 24× too large) | 10,417 | Detected without the reference, then rebuilt like the missing values |
| `open = 0` on a ticker's first day (CVX, JNJ, MRK, PG 1962; MCD 1966) | 5 | Quarantined (unresolved leading gap) |
| Reverting bad ticks (flat, zero-volume bar; price reverts next day) | 4 (TRV ×3, AXP ×1) | Interpolated, with the adjustment factor interpolated too |
| Zero-volume days (1962–1982) | 75 | Kept and flagged |
| Missing raw prices, missing volume, `high < close`, missing trading days | 0 | Demonstrated on an injected example (Section 4 of the notebook) |

Rows: 397,711 raw = 368,246 retained + 29,460 duplicates removed + 5 quarantined.

## Key design decisions

- **Detect before repairing.** The Data Quality Report (`ticker | issue_type | issue_date | description | severity`) is built from the data before any change. It is saved as `ohlc_hw_quality_report`, and the same detectors are re-run after cleaning.
- **Every change is flagged.** `ohlc_hw_cleaned` has `*_repaired` flags for each price column and for volume, a `repair_notes` text column, `zero_volume_flag`, `extreme_return_flag` and `extreme_return_category`. Rows that could not be repaired defensibly go to `ohlc_hw_quarantine` with a `reason`; they are never silently dropped.
- **Severity.** *High*: the value is impossible or wrong and would corrupt the metrics (invalid price, OHLC bound violation, corrupted `adj_close`, conflicting key). *Medium*: information is missing or needs review (missing `adj_close`, missing volume or trading day, extreme return). *Low*: cosmetic or plausible (ticker markup, exact duplicate, zero volume).
- **Calendar.** Expected trading days for each ticker are that ticker's dates in `ohlc`, as the assignment allows; weekdays alone would include exchange holidays. Missing days are reported, never filled with invented prices.
- **Adjusted close.** I kept one adjustment convention: `adj_close = close × factor`, with the factor (`ohlc.adj_close / ohlc.close`) taken from the reference for the **same** `(ticker, ts)`. A factor is never forward-filled across dates, since that could carry it across a split or dividend. Only the flagged rows use the reference; the dirty table is not replaced wholesale.
- **Detecting corrupted `adj_close` without the reference.** The adjustment factor `adj_close / close` changes rarely. A dividend or split moves it to a new level and it stays there (a step). A corrupted value is an isolated spike instead. A value is flagged when its factor differs by more than 3× from the median factor of **both** the previous 10 and the next 10 days. This flagged exactly the 10,417 values that disagree with `ohlc`: no false positives, none missed.
- **Missing prices** (none in the real data): drop rows with all four raw prices missing *before* filling. Otherwise forward-fill within the ticker, then re-check the bounds (see below). Leading gaps are quarantined, never back-filled or borrowed from another ticker.
- **Impossible OHLC relationships:** I chose **Option 1 with a range check**. The bad open or close is set to missing and forward-filled; if the carried price would itself lie outside today's `[low, high]`, the day's midpoint `(high + low) / 2` is used instead (Option 3). Forward fill alone can create a new violation. `high < low` is quarantined, because we can't tell which value is wrong.
- **Missing volume** is filled with 0 **and** flagged `volume_repaired`: unknown volume is not evidence of no trading.
- **Dates** are stored as ISO `YYYY-MM-DD` text, so the report's SQL `BETWEEN` filters work. `(ticker, ts)` has a unique index.

## How outliers were handled, and why

The 3σ screen is run on simple raw-close returns within each ticker: `|r − mean| > 3 × std`, with the sample std (`ddof=1`), both tails, undefined returns excluded, and no flags when the std is zero. It flags 5,056 returns before cleaning. A statistical outlier is a reason to **investigate**, not to delete. Each flag was checked for evidence:

1. **Confirmed data error → interpolate** between valid neighbours (inside the series only, no extrapolation). It counts as confirmed only with evidence beyond the size of the move: (a) the row disagrees with the reference table, or (b) it is a **reverting bad tick**, a flat zero-volume bar whose price is back within 5% of the previous close the next day. TRV, for example, closes at exactly 2× its neighbours on 1976-10-28, 1978-12-20 and 1981-08-10 with zero volume. The reference has the same wrong values, so for (b) the adjustment factor is also interpolated between the neighbouring days, but only when those two factors agree within 1% (no corporate action in between).
2. **Corporate action → keep.** The adjustment factor changed by more than 5% and the adjusted return is not extreme. Example: MSFT on 2004-11-15 shows a raw −8.6% but an adjusted +1.9%, consistent with its November 2004 special dividend. There are few of these, because this data's `close` is already split-adjusted.
3. **Traded move → keep and flag.** Volume > 0 and a real intraday range, e.g. October 1987, September 2008, March 2020, AAPL on 2000-09-29. The price is supported by trading; the cause is not claimed.
4. **Unverified → keep and flag.** No trading evidence (flat bar or zero volume), mostly in very old data, e.g. MCD's lasting −75% level shift on 1968-01-02. There is not enough evidence to call it an error, so the uncertainty is recorded rather than "fixed".

A lesson from this data: the bad ticks **masked** other outliers. TRV's ±100% days inflated its standard deviation so much that, once they were repaired, 119 more TRV returns crossed the 3σ line (87 flags → 206). A full-history 3σ screen should not be the only check, and it must be re-run after repairs.

## Assumptions

- `ohlc` is a trusted source for date-specific adjustment factors and the trading calendar. It is **not** treated as proof that a raw price is correct, since it contains the same TRV bad ticks.
- In this data, `close` is split-adjusted, but `adj_close` also adjusts for dividends.
- The report uses adjusted close for all returns and drawdown, daily log returns for volatility, Sharpe and daily statistics, 252 trading days per year, and a 2% annual risk-free rate.
- **Batch window:** the six calendar months before the month that contains the latest cleaned observation. The latest is 2024-12-06, so the window is **2024-06-01 to 2024-11-30**, anchored to the data rather than the computer clock. The output folder name is the run date, which is unrelated to this window.
- **Uneven coverage:** 8 tickers have data through 2024-11-29 (the last session in the window). The other 22 end on 2024-10-18, so their reports cover only 97 trading days instead of 126. Every report states its actual first and last dates, and nothing is extrapolated through the gap. The summary CSV includes `first_date`, `last_date` and `trading_days`; the script prints a coverage note; and the scatter plot marks the shorter histories with triangles. Compare those tickers with care, since four months of returns are not the same as six.

## How to interpret the reports

Each report (e.g. `reports/2026-10-30/IBM_report.ipynb`) shows:

- **Coverage line:** requested window versus actual first and last observations, and how many rows contain repaired values.
- **2×2 figure.**
  - *Price:* adjusted close colored by volatility regime (red above the period's mean rolling volatility, green at or below it, gray during the 20-day warm-up), with the 50-day moving average, which needs 50 prices before it appears.
  - *Daily returns:* green up-day and red down-day bars.
  - *Rolling volatility:* the 20-day series with its mean and shaded high/low regimes.
  - *Drawdown:* the fall from the running peak, with the maximum drawdown marked.
- **Summary table.**
  - *Total return:* first to last adjusted close.
  - *Annualized return:* `exp(sum of log returns × 252 / n) − 1`. Over a window shorter than a year this scales the result up, so read it next to total return.
  - *Annualized volatility:* over the whole period, which is not the mean rolling volatility used in the CSV.
  - *Sharpe ratio:* above about 1 means the excess return was large relative to volatility in this window. It is historical, not a forecast.
  - *Maximum drawdown:* the worst peak-to-trough fall.
  - *Best/worst day and average up/down day:* based on log returns.
  - *Positive days:* positive returns ÷ all returns, including zero-return days.
  - Undefined values show as `N/A`.

`portfolio_summary.csv` keeps fractions (0.05 = 5%) and is sorted by Sharpe ratio. Its columns:

- `avg_daily_return`: mean daily log return
- `avg_volatility`: mean of the 20-day rolling volatility
- `max_drawdown`
- `sharpe_ratio`
- `total_return`
- `report_path`
- coverage: `first_date`, `last_date`, `trading_days`

In this run, IBM ranks first (Sharpe 3.22, +39.9% total return with about 20% volatility) and MRK last (Sharpe −1.93, −19.9%). NVDA shows why volatility matters: a +20.0% return with 61% average volatility gives a Sharpe of only 0.78. Undefined Sharpe ratios would be left out of the top/bottom ranking. With 30 tickers, the top 10 and bottom 10 do not overlap.

Metrics are collected with Scrapbook from the `report_summary` scrap of each report **that succeeded in this run**. The script never globs the folder, so stale or failed notebooks cannot leak into the summary. A failed report's partial output is renamed `{ticker}_report_FAILED.ipynb`, and its ticker goes into `failed_stocks.txt`.
