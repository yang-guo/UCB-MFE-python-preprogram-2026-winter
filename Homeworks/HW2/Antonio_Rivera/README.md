# HW2 — Data cleaning, single-stock risk reports, and batch reporting

Antonio Rivera

| File | Purpose |
|---|---|
| `01_data_cleaning.ipynb` | Diagnoses and repairs `ohlc_hw`; writes `ohlc_hw_cleaned`, `ohlc_hw_quarantine`, `ohlc_hw_quality_report` |
| `02_stock_report_template.ipynb` | Parameterized (papermill) risk report for one ticker and date range |
| `03_run_all_stocks.py` | Runs the template for every ticker and builds the portfolio summary |
| `requirements.txt` | Python dependencies (Python 3.12) |
| `reports/2026-10-28/` | Sample output: 3 reports (IBM, MRK, AAPL) and `portfolio_summary.csv` |

## How to run

All paths are relative to this folder (`Homeworks/HW2/Antonio_Rivera/`).

1. **Install dependencies** (Python 3.12):
   ```bash
   uv venv --python 3.12 && source .venv/bin/activate
   uv pip install -r requirements.txt
   ```
   (`python -m venv .venv` and `pip install -r requirements.txt` work too.)
2. **Add the database.** Copy `data.db` from bCourses to `data/data.db`. It is not committed.
3. **Clean the data.** Open `01_data_cleaning.ipynb` and run all cells. This recreates `ohlc_hw_cleaned` in `data/data.db`. The raw `ohlc_hw` table is only read, never modified.
4. **(Optional) One report.** Open `02_stock_report_template.ipynb`, edit the `parameters` cell (`ticker`, `start_date`, `end_date`), and run all cells.
5. **All reports.**
   ```bash
   python 03_run_all_stocks.py
   ```
   This writes, to `reports/<today's date>/`:
   - `{TICKER}_report.ipynb` — one executed report per ticker
   - `portfolio_summary.csv` — one row per successful report, sorted by Sharpe ratio
   - `risk_return.png` — the risk-return scatter
   - `failed_stocks.txt` — failed tickers and the reason (empty if none)
   - `run.log` — the run's log

   It also prints the top 10 and bottom 10 tickers by Sharpe ratio.

Run order matters: step 3 must finish before steps 4–5, because the reports read `ohlc_hw_cleaned`. Only the 3 sample reports and the CSV are committed (see `.gitignore`); the full batch is regenerated locally.

## Key design decisions

**Cleaning (Part 1)**
- **Tickers are normalized first.** The 150 raw labels were HTML-wrapped variants of 30 tickers. Tags are stripped and labels trimmed and upper-cased *before* any grouping or de-duplication, otherwise one stock would be split across labels. The original label is kept in `raw_ticker`.
- **The supplied `ohlc` table is the trusted reference.** Raw open/high/low/close/volume in `ohlc_hw` matched it on every row; the damage was in `adj_close` (18,374 missing, 10,417 inconsistent). Adjusted close is rebuilt as `factor × close`, using the reference's adjustment factor for that date (`factor = adj_close / close`), rather than replacing the dataset with the reference.
- **Repairs are recorded, not hidden.** Each row keeps `price_repair` / `adj_close_repair` text and boolean flags. Rows that cannot be repaired honestly go to `ohlc_hw_quarantine` with a reason instead of borrowing another ticker's value.
- **Validation is enforced with assertions** (no missing, non-finite or nonpositive prices; unique `(ticker, ts)`; open/close inside [low, high]; ISO dates; every raw row accounted for: 397,711 = 29,460 duplicates + 5 quarantined + 368,246 retained).

**Reports (Parts 2–3)**
- **Adjusted close for all report returns and drawdown;** daily **log** returns for volatility, Sharpe and the daily statistics. Values are stored as fractions; percentages are formatted only for display.
- **Warm-up gaps are kept.** Rolling volatility is NaN for the first 20 days and the 50-day MA for the first 50 prices; they are never filled with 0.
- **Undefined is shown as N/A.** Sharpe is NaN with fewer than 2 returns or zero volatility, and an empty up/down-day subset shows N/A.
- **Scrapbook passes results to the batch script.** The template's last cell glues the summary metrics and coverage dates. The script reads only the reports that succeeded in the current run, never every notebook in the folder, so failed or stale reports cannot leak into the summary. Values read back are converted to numbers (scrapbook returns NaN as the string `'nan'`).
- **Errors never stop the batch.** Each ticker is wrapped in `try/except`; failures go to `run.log` and `failed_stocks.txt`. If every ticker fails, the script prints that there is no summary, removes any old summary files from that folder, and exits with code 1.

## How outliers were handled, and why

An extreme return was defined as a daily simple return more than 3 standard deviations from that ticker's own mean.

- **Flagged and kept, not deleted (5,176 rows).** In quantitative finance, large moves are often real (earnings, crashes, splits) and deleting them would understate risk. Every flagged close was checked against the reference table and all 5,056 pre-repair flags matched it, so they are moves in the source data, not typos. Each kept flag has a `return_note` in `ohlc_hw_cleaned`:
  - 2 coincide with a step in the reference adjustment factor (a split or special dividend): a large raw move but a continuous adjusted series.
  - The rest are marked "cause not verified", because no news source was available to confirm them.
- **Repaired only when confirmed as errors (3 rows).** Three TRV days had the whole row doubled, zero volume, and a full reversal the next day. A real move does not do all three, so these were interpolated from the neighbouring days.
- **Why the flag count rose (5,056 → 5,176).** Repairing the spikes lowered each ticker's standard deviation, so a few more returns crossed 3σ. This is expected and is documented in the notebook.
- **Reference factor blips.** On 52 dates the reference adjustment factor jumps for one day and returns. These are treated as errors in the reference, and the neighbouring day's factor is used.

## Assumptions

- **The supplied `ohlc` table is correct** except for one-day factor blips, and is used for trading dates and adjustment factors. Raw prices in `ohlc_hw` are kept; only `adj_close` is rebuilt.
- **Missing trading sessions** are dates present in `ohlc` for that ticker (within its first-to-last date in `ohlc_hw`) but absent from `ohlc_hw`. None were found; the detector is tested on an injected gap.
- **Severity:** *high* = the value cannot be true; *medium* = may be true but needs review or blocks a metric; *low* = plausible, price unaffected.
- **Known limitation:** the reference factor for DIS in Jan–Feb 1974 alternates between about 0.621 and 0.663, so those adjusted prices are approximate (±7%).
- **Report window.** All tickers share one requested window: the 6 complete calendar months before the month of the latest cleaned observation. The latest is 2024-12-06, so the window is 2024-06-01 to 2024-11-30. It is anchored to the data, not the computer clock.
- **Shorter histories are not extended.** Supplied histories end on different dates: 8 tickers have data to 2024-11-29 (126 days), while 22 stop at 2024-10-18 (97 days). Prices are not extrapolated, each report shows its actual coverage, and the CSV includes `first_obs`, `last_obs` and `n_days`. Metrics over 97 days and over 126 days are not strictly comparable: the shorter set misses the Oct 18 – Nov 29 period.
- **Annualization** uses 252 trading days. The Sharpe ratio uses a 2% annual risk-free rate and the course approximation `(mean × 252 − rf) / (std × √252)`.

## How to interpret the reports

**Header.** The requested window, then the actual first and last observation dates and the number of trading days. Check these first: they say what period the numbers cover.

**Charts (2×2)**
- **Price and 50-day MA (top left):** adjusted close. The line is red when that day's 20-day volatility is above the report's average, green when at or below it, and gray during the first 20 days, when there is no estimate yet. The dashed line is the 50-day moving average; price above it suggests an uptrend. The coloring uses the whole period's mean, so it is a look back, not a real-time signal.
- **Daily returns (top right):** one bar per day, green up and red down. Clusters of tall bars show turbulent periods.
- **Rolling volatility (bottom left):** annualized 20-day volatility against its mean (dashed). Red shading marks high-volatility stretches, green low.
- **Drawdown (bottom right):** how far the price is below its highest level so far. 0% means a new high. The marked point is the maximum drawdown, the worst peak-to-trough loss.

**Summary table**

| Metric | Meaning |
|---|---|
| Total Return | last adjusted close ÷ first − 1 |
| Annualized Return | `exp(sum of log returns × 252 / n) − 1`. For a window shorter than a year this extrapolates the pace to 12 months, so it can exceed the total return. |
| Annualized Volatility | standard deviation of all daily log returns × √252 (different from the average of the rolling chart) |
| Sharpe Ratio | return above the risk-free rate per unit of volatility. Above 1 is generally good; negative means the stock did worse than the risk-free rate. |
| Maximum Drawdown | worst fall from a peak during the window |
| Best / Worst Day | largest and smallest daily log return |
| Positive Days | share of days with a positive return (zero-return days count in the total) |
| Average Up / Down Day | mean log return on up days and on down days |

**Portfolio summary (`portfolio_summary.csv`).** One row per ticker, sorted by Sharpe ratio. `avg_daily_return` is the mean daily log return, `avg_volatility` the mean of the 20-day rolling volatility, and `max_drawdown`, `sharpe_ratio` and `total_return` match the reports. All values are fractions (0.25 = 25%). Use `last_obs` and `n_days` to see which tickers have shorter coverage before comparing them.

**Sample reports included:** IBM (highest Sharpe, 3.22), MRK (lowest Sharpe, −1.93, with a negative return) and AAPL (data ends 2024-10-18, showing how a shorter history is reported).
