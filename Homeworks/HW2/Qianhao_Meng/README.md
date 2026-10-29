# HW2: Scalable Stock Analysis Pipeline (Qianhao Meng)

`ohlc_hw` -> validated `ohlc_hw_cleaned` -> parameterized single-stock report -> reports for every ticker plus a portfolio summary.

| File | Purpose |
|---|---|
| `01_data_cleaning.ipynb` | Part 1: inspect `ohlc_hw`, build the Data Quality Report, clean, validate, save `ohlc_hw_cleaned` |
| `02_stock_report_template.ipynb` | Part 2: papermill template (tagged `parameters` cell) for one ticker and date range |
| `03_run_all_stocks.py` | Part 3: runs the template for every ticker, collects the results, ranks and plots them |
| `requirements.txt` | Packages used by the three files |
| `reports/2026-10-28/` | Sample output: `AAPL`, `JPM` and `PG` reports and `portfolio_summary.csv` |

## How to run

1. **Environment.** Use the course environment (`uv sync --locked` at the repository root, kernel `.venv`), or Python 3.12 with `pip install -r requirements.txt`.
2. **Database.** Copy the bCourses `data.db` to `data/data.db` inside this folder. It is a local working copy and is not committed (`data.db` is git-ignored).
3. **Clean.** Open `01_data_cleaning.ipynb` with this folder as the working directory and run all cells. It reads `ohlc_hw` read-only and writes `ohlc_hw_cleaned` into `data/data.db`. The raw table is not changed.
4. **One report (optional).** Run `02_stock_report_template.ipynb` directly. Its defaults are AAPL for 2024.
5. **All reports.** From this folder run `uv run python 03_run_all_stocks.py` (or `python 03_run_all_stocks.py` in the activated environment, so the papermill kernel uses the same packages). It writes to `reports/<run date>/`: one `{ticker}_report.ipynb` per ticker, `failed_stocks.txt`, `portfolio_summary.csv` and `risk_return_profile.png`.

The full batch (30 reports) was generated locally, and only three sample reports and the summary CSV are committed. The CSV's `report_path` column lists every report of the run.

## Key design decisions

- **Ticker labels first.** HTML markup, whitespace and case are stripped from the 150 raw labels (30 tickers) before grouping or deduplicating. `raw_ticker` keeps the original label.
- **Duplicates.** 29,460 exact duplicate records are removed. Rows that share `(ticker, ts)` but disagree would be quarantined rather than resolved by keeping the first. This data has none.
- **Missing values.** Rows with all four raw prices missing are dropped before any filling. A missing raw price is forward-filled from the ticker's last valid price. Unknown volume becomes 0 with `volume_repaired`, because unknown volume is not evidence of no trading.
- **Adjusted close.** `ohlc` gives a trusted factor `adj_close / close` for the same `(ticker, ts)`. `adj_close` is rebuilt as validated `close` x that factor when it is missing (18,374 rows), inconsistent with the factor by more than 0.1% (10,417 rows, inflated 4-24x), or its `close` was repaired. Factors are never forward-filled, so no factor crosses an unverified corporate action. Rows with no matching `ohlc` row cannot be rebuilt and are quarantined.
- **Severity.**
  - `high` means a raw or adjusted price is missing, invalid or impossible, so returns cannot be trusted until it is repaired or quarantined.
  - `medium` means a record-level issue that is fixed deterministically (formatting, duplicates, volume) or only reported (absent rows), or a review flag whose value may be genuine (zero volume, extreme return).
- **Audit trail.** Every repair sets a `*_repaired` flag, extreme returns carry `extreme_return` and `return_evidence`, `return_affected` marks returns changed by a repaired close, and all flags are saved with the cleaned table. Rows still invalid after repair are quarantined with a reason. Here 397,711 raw rows = 368,251 retained + 0 quarantined + 29,460 removed.
- **Report inputs.** The report reads `ohlc_hw_cleaned` read-only, re-checks the rows it loads, and stops with a clear error when there are fewer than 21 prices (20 returns). The batch records that as a failure and continues.
- **Batch results.** Each report saves its metrics with scrapbook. The script collects them only from the reports that succeeded in this run, never by scanning the folder, and keeps fractions in the CSV. Each run first deletes the previous `failed_stocks.txt`, summary CSV and chart in that day's folder, so a same-day rerun in which every report fails leaves no earlier summary behind.

## How outliers were handled and why

- **Impossible price relationships** (`open`/`close` outside `[low, high]`, nonpositive prices): Option 1, falling back to Option 3. The value is set to NaN and forward-filled when the ticker's last valid value lies inside today's range. Otherwise it becomes today's `(high + low) / 2`. This keeps continuity where possible and always satisfies the bounds. In this data the only cases are 5 rows with `open = 0`, each on a ticker's first day, so they take the midpoint.
- **Extreme returns** (`|return - mean| > 3` sample std of close-to-close returns within a ticker) are review flags, not deletions. A flagged close is treated as a data error only when it differs from `ohlc` on that day. Only those closes are interpolated between valid neighbours, never at the edges of a ticker's history. All other flags are kept with their evidence: a change of more than 5% in the `ohlc` adjustment factor marks a possible split or dividend, and a close that matches `ohlc` is a real move. Of 5,056 flags, 13 coincide with a factor change and 5,043 match `ohlc`, so nothing in this data was interpolated. The notebook shows the interpolation on an injected error.
- **Zero volume** (75 days) is flagged and kept, since it can be a stale record or a genuine no-trade day.

Cases the data does not contain (missing raw prices, `high < close`, `low > close`, removed first, middle and last trading days, conflicting keys, leading gaps, non-finite prices, missing and negative volume) are demonstrated on small injected examples in sections 1.2.2 and 1.3.2.

## Assumptions

- `ohlc` is the trusted reference: each ticker's dates over its full `ohlc` history are its trading calendar (so a missing first or last day is caught too), its factors define the adjustment convention, and its closes confirm data errors. After cleaning, the two tables cover exactly the same `(ticker, ts)` keys, so no trading day is missing.
- The 0.1% factor tolerance and 5% factor-change threshold are judgement calls. In this data every row either matches `ohlc` exactly or is off by at least 300%, so any tolerance in between flags the same rows.
- Reports use adjusted close, daily log returns, 252 trading days a year and a 2% annual risk-free rate. The Sharpe ratio is undefined (N/A) with fewer than two returns or zero volatility.
- The batch window is the six calendar months before the month of the latest cleaned observation (2024-12-06 -> 2024-06-01 to 2024-11-30). The folder name is the run date, not the data window.
- Histories are not extrapolated. 22 of the 30 tickers end on 2024-10-18 (97 prices in the window), and the other 8 run to 2024-11-29 (126 prices).

## How to interpret the reports

- **Header.** The first line is the requested window. The next line and the figure footer give the actual first and last observations. When they differ, every metric covers only the observed dates.
- **Price and 50-day MA.** The adjusted price is red when 20-day volatility is above its mean for this report, green otherwise, gray during the 20-return warm-up. The MA starts at the 50th price.
- **Daily Returns.** Daily log returns in percent, green up and red down.
- **Rolling Volatility.** Annualized 20-day volatility against its mean. Red shading marks high-volatility stretches, and green marks low ones.
- **Maximum Drawdown.** The decline from the running peak on each date. The marked point is the worst value, which is the summary's Maximum Drawdown.
- **Summary table.** Total return runs from the first to the last adjusted close. Annualized return and volatility come from the log returns. Best/Worst Day and the average up/down days are log returns. Positive Days counts zero-return days in the denominator.
- `portfolio_summary.csv`**.** One row per successful report, sorted by Sharpe ratio, with fractions (0.05 = 5%): `avg_daily_return` (mean daily log return), `avg_volatility` (mean of the 20-day rolling volatility), `max_drawdown`, `sharpe_ratio`, `total_return` and `report_path`. Undefined Sharpe values are left out of the top/bottom 10.
- **Risk-Return Profile.** Mean rolling volatility against total return, colored by Sharpe ratio, with the top five returns labeled. Triangles mark tickers whose data ends before 2024-11-29, so their returns cover about four and a half months instead of six. These are descriptive statistics for one historical window, not forecasts.
