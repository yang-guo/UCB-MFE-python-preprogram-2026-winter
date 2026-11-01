# HW2: Stock data cleaning and risk reports

Gobind Singh. Three pieces, run in this order from this directory:

1. `01_data_cleaning.ipynb` rebuilds `ohlc_hw_cleaned` from the dirty `ohlc_hw` table.
2. `02_stock_report_template.ipynb` is one stock's risk report. Papermill injects the ticker and dates into the cell tagged `parameters`.
3. `03_run_all_stocks.py` runs that template for every ticker and writes the portfolio summary.

## How to run

The course repo already has a lockfile. From the repository root:

```bash
uv sync --locked
source .venv/bin/activate
```

If you are not using that environment, install this folder's `requirements.txt` into a Python 3.12 environment instead. Notebooks have to run with that interpreter; a system Jupyter will not see `scrapbook` or `papermill`.

Download `data.db` from bCourses and copy it to `Homeworks/HW2/GOBIND_SINGH/data/data.db`. It is gitignored. The notebook only writes new tables into that copy and leaves `ohlc_hw` and `ohlc` untouched.

Then, from this directory:

```bash
# 1. Cleaning. Run every cell, top to bottom.
# 2. Optional single report. Edit the parameters cell, or:
python -m papermill 02_stock_report_template.ipynb /tmp/AAPL_check.ipynb \
    -p ticker AAPL -p start_date 2024-01-01 -p end_date 2024-12-31 \
    -p db_path data/data.db

# 3. Full batch. Prints a tqdm bar, one line per ticker.
python 03_run_all_stocks.py
```

The batch writes `reports/<today>/`. Each run deletes the previous run's notebooks, CSV, and chart in that folder before it starts, and appends to `run.log`.

## Design decisions

**Calendar.** Missing dates are judged against the clean `ohlc` table, limited to each ticker's own first and last date in `ohlc_hw`. I do not invent a weekday calendar. A side check found 384 rows that fall on a Sunday in 1974–75 (BA alone has 24). Those dates are in the reference table too, so I kept them rather than dropping anything the reference calls a session.

**Repairs stay reversible.** Every change is recorded in `ohlc_hw_quality_report`, and rows that cannot be repaired go to `ohlc_hw_quarantine` instead of being guessed at. The cleaned table keeps one row per `(ticker, ts)`.

**Adjusted close.** A bad `adj_close` is not a return outlier; the raw OHLC on those rows is fine and the adjustment factor is several times too large. I measure the factor `adj_close / close` and compare it with a centered 21-day rolling median of that factor. A factor more than twice the local median is treated as corrupted. This is my own detector. Lecture 4 is what suggested looking at the adjustment factor instead of the price level, and I checked the repairs against the factor in the clean `ohlc` table: after repair, adjusted closes match that reference. The replacement value is the reference factor times the cleaned close. Factors are never carried across dates.

**Returns in the report.** Daily log returns, 21-day rolling volatility, drawdown from the running peak, and an annualized Sharpe ratio with 252 trading days and a zero risk-free rate. The return bar chart draws only days that have a return. The first price has none, so it is left off the chart rather than shown as 0%.

**Batch window.** The script takes the latest date in the cleaned table (2024-12-06) and requests the six complete calendar months before it, 2024-06-01 through 2024-11-30. It does not extend a ticker past its last real row.

## Outliers

| What | Count | What I did |
|---|---|---|
| Exact duplicate rows | 29,460 | Dropped the copies |
| Missing `adj_close` | 18,374 | Rebuilt from the reference factor when the close was usable |
| `adj_close` several times too large | 10,417 | Repaired with the rolling-median detector above |
| First row with `open = 0` | 5 | Quarantined; there is no previous close to repair from |
| Daily return beyond 3 standard deviations | 5,057 | Flagged in the quality report and kept. A large move is not evidence the print is wrong |
| Zero volume | 74 | Flagged and kept |

The notebook also runs the same functions on a small made-up frame with the bugs injected, so a grader can see the rules fire without reading the whole table. On the real data the cleaned table has 368,246 rows.

## Assumptions

- `ohlc` is a trustworthy reference for dates and for the adjustment factor. It is not a substitute for the cleaning: `ohlc_hw` still has bad prices, missing adjusted closes, and duplicates that have to be handled explicitly.
- A Sunday row that appears in both tables is a real session for this dataset.
- Sharpe uses a risk-free rate of zero, because the assignment does not give one.
- Eight tickers have prices through 2024-11-29. The other twenty-two stop on 2024-10-18. That gap is in the source file. Their metrics cover fewer days, and the script prints the list so the rankings are not read as like-for-like.

## How to read the output

Each report notebook has the price path with the drawdown shaded, a rolling-volatility panel, and one bar per defined daily return (green up, red down). The glued `report_summary` is what the batch collects: total return, annualized return and volatility, average daily return, maximum drawdown, and Sharpe ratio, plus the actual first date, last date, and number of trading days.

`reports/<date>/portfolio_summary.csv` has one row per ticker that finished. `portfolio_risk_return.png` plots annualized volatility against annualized return. `failed_stocks.txt` is in the same folder; a successful run leaves it empty, and a failed ticker is written there as `<ticker><tab><message>` with its notebook renamed `*_report_FAILED.ipynb`.

The CSV keeps a `report_path` for all 30 tickers. This submission commits three sample notebooks, AAPL, IBM, and MRK, so the other 27 paths point at files that were generated locally and then removed.
