# HW2: Scalable Stock Analysis Pipeline

Philip Szkudlarski

| File | Part | What it does |
|---|---|---|
| `01_data_cleaning.ipynb` | 1 | Inspects `ohlc_hw`, detects data-quality issues, cleans them, saves `ohlc_hw_cleaned` |
| `02_stock_report_template.ipynb` | 2 | Parameterized risk report for one ticker (papermill-ready) |
| `03_run_all_stocks.py` | 3 | Runs the template for every ticker, builds the portfolio summary |
| `reports/2026-10-25/` | 3 | Sample reports (IBM = best Sharpe, MRK = worst, NVDA = most volatile), `portfolio_summary.csv`, `risk_return.png` |

## How to run

All commands run from this folder (`Homeworks/HW2/Philip_Szkudlarski/`).

1. **Install dependencies** (Python 3.12):
   ```bash
   pip install -r requirements.txt
   ```
   In the course repo, the project's `.venv` already has everything.

2. **Add the database.** Download `data.db` from bCourses (or copy
   `Lectures/Lecture 4/data/data.db`) to `data/data.db`. The `data/` folder is
   gitignored, so the database is never committed.

3. **Clean the data.** Run all cells in `01_data_cleaning.ipynb`. This writes the
   `ohlc_hw_cleaned` and `ohlc_hw_quarantine` tables into `data/data.db`.

4. **Try a single report.** Open `02_stock_report_template.ipynb` and run it, or
   run it from the command line with other parameters:
   ```bash
   papermill 02_stock_report_template.ipynb out.ipynb -p ticker NVDA -p start_date 2024-06-01 -p end_date 2024-11-30
   ```

5. **Generate every report:**
   ```bash
   python 03_run_all_stocks.py
   ```
   Output goes to `reports/{today's date}/`. Failed tickers (none in my run)
   are listed in `failed_stocks.txt`, and details are in `reports/{date}/run.log`.

## What was wrong with the data

I found these with the checks in Part 1.2. The `ohlc` table (same prices, no
injected problems) is used for two things only: its dates serve as the reference
trading calendar, and its `adj_close / close` factors repair bad adjusted prices.
Otherwise it is used only afterwards, to measure how well the cleaning worked.

| Issue | Rows | Handling |
|---|---|---|
| Tickers wrapped in HTML (`<a ...><span ...>AAPL</span></a>`) | 39,936 (~10%) | Stripped with `clean_ticker`; the raw → clean mapping is kept in `label_audit` |
| Duplicate ticker/date rows | 29,460 | Every pair is an exact copy (including the raw label), so extras are removed. Conflicting pairs would be quarantined; none occur |
| Missing `adj_close` | 18,374 | Rebuilt from the `ohlc` factor (see below) |
| `adj_close` above `close`, or > 1.5× its 21-day median | 10,419 | Rebuilt from the `ohlc` factor (see below) |
| `open = 0` | 5 | Previous day's close, clipped into the day's [low, high] |
| One-day doubled bars (TRV 1976, 1978, 1981) | 3 | Rescaled to the neighboring days' level |
| Zero-volume days | 75 | Kept, flagged as possibly stale |
| Returns beyond 3 standard deviations | ~5,000 | Investigated, classified, kept and flagged |

No trading days were missing relative to the `ohlc` calendar. The only long
gap (> 5 calendar days) is the 2001-09-10 → 2001-09-17 market closure after 9/11,
which is real.

## Key design decisions

**Cleaning order.** Raw prices are repaired first, then `adj_close` is rebuilt
from the repaired `close`. This keeps one consistent adjustment convention, as
the assignment asks.

**Rebuilding `adj_close`.** In clean data, `adj_close / close` is an
"adjustment factor" ≤ 1 that only changes at dividends and splits (backward
adjustment can only lower historical prices). So a factor is treated as bad
when it is missing, above 1, or more than 1.5× its 21-day rolling median. A bad
factor is **not** interpolated or forward-filled, because a dividend or split
inside the gap would make the guess wrong. Instead it is taken from the same
ticker and date in the `ohlc` table, and `adj_close = close × factor` is
recomputed from our own repaired `close`. Only the factor comes from `ohlc`,
not the prices. After cleaning, `adj_close` differs from `ohlc` by more than 1%
on only 3 rows (the repaired TRV spikes, where `ohlc` still has the doubled bar).

**Quarantine and row accounting.** A row is removed only when it carries no
information (an exact duplicate, or all four prices missing). A row with a
problem that can't be fixed without guessing is **quarantined**: moved to the
`ohlc_hw_quarantine` table with a reason. That covers a missing ticker/date,
conflicting duplicates, a leading price gap (nothing earlier in the ticker to
fill from), and a bad `adj_close` with no reference factor. A row-accounting
table shows every one of the 397,711 raw rows as retained (368,251), removed
(29,460 exact duplicates) or quarantined (0).

**Tests on injected examples.** Several required checks find nothing in this
database (missing trading days, close outside high/low, missing volume,
all-missing rows, leading gaps, conflicting duplicates). The notebook runs
each detector and repair on a tiny made-up ticker with those problems injected,
with `assert`s confirming they are caught.

**Impossible price relationships (documented choice).** A zero open is a
recording error, so it gets the previous day's close (Option 2). For any other
open/close outside the day's range, the code trusts the day's own prices and
widens high/low to contain them (Option 3). After cleaning, every row
satisfies `low ≤ open, close ≤ high`.

**Repair flags.** Every cleaning step sets a `flag_*` column in
`ohlc_hw_cleaned`, and `return_flag` labels each extreme return, so nothing is
silently changed.

**Earlier history in the report.** The template loads the ticker's full history
up to `end_date`, so the 50-day moving average and 20-day volatility exist from
the first day of the chart instead of leaving it blank. All reported metrics
(returns, Sharpe, drawdown, summary table) use only the requested window. The
header line shows the actual first and last trading days in that window, which
can differ from the requested dates.

**Passing results to Part 3.** The template takes an optional `metrics_path`
parameter. When set, it writes its summary numbers to JSON, which the batch
script collects. The formulas live in one place (the template) instead of
being duplicated in the script. Only reports that ran successfully contribute
to the summary. Tickers with an undefined Sharpe ratio stay in the CSV but are
left out of the top/bottom 10 rankings.

## How outliers were handled, and why

About 5,000 daily returns lie beyond 3 standard deviations. Over 60 years of
fat-tailed stock returns that's expected, so deleting them all would remove
real history (1987, 2008, March 2020). Each one was investigated instead:

1. **Data error → repaired.** A move of more than ~35% that fully reverses the
   next day (back within 5% of the prior close). Only 3 rows qualify: TRV bars
   where open, high, low and close are all exactly doubled for one day. Real
   crashes and rebounds don't land back exactly where they started.
2. **Split / corporate action → kept.** The raw close jumps but the adjusted
   price doesn't, because `adj_close` already absorbs the split (201 rows).
3. **Market move → kept and flagged.** The adjusted price moves too, so it's
   real news (4,997 rows).

A looser first version of rule 1 (">3σ followed by a reversal") flagged 351
days, but almost all were genuine crash-and-rebound days such as March 12–13,
2020. Requiring a large, exact snap-back fixed that.

## Assumptions

- **`adj_close` is the price for all metrics.** It accounts for splits and
  dividends. The raw `close` would show splits as crashes.
- **252 trading days per year** to annualize; **2% risk-free rate** for Sharpe.
- **Log returns** of `adj_close` for every return statistic in the report:
  volatility, Sharpe, annualized return, best/worst day, average up/down day,
  positive days, the daily-returns chart, and `avg_daily_return` in the
  portfolio summary (as the assignment specifies).
- **"Last 6 complete months"** is anchored to the latest date in the database
  (2024-12-06), giving 2024-06-01 to 2024-11-30: the six calendar months before
  the month containing the latest observation. That month is excluded even if
  the latest observation falls on its last day (e.g. 2024-11-30 would give
  2024-05-01 to 2024-10-31), since a final trading date alone doesn't prove the
  month's data is complete.
- **The report folder is named by run date** (`reports/{YYYY-MM-DD}/`); the
  analysis window is printed in each report and in the chart titles.

**Important caveat for comparing stocks:** 22 of the 30 tickers have data only
up to 2024-10-18, while 8 (AMZN, CVX, DIS, IBM, KO, MRK, NKE, PG) run to
2024-12-06. So in the portfolio summary, those 22 cover 97 trading days and the
8 cover the full 126. The `first_date`, `last_date` and `trading_days` columns
in `portfolio_summary.csv` show each ticker's actual coverage. All tickers use
the same requested window, and no prices are extrapolated past a ticker's last
date. Total returns are not directly comparable across the two groups.

## How to read a report

Each report has a header line (`AAPL from … to …: N trading days`, using the
actual first and last trading days), the first and last rows, a metrics table,
a 2×2 chart and a summary table. A window with fewer than 21 prices stops with a
clear "Insufficient history" error, and if a ticker's history starts inside the
window, the report notes how many days the moving average and volatility are
unavailable (shown as gaps, never filled).

- **Price and 50-day MA (top left).** The line color compares each day's 20-day
  volatility with its mean over the report: red = above the mean, green = at or
  below it, gray = not available yet. Price above the dashed MA means an uptrend.
- **Daily returns (top right).** Green up days, red down days. Clusters of tall
  bars are volatile periods.
- **Rolling volatility (bottom left).** Red shading = riskier than the stock's
  average for the period; green = calmer.
- **Drawdown (bottom right).** How far the price is below its previous peak.
  The marked point is the worst loss from a peak during the period.
- **Summary table.** Sharpe ratio is return per unit of risk above the
  risk-free rate: above 1 is good, below 0 means the stock did worse than cash.

**Portfolio view (`risk_return.png`).** Each dot is a stock. Further right =
more volatile, higher = better return, greener = higher Sharpe. The best
risk-adjusted stocks sit in the upper left. In this window, IBM, HD, SHW, WMT
and MCD had the highest Sharpe ratios, while MRK, BA and NKE lost money.
