# HW2: Scalable Stock Analysis Pipeline

A three-step pipeline: clean the dirty `ohlc_hw` table, generate a parameterized risk report for one stock,
and run that report for every stock with papermill.

| File | Purpose |
|---|---|
| `01_data_cleaning.ipynb` | Diagnose and repair data quality issues; writes `ohlc_hw_cleaned` and its audit tables to `data/data.db` |
| `02_stock_report_template.ipynb` | Parameterized single-stock risk report (`ticker`, `start_date`, `end_date`, plus `db_path`, `risk_free_rate`) |
| `03_run_all_stocks.py` | Runs the template for every ticker over a common 6-month window and builds a portfolio summary |
| `reports/2026-10-24/` | Sample output: IBM, NVDA and MRK reports and `portfolio_summary.csv` |
| `requirements.txt` | Pinned dependencies |

## How to run

Run everything from this folder (`Homeworks/HW2/Oscar_Sallusti/`), with Python 3.12, in this order:

```bash
# 1. Environment
uv venv --python 3.12 .venv
uv pip install --python .venv -r requirements.txt
source .venv/bin/activate

# 2. Data: a local working copy of the course database from bCourses (never committed)
mkdir -p data
cp "../../../Lectures/Lecture 4/data/data.db" data/data.db

# 3. Clean the data: recreates ohlc_hw_cleaned (and the audit tables) in data/data.db
jupyter nbconvert --to notebook --execute --inplace 01_data_cleaning.ipynb

# 4. Optional: one report interactively. Open 02_stock_report_template.ipynb, edit the parameters cell, run all.

# 5. Reports for every ticker
python 03_run_all_stocks.py                     # all tickers
python 03_run_all_stocks.py --tickers AAPL MSFT # a subset, useful for testing
```

The cleaning notebook only writes to the local copy `data/data.db`. The raw `ohlc_hw` table and the course's
files are never modified.

`03_run_all_stocks.py` writes to `reports/{run date}/`: one `{ticker}_report.ipynb` per stock,
`portfolio_summary.csv`, `risk_return.png`, `failed_stocks.txt` and a detailed `run.log`. As requested, only three
sample reports and the summary CSV are committed (this folder's `.gitignore` handles that).

Tables written to `data/data.db` by the cleaning notebook:

| Table | Contents |
|---|---|
| `ohlc_hw_cleaned` | Cleaned prices, ISO `YYYY-MM-DD` dates, repair flags (`flag_*`), `adj_source`, `return_outlier_class` |
| `ohlc_hw_quarantine` | Rows set aside because no defensible repair exists, with the reason (empty for this data) |
| `ohlc_hw_quality_report` | The Data Quality Report: the diagnosis before any repair |
| `ohlc_hw_ticker_audit` | Every original ticker label and the symbol it was normalized to |

## Data quality findings

The checks were run as specified, and several correctly found nothing. No faults were invented to fill the checklist.

| Issue | Found | Treatment |
|---|---|---|
| Tickers wrapped in HTML (`<a ...><span style="color: blue">AAPL</span></a>`); 30 stocks appear as 150 labels | 39,936 rows | Normalized (markup, whitespace, case). Almost all of these rows are dates that exist *only* under an HTML label, so deleting them would lose ~10% of the data |
| Exact duplicate records | 29,460 | Removed after normalizing |
| Conflicting `(ticker, ts)` keys | 0 | Would be quarantined (tested on injected data) |
| Missing `adj_close` | 18,374 | Rebuilt from a validated adjustment factor (see below) |
| Corrupted `adj_close`, 4–24× the raw close | 10,417 | Rebuilt from a validated adjustment factor |
| Missing raw prices or volume; rows with all prices missing | 0 | Handled by the pipeline (tested on injected data) |
| `high < close`, `low > close`, negative or non-finite prices | 0 | Handled by the pipeline (tested on injected data) |
| `open = 0` on a ticker's first day (1962–66) | 5 | Estimated from the same day's prices (Option 3) |
| One-day raw price spikes (TRV 1976/78/81: price doubles, then fully reverts) | 3 | Interpolated between valid neighbours |
| Missing trading days, against the `ohlc` reference calendar | 0 | Detector demonstrated on a tiny example with one removed date |
| Zero volume (mostly 1962–82 with O=H=L=C, i.e. stale quotes) | 75 | Kept, flagged |
| Rows dated on a Sunday (1974–75) | 384 | Kept, flagged |
| Returns beyond 3σ | 5,056 | Kept, flagged, classified (see outliers) |

**Row accounting:** 397,711 raw rows = 29,460 exact duplicates removed + 368,251 retained (0 quarantined).

## Key design decisions

- **Diagnose first, then repair, then re-diagnose.** All detectors write to one Data Quality Report (one observation
  can have several issues). The same `run_all_detectors` function is rerun on the cleaned table. Only the issues
  deliberately kept remain: extreme returns, zero volume and weekend dates.
- **Every per-ticker operation is grouped by ticker and sorted by date.** This applies to returns, forward-fill and
  interpolation, so no value is ever borrowed from another ticker.
- **Normalize first, then de-duplicate.** Conflicting keys would be quarantined (all rows of the key), not resolved
  by "keep first". A label audit table keeps the original labels.
- **Quarantine instead of guessing.** Rows that can't be repaired defensibly are set aside with a reason: a leading
  gap with no earlier price, conflicting duplicates, or an `adj_close` with no validated factor. None occurred
  in this data, but each case is tested.
- **`adj_close` keeps a consistent adjustment convention.** A missing or corrupted value is rebuilt as
  `validated raw close × factor`, where the factor comes from:
  1. the date-matched factor `adj_close / close` from the supplied `ohlc` table, accepted only if it agrees within 5%
     with the factor on the nearest good `ohlc_hw` day before or after (28,785 rows); otherwise
  2. the factor interpolated between the nearest good days, used only if those two agree within 5%, i.e. no split
     or large dividend in between (8 rows); otherwise
  3. quarantine (0 rows).

  The 8 neighbour-based repairs are days where the reference itself is inconsistent: the TRV spike days, and a few
  1974–75 DIS rows where the reference factor jumps for a day or two. `ohlc` is never used as a source of raw
  prices. Every raw price in `ohlc_hw_cleaned` comes from `ohlc_hw`. The `adj_source` column records which source
  each row used.
- **Every repair leaves a flag:** `flag_price_invalid`, `flag_price_filled`, `flag_volume_filled` (0 means
  *unknown*, not *no trading*), `flag_close_repaired`, `flag_open_repaired`, `flag_bounds_fixed`,
  `flag_adj_repaired`, plus `flag_zero_volume`, `flag_weekend`, `flag_return_outlier` and `return_outlier_class`.
- **Validation is explicit.** The checks assert:
  - no missing, non-finite or nonpositive raw or adjusted prices;
  - finite, nonnegative volume;
  - `low <= open, close <= high`;
  - unique `(ticker, ts)` keys and plain symbols;
  - no spikes left, and `adj_close` consistent with `close`.

  They run again on the table read back from the database, along with a check that every date is stored as ISO `YYYY-MM-DD`.
- **Tested on injected faults.** For each required case absent from the database, a small two-ticker example shows the
  detector firing and the repair (or quarantine) working. The repaired example passes the same validation.
- **Reports export their numbers.** The template sends its metrics back with `scrapbook`, and the batch script
  collects them **only from the reports that succeeded in this run** (never by scanning the folder). The CSV
  therefore always matches the reports.
- **Failures are expected, not fatal.** A ticker with too little data raises a clear error. The batch script logs it,
  keeps the partial notebook as `{ticker}_report_FAILED.ipynb`, lists the ticker in `failed_stocks.txt` and continues.
  If every ticker fails, it stops without pretending there is a summary.

## How outliers were handled, and why

**Impossible prices.** The only real cases were five zero opens, each on a ticker's first day. I used **Option 3**,
estimating from the same day's prices: the previous close clipped into `[low, high]`, or the high/low midpoint when
there is no previous day. The overnight gap is usually small, and clipping guarantees a valid row. For `high < close`
or `low > close` (tested on injected data), close is kept, because returns depend on it, and high/low are widened to
contain open and close. Nonpositive or non-finite high/low/close/`adj_close` are treated as missing and repaired the
same way. After every repair, the raw bounds and the `adj_close`/`close` consistency are rechecked.

**Extreme returns.** The screen flags `abs(return − mean) > 3 × std` within each ticker, using simple raw-close returns,
the sample std (`ddof=1`) and both tails. Undefined returns are excluded, and a zero or undefined std flags nothing.
It is a **review flag, not a deletion rule**: daily returns have fat tails, so most flags are real moves. Removing them
would make stocks look safer than they are. Each flag is classified:

- `market_move` (5,028): a large but plausible move. Kept.
- `adjustment_explained` (144): the raw return is extreme but the adjusted return is not. Kept.
- `large_move_review` (3): moves beyond ±40%, checked by hand. The claims are limited to what the data shows:
  - AAPL 2000-09-29, −52%: the drop persists and the adjusted series agrees, which is consistent with a genuine
    repricing. (A profit warning was reported around then, but that can't be verified from this data.) Kept.
  - NVDA 2000-03-07, +42%: the move persists. Kept.
  - MCD 1968-01-02, −75%: raw and adjusted prices both drop and stay down. The ratio of ~1/4 *could* be an
    unadjusted split, but nothing in the data confirms it, so it is kept and flagged rather than rescaled. It is
    decades outside the report window.

**Data errors** were separated from real moves by a specific pattern: a jump above 40% that fully reverses the next
day. Only the three TRV spikes match. They are interpolated between valid neighbours (never extrapolated at the edges
of a history). Black Monday 1987 (drops of 20–30% with partial rebounds) correctly does not match.

## Assumptions

- **Reference calendar:** each ticker's dates in the supplied `ohlc` table, over that ticker's own first-to-last
  history. Weekdays alone would wrongly count exchange holidays.
- **Annualization** uses 252 trading days. The Sharpe ratio uses a 2% annual risk-free rate and the course formula,
  `(mean daily log return × 252 − rf) / (std × √252)`. It is reported as N/A when volatility is zero or there are
  too few returns.
- **Report window:** the six calendar months before the month containing the latest cleaned observation. For a
  latest date of 2024-12-06 that is **2024-06-01 to 2024-11-30**. The latest month is always excluded, even if the
  data ends on its last day. Output folders are named by the **run date**, which is independent of the window.
- **Uneven coverage:** all tickers share one requested window, but 22 of the 30 have no data after **2024-10-18**.
  Their reports cover Jun 3 – Oct 18 (97 trading days) instead of Jun 3 – Nov 29 (126). Prices are never
  extrapolated through trailing gaps. Each report states its actual first and last dates, and
  `portfolio_summary.csv` includes `first_day`, `last_day` and `trading_days`. Comparisons between tickers with
  different coverage should be read with that in mind.
- **Warm-up periods are preserved, not filled.** Within each report, the 20-day volatility starts after 20 returns
  (drawn in gray before that). The 50-day moving average starts once 50 prices exist. Reports with fewer than 50
  prices say so instead of showing it.

## How to interpret the reports

Each report starts with the requested window and the actual first and last observations, then shows four charts and a
summary table. Returns and drawdowns use **adjusted** closes. Volatility, the Sharpe ratio and daily statistics use
daily **log** returns.

- **Price and 50-day MA:** the price line is green while 20-day volatility is at or below its average for the report,
  red when above, and gray during the warm-up. This colouring is retrospective, since the average uses the whole window.
- **Daily Returns:** green and red bars for up and down days. Clusters of tall bars show turbulent periods.
- **Rolling Volatility (20-day):** annualized, with the report's mean as a dashed line. Red shading marks above-mean
  regimes and green below. The gray band is the warm-up.
- **Maximum Drawdown:** how far the stock is below its highest price so far in the window. It recovers when the stock
  makes a new high. The marked point is the worst decline from a peak.

In the summary table:

- **Total return:** last / first adjusted close − 1.
- **Annualized return:** `exp(sum of log returns × 252 / n) − 1`.
- **Annualized volatility:** the whole-period std × √252. This is different from the *mean rolling* volatility in the
  portfolio summary.
- **Sharpe ratio:** above 1 is good; below 0 means the stock did worse than the risk-free rate.
- **Best/worst day** and **average up/down day:** daily log returns.
- **Positive days:** the share of returns above zero.

**Caution:** the window is short (4.5–6 months). Annualized returns and Sharpe ratios extrapolate a few months to a year,
which is why several stocks show a Sharpe near 3. Use them to compare stocks over the same period, not to forecast.

**Portfolio summary.** `portfolio_summary.csv` stores values as fractions (e.g. `0.2002` = 20.02%), sorted by Sharpe
ratio. The columns are:

- `avg_daily_return`: the mean daily log return.
- `avg_volatility`: the mean of the 20-day rolling volatility.
- `max_drawdown`, `sharpe_ratio`, and `total_return` (last / first adjusted close − 1).
- `report_path`, plus each report's coverage dates.

Rankings exclude undefined Sharpe ratios. In `risk_return.png`, points high and to the left (high return, low
volatility, dark green) had the best risk-adjusted performance over the window.
