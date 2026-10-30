# HW2: Scalable Stock Analysis Pipeline (Romir Kant)

`ohlc_hw` → validated `ohlc_hw_cleaned` → parameterized single-stock report → all-stock batch + portfolio summary.

## How to run

All commands run from this folder (`Homeworks/HW2/Romir_Kant/`), using the course environment
(`uv sync --locked` at the repository root) or `pip install -r requirements.txt`.

1. Download `data.db` from bCourses into `Lectures/Lecture 4/data/data.db`.
2. **Clean:** run `01_data_cleaning.ipynb` top to bottom. On first run it copies the course database
   to `data/data.db` (a gitignored local working copy) and writes `ohlc_hw_cleaned` there.
   The source database and the raw `ohlc_hw` table are never modified.
3. **One report:** run `02_stock_report_template.ipynb`, or with papermill:
   `papermill 02_stock_report_template.ipynb out.ipynb -p ticker MSFT -p start_date 2024-06-01 -p end_date 2024-11-30`
4. **All reports:** `python 03_run_all_stocks.py` (about 2 minutes for 30 tickers). Optional:
   `--tickers AAPL MSFT` to run a subset, `--db-path` / `--output-root` to change locations.

## Part 1: data cleaning

### Output

`ohlc_hw_cleaned` in `data/data.db`: one validated row per `(ticker, ts)`, with `ts` stored as ISO
`YYYY-MM-DD`. Alongside the prices it keeps the original label (`raw_ticker`), the raw record's
`source_row`, a `<col>_repaired` flag for each price/volume column, and `extreme_return_flag` /
`extreme_return_class`. The Data Quality Report (diagnosed **before** any repair) and the lists of removed
and quarantined rows with their reasons are shown in the notebook.

Every raw row is accounted for: `retained + removed + quarantined = rows in ohlc_hw` (asserted).

### Key design decisions

* **Keys first.** Ticker labels are normalized (HTML tags stripped, whitespace trimmed, upper-cased)
  before any grouping, deduplication or return calculation; the original label is kept in `raw_ticker`.
* **Duplicates.** Exact duplicates (after normalization) are dropped. A conflicting `(ticker, ts)` key
  (same date, different values) has no defensible "first" record, so its rows are displayed and all of them
  quarantined rather than arbitrarily keeping one.
* **Reference table use.** The supplied `ohlc` table is used only as (1) each ticker's trading calendar for
  gap detection (so exchange holidays are not gaps) and (2) evidence: trusted date-specific adjustment
  factors (`adj_close / close`) and a check of whether an extreme price is a vendor error. It never
  replaces the dirty data wholesale. It is **not independent evidence**, though: `ohlc_hw`'s raw prices are
  an exact copy of it, so an error present in both looks "confirmed". See `reference_error` below.
* **Diagnose before repairing.** The quality report is built on the deduplicated data before any value
  changes, then the same checks are rerun on the cleaned data.

### Severity

* **high** - impossible or unusable value that must be repaired or quarantined (missing/nonpositive/
  non-finite raw price, OHLC bound violation, invalid volume or adjusted close, an adjusted close whose
  factor disagrees with the reference, conflicting duplicate).
* **medium** - plausible but needs a documented decision (missing `adj_close`, missing volume, missing
  trading session, 3σ extreme return).
* **low** - harmless once handled or informational (exact duplicate, zero-volume day).

### Repairs

* Rows with **all four raw prices missing** are dropped before filling.
* A **missing or invalid raw price** is forward-filled within the ticker (Option 1). If the carried value
  would fall outside *today's* high/low, today's (high + low)/2 is used instead (Option 3). Missing
  high/low are carried forward and then widened just enough to contain today's open/close.
* **Impossible OHLC relationships:** an open/close outside a consistent [low, high] is treated as the bad
  field and repaired as above. `high < low` cannot identify the bad bound, so those rows are quarantined.
* **Missing volume** becomes 0 with `volume_repaired = 1`: unknown volume is not evidence of no trading,
  and the flag keeps placeholders distinguishable from observed zeros.
* **Adjusted close** is rebuilt as `validated close x reference factor` for the same `(ticker, ts)` wherever
  it is missing/invalid, implies a factor (`adj_close / close`) more than 0.1% away from the reference
  factor, or the raw close was repaired. Factors are never forward-filled (that could carry a factor across
  an unseen split). Rows with no trusted factor are quarantined. After cleaning, every retained `adj_close`
  is checked against the reference factor.
* **Leading gaps** (nothing to carry forward) are quarantined, not extrapolated. Removed and quarantined
  dates remain absent: we never invent rows for missing sessions.
* **Rechecks:** raw OHLC bounds are asserted after each repair step (forward-fill, data-error
  interpolation, reference-error interpolation) and `adj_close` consistency after it is rebuilt, not only
  at the end.

### Outliers (3σ extreme returns)

Simple raw-close returns are screened per ticker with `|r - mean| > 3 * std` (sample std, both tails).
This is a review flag, not a deletion rule. Before anything is interpolated, each flag is classified with
evidence (saved as `extreme_return_class`):

| class | evidence | action |
|---|---|---|
| `data_error` | raw close differs from the reference by more than 1% | whole OHLC row interpolated between valid neighbours (interior only) |
| `corporate_action` | reference adjustment factor changes that day (split/dividend) | keep; the raw jump is real and adjusted prices are continuous |
| `confirmed_by_reference` | raw close matches the reference | keep and flag; we do not claim a news cause we cannot see |
| `unverified` | no reference row | keep and flag |
| `reference_error` | one-day round trip in the **adjusted** series: \|log return\| > 25% that the next day reverses to within 5% (the reference has the same error) | interpolate the raw row between neighbours; rebuild `adj_close` with the neighbouring days' factor if they agree within 1% (no split), else quarantine |

A flagged row whose close was already repaired by the bounds/forward-fill step is labelled `repaired_value`.
After all repairs the screen is rerun on the repaired series and saved as `extreme_return_flag`.

The `reference_error` rule exists because "matches the reference" is weak evidence here. TRV has three
days (1976-10-28, 1978-12-20, 1981-08-10) where the reference itself doubles the raw price and jumps the
adjustment factor to about 0.348, then both snap back the next day. No real price path does that, so the
earlier classification of these days as `corporate_action` was wrong. Large moves that do *not* reverse
(e.g. Black Monday, 1987-10-19) are left alone; MCD's −76% on 1968-01-02 remains flagged as unverified.

### Results on the course database

| finding | rows | handling |
|---|---|---|
| Raw rows in `ohlc_hw` | 397,711 | |
| Raw ticker labels (HTML-wrapped, mixed case/whitespace) | 150 labels → 30 tickers | normalized |
| Exact duplicates after normalization | 29,460 | removed |
| Conflicting `(ticker, ts)` keys | 0 | detector correctly finds none |
| Missing trading days vs. each ticker's reference calendar | 0 | detector correctly finds none (demonstrated on an injected gap) |
| Missing `adj_close` (after deduplication) | 18,374 | rebuilt from reference factor |
| `adj_close` inflated 4x-24x vs. the reference factor | 10,417 | rebuilt from reference factor |
| `open = 0` on a ticker's first-ever observation (PG, MCD, CVX, JNJ, MRK, 1960s) | 5 | quarantined: nothing to carry forward |
| Zero-volume days | 75 | kept, flagged for review |
| 3σ extreme returns | 5,056 | kept and flagged: 4,233 confirmed by reference, 820 corporate actions; 0 data errors |
| One-day round trips shared with the reference (TRV 1976, 1978, 1981) | 3 | `reference_error`: raw row interpolated, `adj_close` rebuilt from neighbours' factor |
| **Retained in `ohlc_hw_cleaned`** | **368,246** | 368,246 + 29,460 removed + 5 quarantined = 397,711 |

Raw open/high/low/close/volume in `ohlc_hw` match the reference exactly; the faults are in the ticker labels,
duplicates, `adj_close`, the five zero opens, and three TRV errors inherited from the reference. Only those
three rows changed when the `reference_error` rule was added; no 2024 data (used by the reports) changed. Histories end on different dates (most tickers on
2024-10-18; AMZN, CVX, DIS, IBM, KO, MRK, NKE, PG on 2024-12-06), which the reports must disclose.

### Assumptions

* The supplied `ohlc` dates for each ticker are the reference trading calendar over that ticker's history.
* `ohlc` raw prices and adjusted closes are trustworthy enough to serve as evidence and adjustment factors.
* A >1% disagreement with the reference is treated as evidence of a vendor error.
* The reference can share the vendor's errors, so agreement with it is only evidence that a value was not
  altered in `ohlc_hw`. A one-day adjusted round trip (> 25%, reversed to within 5%) is treated as an error.

### Testing (Part 1)

The notebook builds a tiny delivery with one of every required fault (removed date, all-missing row,
close above high, negative and missing prices, missing volume, 10x row, 2:1 split, missing and inflated
`adj_close`, a one-day spike present in the reference itself, leading gap, exact and conflicting
duplicates) and asserts that the identical pipeline detects and handles each. On the real data it then checks row accounting, six validation rules (including
`adj_close` consistency with the reference factor) and reruns the diagnosis on the cleaned table.

## Part 2: single-stock report

`02_stock_report_template.ipynb` reports on one `ticker` between `start_date` and `end_date`, reading
`ohlc_hw_cleaned` from `db_path` (read-only). The first code cell carries the papermill `parameters` tag.
The committed copy shows the default run (AAPL, 2024).

### Conventions

* **Adjusted close** for all returns, volatility and drawdown; daily returns are **log** returns
  (Part 1's 3σ screen used simple raw-close returns for a different, diagnostic purpose).
* **Volatility:** 20-day rolling sample std × √252. The first 20 values are a warm-up and stay empty
  (never filled with zero), so the first estimate needs 21 prices.
* **Drawdown** is per-date, `(price − running peak) / running peak`; maximum drawdown is its minimum.
* **Sharpe ratio:** `(mean × 252 − 2%) / (std × √252)`; shown as N/A for zero volatility or too few returns.
* Values are kept as fractions and formatted as percentages only for display.

### Input checks

The ticker is trimmed and upper-cased (`" aapl "` works). The report stops with a clear error (which
the batch runner records as a failure) for an unknown ticker,
`start_date` after `end_date`, or fewer than 21 prices in the window. It prints the requested window and
the **actual** first/last observation dates: 22 tickers' data end on 2024-10-18, so a request through
December describes a shorter period, and the report says so.

### Reading the report

* **Price (top left):** the adjusted close coloured red where 20-day volatility is above the period's mean,
  green at or below it, grey during warm-up, with the 50-day moving average (omitted, with a note, if
  there are fewer than 50 prices). The colouring uses the whole period's mean, so it is retrospective.
* **Daily returns (top right):** green up days, red down days.
* **Rolling volatility (bottom left):** shaded above/below its mean, which is shown as a dashed line.
  A single large return lifts the 20-day estimate for exactly 20 days and then drops out, so a flat-topped
  plateau with sudden edges (e.g. MRK, Aug 1–29, 2024, after one −10% day) is one event, not a lasting
  change in risk.
* **Drawdown (bottom right):** how far the price is below its previous peak; the deepest point is marked.
* **Summary table:** total return `last/first − 1`; annualized return `exp(Σ log r × 252 / n) − 1`;
  whole-period annualized volatility; Sharpe; maximum drawdown; best/worst day; share of positive days
  (out of all nonmissing returns, zero days included); average up/down day (N/A if there are none).
  **Annualized figures extrapolate** the window to a full year; over the ~4–6 month windows used here they
  are much larger than what was earned (e.g. IBM: +39.9% total, +96.7% annualized), and the report prints
  a note when the window is shorter than a year.

### Outputs for Part 3

Each report records `ticker`, `first_date`, `last_date`, `trading_days`, `avg_daily_return`,
`avg_volatility` (mean rolling volatility), `max_drawdown`, `sharpe_ratio` and `total_return` with
Scrapbook (`None` when undefined), so the batch runner reads results only from reports that succeeded.

### Testing (Part 2)

The notebook checks the metric functions on tiny inputs (e.g. prices 80 → 100 → 70 → 85 give a
−30% maximum drawdown but a −15% final drawdown). It was also run through papermill for AAPL 2024,
AAPL and AMZN for June–November 2024, a 34-price window (50-day MA omitted with a note), and four
failure cases (unknown ticker, empty window, fewer than 21 prices, reversed dates), each of which stops
with the intended message.

## Part 3: all-stock batch

`03_run_all_stocks.py` executes the Part 2 template for every ticker in `ohlc_hw_cleaned` with papermill.

* **Window:** anchored to the latest cleaned observation, not the computer clock: the six calendar months
  before the month that contains it (even if that observation is a month-end, since a final trading date
  doesn't prove the month's delivery is complete). The latest date is 2024-12-06, so every ticker uses the
  same requested window, **2024-06-01 to 2024-11-30**.
* **Outputs** go to `reports/{run date}/` (the date the batch ran, unrelated to the analysis window):
  `{TICKER}_report.ipynb` for each stock, `failed_stocks.txt` (one `TICKER: error` line per failure; empty
  if none), `portfolio_summary.csv` and `risk_return_profile.png`.
* **Errors:** each ticker runs in its own `try/except`; a failure is logged and the batch continues. The
  script prints `Successfully generated N/M reports`. If every ticker fails, it says so, writes the failure
  list, skips the summary and exits with code 1.
* **Collecting results:** metrics are read with Scrapbook only from the reports that succeeded in this run,
  never by scanning the folder, so failed or stale notebooks there are ignored.
* **Summary:** `portfolio_summary.csv` holds `ticker`, `avg_daily_return` (mean daily log return),
  `avg_volatility` (mean 20-day rolling volatility), `max_drawdown`, `sharpe_ratio`, `total_return`,
  `report_path`, plus each report's actual `first_date`, `last_date` and `trading_days`. Values are
  fractions. Stocks are sorted by Sharpe; the script prints the top and bottom 10, leaving out undefined
  Sharpe values (with fewer than 20 ranked stocks the two lists overlap).
* **Chart:** average volatility vs. total return, coloured by Sharpe, top 5 returns labelled. Circles had
  data for the full window; triangles' data end early.

### Results (run on 2026-10-29)

30/30 reports succeeded. Highest Sharpe: IBM (3.22), HD, SHW, WMT, MCD. Lowest: MRK (−1.93), BA, NKE.

**Coverage caveat:** 22 of the 30 tickers' data end on 2024-10-18, so their reports cover
2024-06-03 to 2024-10-18 (about 97 trading days) while the other 8 (AMZN, CVX, DIS, IBM, KO, MRK, NKE, PG)
run to 2024-11-29 (126 days). Total returns over these unequal periods are not directly comparable;
Sharpe and volatility are annualized, which helps, but they still describe different market periods.
Prices are never extrapolated through the missing weeks.

### Sample output committed

Only three of the 30 reports are committed (the rest are gitignored; rerun the script to regenerate
them; the `.gitignore` rules are pinned to the `2026-10-29` run, so a later rerun is never committed by
accident): `AAPL` (data end early), `IBM` (highest Sharpe, full window) and `MRK` (lowest Sharpe, full window),
together with `portfolio_summary.csv` and `risk_return_profile.png`. `failed_stocks.txt` is not committed
because all 30 reports succeeded, so it is empty. The CSV's `report_path` lists all 30 reports as generated locally.

### Testing (Part 3)

The window rule was checked on 2024-12-06 → Jun 1–Nov 30, a month-end (2024-11-30 → May 1–Oct 31),
a year boundary (2025-01-15 → Jul 1–Dec 31) and a leap year (2024-03-31 → Sep 1, 2023–Feb 29, 2024).
A run with an invalid ticker logged it and continued (2/3), and a stale report placed in the folder was
not collected. A run where every ticker was invalid exited with code 1 and no summary.

## Assumptions and caveats (Parts 2-3)

* **Calendar and rates:** 252 trading days per year; a constant 2% annual risk-free rate (the course
  approximation); Sharpe uses the arithmetic mean of daily log returns, as the assignment specifies.
* **Window:** one common requested window for all stocks, chosen from the data's latest date (see Part 3);
  each report's actual first/last dates are shown and saved, and prices are never extrapolated.
* **Unequal coverage:** 22 stocks have ~97 trading days in the window and 8 have 126, so cross-stock
  comparisons (especially total return) describe different periods.
* **Survivorship and selection bias:** the 30 stocks are the Dow Jones Industrial Average as of late 2024.
  Several were added to the index after strong performance (e.g. AMZN in 2024, NVDA and SHW in
  November 2024; CRM, AMGN and HON in 2020), and companies that dropped out are absent. Rankings and returns
  are therefore not representative of what an investor choosing stocks at the start of a period would
  have experienced.
* **Retrospective colouring:** the high/low volatility threshold is the mean over the whole report
  period, which is only known at the end; it describes the period, it is not a trading signal.
* **Annualization:** see "Reading the report"; annualized returns over short windows overstate what was
  earned.
