# Homework 2 — Scalable Stock Analysis Pipeline

Vasileios Dimitropoulos · UCB MFE Python Pre-Program 2026

The pipeline cleans the dirty `ohlc_hw` table, produces a parameterized risk report for one stock, and runs that
report for all 30 stocks with papermill to build a portfolio summary.

## Contents

| file | purpose |
|---|---|
| `01_data_cleaning.ipynb` | Part 1: inspect, diagnose, clean and validate `ohlc_hw`; writes `ohlc_hw_cleaned` |
| `02_stock_report_template.ipynb` | Part 2: papermill template (cell tagged `parameters`) for a single-stock risk report; saved executed with its default parameters (AAPL, 2024) |
| `03_run_all_stocks.py` | Part 3: runs the template for every ticker and builds the portfolio summary |
| `requirements.txt` | pinned dependencies |
| `reports/2026-10-25/` | sample output: 3 reports and `portfolio_summary.csv` (see below) |

The database (`data/data.db`) and the virtual environment (`HW2/`) are local only and are not committed.
As the assignment asks, only 3 of the 30 generated reports are committed, together with the summary CSV:
**IBM** (best Sharpe, full 6-month coverage), **AAPL** (data ends 2024-10-18, so it shows the shorter-coverage
disclosure), and **NKE** (negative return, with a −20% single day). The CSV's `report_path` column lists all 30
reports of the run. Running `03_run_all_stocks.py` regenerates all of them, plus `risk_return.png`,
`failed_stocks.txt` and `run.log`.

## How to run

1. **Environment** (Python 3.12.9 with uv, the same setup as HW1), from this folder:

   ```bash
   uv venv HW2 --python 3.12.9
   uv pip install --python HW2/Scripts/python.exe -r requirements.txt   # macOS/Linux: HW2/bin/python
   ```

2. **Database**: download `data.db` from bCourses and place it at `data/data.db` (relative to this folder).
   It must contain the `ohlc_hw` and `ohlc` tables.

3. **Run in order**, from this folder:

   ```bash
   # Part 1: creates ohlc_hw_cleaned (plus quarantine/audit/quality tables) in data/data.db
   HW2/Scripts/jupyter-nbconvert --to notebook --execute --inplace 01_data_cleaning.ipynb

   # Part 2 (optional, single report): open 02_stock_report_template.ipynb and run it,
   # or change the parameters cell (ticker, start_date, end_date, db_path)

   # Part 3: all reports + summary into reports/<today>/
   HW2/Scripts/python 03_run_all_stocks.py
   ```

   Part 3 takes about 2–3 minutes for 30 tickers. The notebooks can also be run interactively in Jupyter
   with the HW2 environment as the kernel.

## Data issues found and how they were handled

Every check was run on the real data. Checks that found nothing were tested on small injected examples in the
notebook. Faults were never invented in the real data.

| issue | rows | handling |
|---|---:|---|
| HTML markup in ticker labels (`<a><span style=…>AAPL</span></a>`) | 39,936 | normalized (strip tags, whitespace, upper case) **before** grouping/deduplicating; original labels kept in `ohlc_hw_ticker_audit` |
| exact duplicate records | 29,460 | removed (one copy kept) |
| conflicting `(ticker, ts)` keys | 0 | rule: keep the version that matches `ohlc`, otherwise quarantine; demonstrated on toy data |
| missing trading days | 0 | calendar = each ticker's dates in `ohlc`; demonstrated on toy data |
| missing raw prices / volume | 0 | forward-fill within ticker (kept inside the day's range) / 0 + `volume_filled`; demonstrated |
| missing `adj_close` | 18,374 | rebuilt as `close × (adj_close / close)` of the **same** `(ticker, ts)` in `ohlc` |
| corrupted `adj_close` (4–24× too large) | 10,417 | detected by a jump in `adj_close / close` vs a 21-day rolling median; rebuilt the same way |
| `open = 0` on a ticker's first day | 5 | Option 1 (NaN → forward-fill) is impossible for a first row → **quarantined** |
| one-day doubled prints (TRV 1976, 1978, 1981) | 3 | confirmed errors → interpolated between neighbours |
| 3σ return outliers | 5,056 | kept, with an evidence note |
| zero-volume days | 75 | kept, flagged |
| Sunday-dated rows (Jan–Apr 1974, Feb–Apr 1975; 16 tickers) | 384 | kept, flagged `weekend_date` (see below) |

Counts are unique rows after deduplication (the raw table shows 19,855 missing `adj_close` including duplicates).
Row accounting: 397,711 raw = 368,246 retained + 29,460 duplicates removed + 5 quarantined.
Raw OHLC bounds and adjusted-price consistency are **rechecked after every repair step** (a table in the notebook).
After cleaning, the table passes every rule: no missing, non-finite or nonpositive prices; unique keys; valid OHLC
bounds; consistent adjusted prices; ISO dates.

## Key design decisions

* **Normalize first, then deduplicate.** Otherwise the five label variants of a stock would be treated as five
  tickers and 29,460 duplicates would go unnoticed.
* **Diagnose before repairing.** The full Data Quality Report (`ticker | issue_type | issue_date | description | severity`)
  is saved before any change and recomputed after cleaning (`ohlc_hw_quality_report_before/_after`).
* **Every change is traceable.** The cleaned table carries repair flags (`price_filled`, `price_interpolated`,
  `volume_filled`, `adj_close_repair`, `return_outlier`, `outlier_note`, `zero_volume`, `weekend_date`, `repair_note`).
  Rows that could not be repaired defensibly are kept in `ohlc_hw_quarantine` with a reason.
  The raw `ohlc_hw` table is never modified.
* **`ohlc` is a reference, not a replacement.** It gives the trading calendar and date-specific adjustment factors.
  Raw OHLC in `ohlc_hw` matched it exactly, which is what allows trusting its factors.
* **No forward-filling of adjusted prices or factors.** The factor changes on ex-dividend and corporate-action dates,
  so carrying it forward could cross such a date.
* **Severity:** *high* = the value is certainly wrong and would corrupt the metrics (impossible prices,
  nonpositive values, corrupted `adj_close`, conflicting keys). *medium* = missing information or something
  to review (missing values or dates, 3σ returns). *low* = mechanical or harmless for price metrics (markup,
  exact duplicates, zero volume).

## How outliers were handled and why

* **Impossible price relationships** (`close > high`, `close < low`, open outside the range): **Option 1**.
  The offending value is set to NaN, forward-filled within the ticker, then clipped into that day's observed
  `[low, high]` range, because forward-filling alone can break today's range. `adj_close` is then rebuilt from
  the repaired close so the adjustment convention stays consistent. Option 3 (rebuilding from O/H/L) would
  invent a value we cannot check. The real data had no such rows besides the five `open = 0` first days.
* **Extreme returns (> 3σ, sample std, both tails, per ticker)** are a **review flag, not a deletion rule**.
  Each one is investigated and given an evidence note:
  * *Confirmed data error*: a ≥ 50% move that fully reverses the next day. Only TRV's three doubled prints
    qualify (all four prices exactly 2× the neighbours, volume 0). These are interpolated.
  * *Market-wide move*: ≥ 5 of 30 tickers flagged on the same date (e.g. 1987, 2008, March 2020). Kept.
  * *Possible corporate action (split, merger/spin-off, special dividend)*: the adjustment factor changed > 5% on
    that day and stayed. Kept. With no corporate-actions or news feed we cannot tell these apart, so the note says
    "not verified" and no event is named.
  * *Vendor factor blip*: the factor jumped and was back the next day (e.g. AAPL 1990-10-19). This is a quirk of the
    reference's adjustment factor, not a corporate action. Kept and labelled as such; all are before 1991.
  * *Otherwise*: kept as a single-stock move whose raw close agrees with `ohlc`; the cause (news, earnings, …)
    is explicitly **not verified**. I did not claim news events without evidence.

## Assumptions

* The reference calendar is each ticker's dates in `ohlc` over its supplied history, so exchange holidays are not
  counted as gaps.
* Because that calendar could share a vendor problem, the dates themselves were also checked. In Jan–Apr 1974 and
  Feb–Apr 1975 no ticker has a Friday row: 16 tickers have a Sunday row instead (most likely the Friday session
  stamped with Sunday's date), and the others simply lack that day. The rows are kept and flagged `weekend_date`
  rather than re-dated, because that would need an external exchange calendar to confirm. Row order, and so every
  return, is unaffected, and both periods are far outside every report window.
* In this data set, raw `close` is already split-adjusted, and `adj_close` additionally includes dividends.
  Reports use `adj_close`.
* Part 3 window: the six calendar months **before** the month of the latest cleaned observation
  (2024-12-06 → **2024-06-01 to 2024-11-30**), anchored to the data, not the computer clock. The output folder
  is named after the **run date**.
* Course conventions: 252 trading days, 2% annual risk-free rate, Sharpe =
  `(mean log return × 252 − 0.02) / (std × √252)`.

## How to interpret the reports

Each report (`reports/<run date>/<TICKER>_report.ipynb`) contains:

* **Coverage**: the requested window and the dates actually observed. **22 of the 30 tickers have data only until
  2024-10-18**, so their metrics cover about 4.5 months instead of 6 and are not fully comparable with the 8 tickers
  that run to 2024-11-29. The CSV includes `first_observation`, `last_observation` and `trading_days` to make this
  visible. Prices are never extrapolated.
* **Top-left**: adjusted price, coloured red where the 20-day volatility is above the window's mean, green where it
  is at or below, gray during the 20-day warm-up. The dashed line is the 50-day moving average, which appears only
  after 50 prices.
* **Top-right**: daily log returns (green up, red down).
* **Bottom-left**: 20-day annualized rolling volatility versus its mean, with high and low regimes shaded.
* **Bottom-right**: drawdown from the running peak; the black dot is the maximum drawdown.
* **Summary table**: total return, compounded annualized return, whole-period annualized volatility, Sharpe,
  maximum drawdown, best/worst day, % positive days and average up/down day (log returns).
  **N/A** means undefined (too little data or zero volatility), not zero.

`portfolio_summary.csv` holds one row per successful report, sorted by Sharpe ratio. All values are fractions:
`avg_daily_return` is the mean daily log return, `avg_volatility` is the **mean of the 20-day rolling volatility**
(not the whole-period volatility in the report table), `max_drawdown` is the worst drawdown, and `total_return` is
`last / first adj_close − 1`. Results are collected only from this run's successful reports, through the scrapbook
scrap `report_results`. Failed tickers go to `failed_stocks.txt` and `run.log`. If every report fails, no summary
is produced. Rankings exclude undefined Sharpe ratios.

**Results for 2024-06-01 to 2024-11-30 (30/30 reports succeeded):** IBM had the best risk-adjusted performance
(Sharpe 3.2, +39.9%), followed by HD, SHW, WMT and MCD. The bottom of the ranking was MRK (−19.9%), BA and
NKE, the only stocks with negative returns. NVDA had by far the highest average volatility (~61%).
