# HW2: Scalable Stock Analysis Pipeline

A three-stage pipeline that cleans the intentionally dirty `ohlc_hw` table, produces a parameterized
risk report for one stock, and runs that report across the whole portfolio.

| Component | Purpose |
|---|---|
| `01_data_cleaning.ipynb` | Diagnose and repair `ohlc_hw`; save the validated `ohlc_hw_cleaned` table plus an issue report and a quarantine table |
| `02_stock_report_template.ipynb` | Parameterized single-stock risk report (metrics, 2×2 chart, 10-metric summary, Scrapbook outputs) |
| `03_run_all_stocks.py` | Run the template for every ticker with papermill, log failures, build the portfolio summary and risk-return plot |
| `hw2_cleaning.py` | Cleaning functions used by notebook 01 (pure functions, unit-tested) |
| `tests/` | 53 pytest tests for cleaning, report metrics and the batch script (`test_cleaning.py`, `test_report_template.py`, `test_run_all_stocks.py`) |

## Setup and execution

**Environment:** developed and tested with Python 3.12. Install the dependencies from `requirements.txt`
into a virtual environment inside this folder:

```bash
python3.12 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install pytest                   # only needed to run the tests
```

`requirements.txt` pins every package version used (pandas 3.1, numpy, matplotlib, papermill, scrapbook,
tqdm, ipykernel, jupyterlab and their dependencies). `pytest` is not in it because the homework code does not
need it at run time.

**Database:** download `data.db` from bCourses and place a copy at `data/data.db` inside this folder. It
must contain the instructor tables `ohlc`, `ohlc_hw` and `market_caps`. The database is never committed.

**Run order** (from this folder, in the environment above):

```bash
# 1. clean the data -> writes ohlc_hw_cleaned, ohlc_hw_quality_report, ohlc_hw_quarantine into data/data.db
jupyter nbconvert --to notebook --execute --inplace 01_data_cleaning.ipynb

# 2. (optional) one report by hand, e.g. in JupyterLab: open 02_stock_report_template.ipynb and Run All

# 3. all reports + portfolio summary -> reports/{run_date}/
python 03_run_all_stocks.py
python 03_run_all_stocks.py --tickers AAPL KO   # a quick test on a few tickers

# tests (run from this folder so hw2_cleaning.py is importable)
python -m pytest tests
```

Notebook 01 re-creates the three tables from `ohlc_hw` each time (`if_exists="replace"`); `ohlc_hw` and
`ohlc` are only read. The script runs reports with the notebook's `python3` kernel, i.e. the Python
environment that launched it. Options: `--db-path`, `--reports-root`, `--run-date`, `--kernel`.

## Data cleaning (`01_data_cleaning.ipynb`)

The notebook diagnoses first, saves that diagnosis, then repairs and re-validates.

| Step | What was done | Result on the supplied data |
|---|---|---|
| Ticker labels | Strip HTML tags and entities, remove whitespace, upper-case — **before** any grouping. Original label kept in `ticker_original` | 150 raw labels → 30 tickers |
| Duplicates | Exact duplicates (all values equal, NaN = NaN) removed. Rows sharing `(ticker, ts)` with different values would be quarantined, not resolved by "keep first" | 29,460 exact duplicates removed; 0 conflicting keys |
| Missing cells | Counted per column | Only `adj_close` (19,855 raw; 18,374 after de-duplication) |
| Missing trading days | Each ticker's dates in the supplied `ohlc` table are its reference calendar (weekdays would include holidays). Demonstrated on an example with one date removed | 0 missing days |
| Adjusted close | Compared with `close × (ohlc.adj_close / ohlc.close)` for the same `(ticker, ts)`, used only where raw close equals the reference close. Missing or inconsistent values rebuilt from that same-date factor; the factor is never carried to another date. Vendor value kept in `adj_close_vendor` | 18,374 filled; 10,417 vendor values replaced (they were 4–24× too high) |
| Impossible raw prices | Missing/nonpositive/non-finite OHLC or `low ≤ open, close ≤ high` violated → whole price row carried forward from the previous valid row of the same ticker. A leading row with nothing before it is quarantined | 5 rows (`open = 0` on each ticker's first day) quarantined |
| Volume | Missing → 0 with a repair flag (unknown, not "no trading"); reported zeros kept with `zero_volume_flag` | 0 missing; 75 zero-volume days (1962–1982) kept and flagged |
| 3σ returns | Simple returns of raw close per ticker; flag `abs(r − mean) > 3·std` (sample std, both tails); undefined returns excluded; zero/undefined std reported as unscreened | 5,056 days flagged (see *Outlier handling*) |

**Validation** (`validate_clean`, run in the notebook and again on the table read back from SQLite):
no missing, non-finite or nonpositive values in raw OHLC or `adj_close`; OHLC bounds hold; volume finite and
≥ 0; unique `(ticker, ts)`; `ts` stored as `YYYY-MM-DD` text. All checks pass. The diagnosis is re-run on
the cleaned data: no high-severity issue remains.

**Row accounting:** 397,711 raw = 29,460 duplicates removed + 0 all-missing rows removed + 5 quarantined +
**368,246 retained**. A fingerprint confirms `ohlc_hw` is unchanged after writing.

**Saved tables:** `ohlc_hw_cleaned` (with `ticker_original`, `adj_close_vendor`, `repair_flags`,
`zero_volume_flag`, `return_outlier`, `return_outlier_class`, `raw_rowid`; unique index on `(ticker, ts)`),
`ohlc_hw_quality_report` (63,392 rows: `ticker | issue_type | issue_date | description | severity`) and
`ohlc_hw_quarantine` (5 rows with `quarantine_reason`).

Severity: **high** — unusable in calculations (impossible prices, inconsistent `adj_close`, conflicting
duplicates); **medium** — repairable from a trusted source or an absent observation; **low** — informational
review flag (3σ returns, zero volume, removed exact duplicates).

## Metric definitions (`02_stock_report_template.ipynb`)

All report metrics use **adjusted close** and daily **log returns** `r_t = ln(P_t / P_{t−1})`; 252 trading
days per year; risk-free rate 2%. Values are fractions; percentages are display formatting only.

| Metric | Definition |
|---|---|
| `calculate_returns` | `log(P_t / P_{t−1})`; first value NaN (undefined, not zero) |
| `calculate_volatility` | `returns.rolling(20).std() × √252`; needs 20 returns = 21 prices; warm-up stays NaN |
| `calculate_max_drawdown` | Per-date drawdown `(P − running max) / running max`; maximum drawdown = its `.min()` |
| `calculate_sharpe_ratio` | `(mean(r) × 252 − 0.02) / (std(r) × √252)`; NaN with < 2 returns or zero volatility |
| Total return | `last adj close / first adj close − 1` |
| Annualized return | `exp(sum(r) × 252 / n) − 1`, n = nonmissing returns |
| Annualized volatility | `std(r) × √252` over the whole period (not the mean rolling volatility) |
| Best / worst day, average up / down day | From log returns; empty subset → `N/A` |
| Positive days | positive returns / all nonmissing returns (zero days count in the denominator) |

A report stops with a clear error when the ticker has no data or fewer than 21 prices in the window, so
the batch records it as a failure. Unit tests check every formula against hand calculations, and the
batch metrics were recomputed independently from the database (agreement within 1e-12).

## Design choices and assumptions

- **Reference table use.** `ohlc` is used only as a calendar and as the source of same-date adjustment
  factors, as the assignment allows; the dirty data are never replaced wholesale.
- **Raw prices appear split-adjusted.** AAPL's raw close does not jump at its 2014 (7:1) or 2020 (4:1)
  splits; `adj_close` differs from `close` mainly through dividends. A 3σ flag is therefore not assumed
  to be a split.
- **Whole-row forward fill.** Filling only the bad cell can violate that day's high/low; carrying the
  full price row keeps OHLC bounds and raw/adjusted consistency. No backward fill, no borrowing across tickers.
- **Quarantine instead of guessing** for leading invalid rows and (if present) conflicting duplicates.
- **Analysis window.** Six calendar months before the month of the latest cleaned observation
  (2024-12-06) → **2024-06-01 to 2024-11-30**, the same for every ticker; the computer clock is not used.
  The output folder `reports/{YYYY-MM-DD}/` is the run date.
- **Results collection.** Metrics are read with Scrapbook only from reports that completed in the
  current run and match its ticker and window; stale or failed notebooks in the folder are never read.
  Undefined metrics are stored as `None` in the notebook and become NaN in the CSV.
- **Template not executed in place.** `02_stock_report_template.ipynb` is kept without outputs; executed
  examples are in `reports/`.

## Outlier handling

**Policy:** a 3σ return is a review flag, not a deletion rule. Flags are kept in `return_outlier`.
`return_outlier_class = matches_reference` means the return's two closes equal the reference table, so the
vendor did not mistype them. All 5,057 flags on the cleaned data match. **Matching the reference does not
independently prove a price is correct**: both tables can carry the same bad value. Nothing was
interpolated, because the assignment interpolates only *confirmed* errors and none was confirmed.
`interpolate_confirmed_errors` (interpolate inside a ticker between valid neighbours, never at the ends) is
implemented and tested for that case.

**Three suspicious TRV spikes — flagged, not interpolated.** On 1976-10-28, 1978-12-20 and 1981-08-10 the
TRV close roughly doubles (+100.0%, +99.8%, +100.0%) for one day with open = close and **zero volume**,
then reverts the next day on normal volume. This pattern points to a bad print, but:
- the reference `ohlc` table contains identical close, adjusted close and volume for all nine rows around
  these dates, so it is not a second, independent observation and cannot supply a correct price;
- without a confirmed error or an independent source, interpolation would replace an observed value with
  an unverifiable estimate.

They remain unchanged, flagged (`return_outlier`, `zero_volume_flag`) and documented, and they lie decades
before the analysis window. Also noted: **MCD 1968-01-02** falls about 75% permanently in raw and adjusted
prices on heavy volume — possibly an unadjusted split, not verified; kept and flagged.

**Inside the analysis window** (2024-06-01 to 2024-11-30) 20 stock-days are 3σ-flagged; all match the
reference and are included in the metrics. The largest single-day moves are NKE (log return −22%,
2024-06-28) and MMM (+21%, 2024-07-26). Their causes were not verified against news sources. 252 rows in
the window carry `adj_close` values rebuilt from the reference factor (162 filled, 90 replaced).

## Reading the reports

**Single-stock report** (`reports/{run_date}/{TICKER}_report.ipynb`) prints the requested window, the
observed first/last dates and trading days, and a note when data end before the requested end. Chart
panels (16×12 in, main title `{Ticker} Risk Analysis Report - {start} to {end}`):

- **Price and 50-day MA** — adjusted close colored by 20-day volatility: red above the report's mean
  rolling volatility, green at or below, gray during the warm-up. The dashed 50-day average is absent (with
  an explanation) when fewer than 50 prices exist. Coloring is retrospective.
- **Daily returns** — green bars for gains, red for losses, zero line.
- **Rolling volatility (20-day)** — annualized, with its mean; shaded red above and green below.
- **Maximum drawdown** — drawdown from the running peak (red gradient); the worst point is marked.

**Summary table:** the ten metrics above. Annualized return compounds the average daily log return, so a
short window can produce a large annualized figure; Sharpe above 1 is strong for a single stock and
negative means the stock did worse than the 2% risk-free rate.

**`portfolio_summary.csv`** — exactly the seven required columns: `ticker`, `avg_daily_return` (mean daily
log return), `avg_volatility` (mean 20-day rolling volatility), `max_drawdown`, `sharpe_ratio`,
`total_return`, `report_path`; sorted by Sharpe, descending. The script prints the top and bottom 10 by
Sharpe, excluding undefined values (with fewer than 20 stocks the groups overlap).

**`risk_return_scatter.png`** — x: average rolling volatility; y: total return; color: Sharpe on a scale
centred at zero (red negative, green positive); top 5 by return labelled; circles = full window,
triangles = data end early. Also written: `report_coverage.csv` (requested vs observed dates,
`full_coverage` flag), `failed_stocks.txt` (one ticker per line), `failed_stocks.json` (errors),
`run_manifest.json`. In the full run all 30 reports succeeded, so the failure files are empty.

## Data coverage limitation

The requested window is the same for every stock, but **observed coverage is not**:

- **22 tickers end on 2024-10-18** (97 trading days in the window): AAPL, AMGN, AXP, BA, CAT, CRM, CSCO,
  GS, HD, HON, JNJ, JPM, MCD, MMM, MSFT, NVDA, SHW, TRV, UNH, V, VZ, WMT.
- **8 tickers have data through 2024-11-29** (126 trading days): AMZN, CVX, DIS, IBM, KO, MRK, NKE, PG.

No prices were extrapolated past a ticker's last observation. Metrics for the 22 shorter histories describe
June 1 – October 18 only, so **the rankings mix two different observation periods and are not perfectly
comparable** — 9 of the top 10 Sharpe ratios belong to the shorter histories, whose period excludes
late October and November. Compare like with like using `report_coverage.csv` or the scatter markers.
Each partial report states its end date.

## Submission contents

```
Homeworks/HW2/Anjali_Sudhakar_Bandewar/
├── 01_data_cleaning.ipynb          # executed, recreates ohlc_hw_cleaned
├── 02_stock_report_template.ipynb  # parameters cell tagged "parameters"
├── 03_run_all_stocks.py
├── hw2_cleaning.py                 # imported by 01
├── tests/                          # test_cleaning.py, test_report_template.py, test_run_all_stocks.py
├── requirements.txt
├── README.md
├── .gitignore
└── reports/2026-10-29/
    ├── portfolio_summary.csv       # all 30 stocks (see note on report_path below)
    ├── AAPL_report.ipynb           # partial coverage (ends 2024-10-18)
    ├── IBM_report.ipynb            # full coverage, highest Sharpe
    └── NKE_report.ipynb            # full coverage, negative Sharpe, flagged extreme days
```

`.gitignore` keeps exactly these four files from `reports/` and excludes everything else there.

**About `report_path`:** `portfolio_summary.csv` comes from the full local run and has one row per stock,
so it lists `report_path` for all **30** reports. Only the **3** sample notebooks above are submitted: the
paths for AAPL, IBM and NKE resolve in this submission, and the other **27** paths do **not**. To get every
report, rebuild the cleaned table with notebook 01 and run `python 03_run_all_stocks.py`. That writes a new
folder named after the date it is run, so pass `--run-date 2026-10-29` to reproduce these exact paths.

Also generated by the full run but **not submitted**: the other 27 reports, `report_coverage.csv`,
`risk_return_scatter.png`, `failed_stocks.txt`, `failed_stocks.json` (both empty: no failures) and
`run_manifest.json`. Never submitted: `data/`, any `*.db` file, `.venv/` and caches.
