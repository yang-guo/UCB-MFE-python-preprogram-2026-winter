# HW2: Scalable Stock Analysis Pipeline (Guillaume Estepa)

This pipeline cleans the dirty `ohlc_hw` table, builds a parameterized single-stock risk report, and runs that report for all 30 tickers with papermill.

## Files

| File | Purpose |
|---|---|
| `01_data_cleaning.ipynb` | Part 1: inspect, diagnose, clean, and validate `ohlc_hw`. Writes `ohlc_hw_cleaned` and its audit tables |
| `02_stock_report_template.ipynb` | Part 2: parameterized risk report for one ticker (`parameters` cell tagged) |
| `03_run_all_stocks.py` | Part 3: runs the template for every ticker, collects results, and builds the portfolio summary |
| `reports/2026-10-29/` | sample reports (AAPL, IBM, MRK), `portfolio_summary.csv`, `risk_return_profile.png`, `failed_stocks.txt` |
| `requirements.txt` | dependencies |

## How to run

All paths are relative to this folder (`Homeworks/HW2/Guillaume_Estepa/`).

```bash
# 0. Environment: the course's uv project already contains every dependency.
#    Alternatively: pip install -r requirements.txt
cd Homeworks/HW2/Guillaume_Estepa

# 1. Put a local copy of the bCourses database here (gitignored, never committed)
mkdir -p data && cp /path/to/data.db data/data.db

# 2. Clean the data (creates ohlc_hw_cleaned in data/data.db)
uv run papermill 01_data_cleaning.ipynb 01_data_cleaning.ipynb --cwd .

# 3. (optional) one report, e.g. MSFT for 2024
uv run papermill 02_stock_report_template.ipynb msft.ipynb -p ticker MSFT -p start_date 2024-01-01 -p end_date 2024-12-31

# 4. All reports + portfolio summary
uv run python 03_run_all_stocks.py              # all 30 tickers
uv run python 03_run_all_stocks.py --tickers AAPL MSFT   # quick subset test
```

The notebooks can also be run top-to-bottom in Jupyter. Run order: **01 → 02 (optional) → 03**.

## What I found in the data

| Issue | Count | Severity | Handling |
|---|---|---|---|
| HTML markup in ticker labels (`<a ...><span style="color: red">AAPL</span></a>`) | 150 labels → 30 tickers (39,936 raw rows) | low | normalized *before* grouping. Original kept in `raw_ticker` plus the `ohlc_hw_ticker_audit` table |
| Exact duplicate records | 29,460 | medium | collapsed, `n_copies` kept |
| Conflicting `(ticker, ts)` keys | 0 | (high) | would be quarantined. Tested on injected data |
| Missing `adj_close` | 18,374 | high | rebuilt from the trusted factor |
| Corrupted `adj_close` (4×–24× too high) | 10,417 | high | rebuilt from the trusted factor |
| `open = 0` on a ticker's first day | 5 | high | quarantined (leading gap, nothing to fill from) |
| One-day price "doubling" (TRV 1976, 1978, 1981) | 3 | high | interpolated (confirmed data error) |
| Other 3σ return outliers | ~5,050 | medium/low | kept and flagged with a category |
| Zero-volume days | 75 | low | kept and flagged |
| Weekend (Sunday) dates, 1974–75 | 384 | low | kept and flagged (vendor convention, present in the reference too) |
| Missing raw prices / volume, missing trading days, `high < close`, `low > close` | 0 | - | detectors and repairs demonstrated on injected examples |

Row accounting: 397,711 raw rows = 368,246 retained + 29,460 exact duplicates + 5 quarantined (0 removed).

## Key design decisions

* **Diagnose → repair → re-validate.** Detectors only return flags. The Data Quality Report (`ticker | issue_type | issue_date | description | severity`, about 64k rows, one row per issue per observation) is built *before* any repair and saved as `ohlc_hw_quality_report`. After cleaning, every check runs again.
* **Nothing is silently dropped.** Every repair has a flag column in `ohlc_hw_cleaned` (`flag_price_ffilled`, `flag_volume_filled`, `flag_ohlc_repaired`, `flag_spike_interpolated`, `flag_adj_reconstructed`, `flag_return_outlier`, `flag_zero_volume`, `flag_weekend_date`, plus `adj_source` and `outlier_category`). Rows I cannot repair defensibly go to `ohlc_hw_quarantine` with a reason. The raw `ohlc_hw` table is left unchanged, which a content hash checks.
* **Adjusted close.** The raw OHLC/volume in `ohlc_hw` are identical to the supplied `ohlc` table, so I use `ohlc` as the trusted source of the date-specific adjustment factor `adj_close / close` at the matched `(ticker, ts)`. Then `adj_close = validated raw close × factor`. I do **not** copy the reference table wholesale. Values that already agree are kept (`adj_source = vendor_verified`). For a row whose raw close I repaired, the factor comes from the neighboring trusted days, but only if the previous and next factors agree within 5% (no corporate action in between). Otherwise the row is quarantined.
* **Impossible OHLC relationships: Option 3 (use the other prices).** A forward-filled open/close is clipped into that day's `[low, high]`. Any other violation rebuilds the envelope (`high = max(O,H,L,C)`, `low = min(O,H,L,C)`). I keep the close because it drives every return. Bounds are re-checked after every repair, because forward-filling alone can violate the current day's range.
* **Missing values.** Rows with all four raw prices missing are dropped *before* filling. Single missing/invalid prices are forward-filled **within the ticker**, never from another ticker. Leading gaps are quarantined, never back-filled. Missing volume becomes 0 with `flag_volume_filled` (unknown volume is not evidence of no trading).
* **Dates** are stored as ISO `YYYY-MM-DD` text, so the report's SQL date filters work. `(ticker, ts)` has a unique index.

## How I handled outliers, and why

The 3σ screen (simple raw-close returns, per ticker, sample std `ddof=1`, both tails) flags 5,056 days over 60 years. Most are real: 1987, 2008, March 2020. So it is a **review flag, not a deletion rule.** Each flagged day gets a category based on evidence:

* `adjustment_explained` (206): the trusted *adjusted* return that day is normal, so the raw jump is a split/dividend that the adjustment absorbs. Kept.
* `confirmed_data_error` (3, all TRV): **four** pieces of evidence agree. The close exactly doubles, the next day fully reverts, volume is zero, and the vendor's own adjustment factor breaks on that day only. These are interpolated linearly between the two valid neighbors (no extrapolation).
* `reversal_of_error` (3): the day after each error, which becomes normal once the error is fixed.
* `unexplained_move` (4,844): kept and flagged. I don't claim a news cause without a news source. For example, AAPL 2000-09-29 (−52% on ~3B shares, price stays down) looks like a genuine move. MCD 1968-01-02 (−75%, persists) looks like an unadjusted split, but I can't verify it from this data, so it stays flagged with that uncertainty stated. AXP 1972-12-14 (−13%, zero volume, reverts) is suspicious, but its adjustment factor is consistent, so it does not meet my bar for a confirmed error.

## Assumptions

* Reference calendar for missing trading days: each ticker's own dates in the supplied `ohlc` table, over that ticker's own first-to-last date in `ohlc_hw`. Business weekdays would count exchange holidays as gaps. Result: **0 missing days**, which I report as such. The detector is demonstrated on a tiny example with one removed date.
* Backward-adjusted convention: `adj_close ≤ close` (factor ≤ 1). This holds for every row of the reference table.
* 252 trading days per year, and a 2% annual risk-free rate for the course Sharpe approximation.
* **Part 3 window.** The latest cleaned observation is 2024-12-06, so the window is the six calendar months before December: **2024-06-01 to 2024-11-30**. The window is anchored to the data, not the computer clock. The output folder (`reports/2026-10-29/`) is the *run date*.
* **Uneven coverage.** 22 tickers' supplied histories end on 2024-10-18, and 8 run to 2024-12-06. All reports use the same requested window, and each report discloses its actual first/last observation. The CSV includes `first_obs`, `last_obs`, and `trading_days`. Prices are never extrapolated through trailing gaps, so short-history tickers are measured over about 97 days instead of 126. Keep that in mind when comparing them.

## How to interpret the reports

Each `{TICKER}_report.ipynb` contains:

* **Top-left: price and 50-day MA.** The adjusted close is colored by volatility regime: red when the 20-day rolling volatility is above its mean for the report, green at or below the mean, gray during the 20-day warm-up. The coloring is retrospective (it uses the whole period's mean).
* **Top-right: daily log returns.** Green bars are up days, red bars are down days.
* **Bottom-left: 20-day rolling volatility (annualized).** It is shaded red above its mean and green below.
* **Bottom-right: drawdown.** The fall from the running peak, with the worst point marked. The worst point is the maximum drawdown.
* **Summary table.** Total return (`last/first adj close − 1`), compounded annualized return, whole-period annualized volatility, Sharpe, max drawdown, best/worst day, % positive days (zeros count in the denominator), and the average up/down day. Undefined values display `N/A`.

`portfolio_summary.csv` (fractions, not percent) contains one row per **successful report from this run**, collected through scrapbook scraps. Failed or stale notebooks in the folder are never read. It is sorted by Sharpe ratio. `avg_volatility` is the *mean of the 20-day rolling volatility*, which is not the same number as the report's whole-period annualized volatility. Tickers with an undefined Sharpe would be excluded from the top/bottom-10 rankings. In `risk_return_profile.png`, points up and to the left are better risk-adjusted performers. These are descriptive statistics for a single 6-month period, not forecasts or investment recommendations.
