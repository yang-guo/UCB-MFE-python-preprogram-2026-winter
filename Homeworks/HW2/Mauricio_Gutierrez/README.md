# HW2: Scalable Stock Analysis Pipeline (Mauricio Gutierrez)

Clean a dirty daily OHLC table, build a parameterized single-stock risk report, and run it for every stock with papermill.

## Files

| File | Part | What it does |
|---|---|---|
| `01_data_cleaning.ipynb` | 1 | Inspects `ohlc_hw`, builds a Data Quality Report, cleans the data, validates it and saves `ohlc_hw_cleaned` (plus `ohlc_hw_quarantine`) to `data/data.db` |
| `02_stock_report_template.ipynb` | 2 | Parameterized risk report for one ticker and date range (cell tagged `parameters`) |
| `03_run_all_stocks.py` | 3 | Runs the template for every ticker over the last six complete months, then builds the portfolio summary |
| `requirements.txt` | | Python dependencies |
| `reports/<run date>/` | 3 | Sample reports, `portfolio_summary.csv`, `risk_return_profile.png`, `failed_stocks.txt` |

`data/data.db` is **not** committed. Download it from bCourses and place it at `data/data.db` inside this folder.

## How to run

From the repository root, with the course environment (`uv sync --locked` creates it):

1. **Part 1:** open `01_data_cleaning.ipynb` and run all cells. This (re)creates the `ohlc_hw_cleaned` table; the raw `ohlc_hw` table is never modified.
2. **Part 2 (one stock):** open `02_stock_report_template.ipynb`, edit `ticker`, `start_date`, `end_date` in the parameters cell, and run all cells.
3. **Part 3 (all stocks):**
   ```bash
   uv run python Homeworks/HW2/Mauricio_Gutierrez/03_run_all_stocks.py
   ```
   Output goes to `reports/<today's date>/`.

Paths are resolved relative to the files themselves, so Part 3 works from any working directory. Part 3 runs each report with the same Python interpreter that runs the script (kernel `python3`), so it does not depend on the kernel name saved in the template.

## Part 1: data issues found and how they were handled

The Data Quality Report (`ticker | issue_type | issue_date | description | severity`) found these issues in the 397,711 raw rows:

| Issue | Count | Severity | Handling |
|---|---|---|---|
| Ticker labels wrapped in HTML (`<span …>KO</span>`): 150 labels for 30 stocks | 120 labels | medium | Normalized (HTML, quotes, spaces removed, upper-cased) **before** grouping or de-duplicating; the original label is kept in `raw_ticker` |
| Exact duplicate rows | 29,460 | low | Removed (identical values, nothing lost) |
| Conflicting `(ticker, ts)` records | 0 | high | Would be quarantined (no way to know which version is right) |
| Missing `adj_close` | 18,374 | medium | Rebuilt as `close × (ohlc.adj_close / ohlc.close)` on the same `(ticker, ts)`, i.e. a trusted date-specific adjustment factor from the supplied `ohlc` table |
| `adj_close` spikes (4–24× too large for one day) | 10,417 | medium | Same rebuild. A spike detector (adjustment factor jumping away from both the days before and after) found 10,404 of them; the rebuild uses any disagreement with the trusted factor, so it also fixes the 13 the detector missed |
| `open = 0` | 5 | high | **Quarantined** (saved in `ohlc_hw_quarantine` with a reason). All five are a ticker's first row, so there is no earlier day to fill from, and making up an open from the same day's prices would be a guess |
| OHLC bound violations (`low > open` on the same 5 rows) | 5 | high | Quarantined with the rows above |
| Missing raw prices, missing volume, rows with all prices missing, missing trading days, `high < close`, negative prices | 0 | | Functions implemented anyway (forward-fill within the ticker, volume → 0 with a flag, drop before filling, quarantine of leading gaps) and demonstrated on a small injected example |
| Zero volume | 75 | low | Kept and reported (zero volume is not proof of an error) |

Missing trading days were checked against the dates in the supplied `ohlc` table for each ticker, only between that ticker's own first and last observation (business days alone would count exchange holidays as gaps).

**Severity policy:** *high* = impossible or ambiguous value with no safe automatic fix; *medium* = probably wrong or missing, but with a documented repair rule; *low* = harmless once handled, or only a review flag.

Every repair is recorded in a flag column saved with `ohlc_hw_cleaned` (`flag_price_ffill`, `flag_volume_filled`, `flag_round_trip_interpolated`, `flag_adj_filled`, `flag_adj_replaced`, `flag_return_outlier`). All raw rows are accounted for: 368,246 retained + 29,460 duplicates removed + 5 quarantined = 397,711.

After cleaning, all validation checks pass: no missing, non-finite or nonpositive prices; unique `(ticker, ts)`; finite nonnegative volume; valid OHLC bounds; `adj_close` consistent with the trusted factor; `ts` stored as `YYYY-MM-DD`.

## How outliers were handled, and why

The 3-standard-deviation screen (per ticker, sample std, both tails) flagged 5,056 daily returns. Before deciding, I checked the evidence:

- **Every flagged row's open, high, low and close is identical to the trusted `ohlc` table**, so none of them is a typing error in `ohlc_hw`. Most coincide with real market events (October 2008, March 2020, AAPL −52% on 2000-09-29 after earnings). These are **kept and flagged** (`flag_return_outlier`); news was not verified row by row.
- **Three rows are confirmed errors** (TRV 1976-10-28, 1978-12-20, 1981-08-10): the whole row doubles (+100%) and returns to the previous level the next day. These are **interpolated**: open, high, low, close and adj_close become the midpoint of the day before and after (`flag_round_trip_interpolated`). The rule that finds them is "a close move above 30% that is undone within 3% the next day"; it matches exactly these three rows. The reference table has the same error on those dates, so its adjustment factor is not used there.
- MCD 1968-01-02 (−75%, no recovery) is uncertain (possibly an unadjusted corporate action) and is kept and flagged.

Raw close prices here are already split-adjusted, so stock splits do not appear as raw jumps.

After cleaning, 5,176 returns are flagged (more than before): removing TRV's three ±100% days lowers TRV's standard deviation, so more of its ordinary large days exceed 3 std.

## Part 2: report design

- Returns, volatility, Sharpe and daily statistics use daily **log returns** of the **adjusted close**; values stay as fractions in calculations and are formatted as percentages only for display.
- Rolling volatility: 20-return sample std × √252. The first 20 values are left missing (warm-up), never filled with zero.
- `calculate_max_drawdown` keeps the assignment's name but returns the per-date drawdown; the maximum drawdown is its minimum.
- Sharpe: `(mean × 252 − 2%) / (std × √252)`, `N/A` with fewer than 2 returns or zero volatility.
- The report fails with a clear message if the ticker is unknown, the dates are invalid or reversed, or there are fewer than 21 prices (20 returns are needed for one volatility estimate).
- The last cell saves the key metrics with scrapbook (`sb.glue`) so Part 3 can collect them without recalculating.

## Part 3: batch run

- **Window:** the six calendar months before the month of the latest observation in `ohlc_hw_cleaned` (latest 2024-12-06 → **2024-06-01 to 2024-11-30**), the same window for every ticker. The month of the latest observation is excluded even if it ends on a month-end.
- **Output folder:** `reports/<run date>/`, the day the script runs (independent of the analysis window).
- **Errors:** each ticker runs inside `try/except`; failures are logged (screen and `run.log`) and listed in `failed_stocks.txt`, and the run continues. If every report fails, no summary is produced.
- **Summary:** built only from the reports that succeeded in this run (never by scanning the folder), from the values each report saved with scrapbook. Sorted by Sharpe ratio; undefined Sharpe values are left out of the rankings. The CSV keeps fractions and adds each report's actual coverage (`first_date`, `last_date`, `trading_days`).

### Results (run of 2026-10-22, window 2024-06-01 to 2024-11-30)

**30/30 reports succeeded** (`failed_stocks.txt` is empty). The full batch was generated locally; this folder includes three sample reports:

| Sample | Why it was chosen |
|---|---|
| `reports/2026-10-22/IBM_report.ipynb` | Best Sharpe ratio (3.22, +39.9%), full coverage |
| `reports/2026-10-22/MRK_report.ipynb` | Worst Sharpe ratio (−1.93, −19.9%), full coverage |
| `reports/2026-10-22/HD_report.ipynb` | Second-best Sharpe, but its data ends 2024-10-18 (shorter coverage) |

Top 3 by Sharpe: IBM 3.22, HD 3.07, SHW 2.99. Bottom 3: MRK −1.93, BA −1.41, NKE −0.90. Full table in `portfolio_summary.csv`; chart in `risk_return_profile.png`.

**Coverage:** the window ends 2024-11-30 (a Saturday), so the last trading day with data is 2024-11-29. Eight stocks have data up to that day; **22 stocks' data ends on 2024-10-18**, so their reports cover about 4.5 months instead of 6 (97 trading days instead of 126). Prices are never extended past a stock's last observation. These stocks' total returns are therefore over a shorter period and are not directly comparable with the full-coverage ones; ratios such as Sharpe and mean volatility are annualized and compare better, but still describe a different period.

## Assumptions

- The supplied `ohlc` table is a trusted reference for trading dates and adjustment factors (the assignment allows this).
- 252 trading days per year; 2% annual risk-free rate (course convention).
- Volume is not used in any metric, so zero-volume days are kept unchanged.
- Return outliers whose prices match the reference are treated as genuine; news was not checked row by row.

## How to interpret the reports

Each report shows, for one stock over the window:

- **Price and 50-day MA:** price line red where 20-day volatility is above its average for the period, green otherwise, gray during the first 20 days. The coloring uses the whole period's average, so it describes the past.
- **Daily returns:** green up days, red down days; tall bars are volatile days.
- **Rolling volatility:** the 20-day annualized volatility against its mean; shaded red when above, green when below.
- **Drawdown:** how far the price is below its previous peak; the dot marks the maximum drawdown.
- **Summary table:** total and annualized return, annualized volatility, Sharpe ratio (return per unit of risk above the risk-free rate; above 1 is good), maximum drawdown, best/worst day, share of positive days and average up/down day.

In the portfolio chart, stocks further up made more money, stocks further right were more volatile, and greener points have a better Sharpe ratio. NVDA stands far to the right (about 61% mean volatility with a +20% return): this is the stock's real behaviour in mid-2024, not a data problem. Its prices match the reference table exactly, its 10-for-1 split on 2024-06-10 shows no jump, and the volatility comes from many large days (16 of 96 moved more than 5%), so its Sharpe ratio is only 0.78. Compare stocks with a shorter coverage (earlier `last_date`) with care: their numbers describe a shorter period.
