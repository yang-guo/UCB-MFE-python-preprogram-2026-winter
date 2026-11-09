# HW2: Scalable Stock Analysis Pipeline (Nirmit Ghughu)

| File | Purpose |
|---|---|
| [`01_data_cleaning.ipynb`](01_data_cleaning.ipynb) | Diagnoses the dirty `ohlc_hw` table, repairs it with flags, validates it, and writes `ohlc_hw_cleaned` |
| [`02_stock_report_template.ipynb`](02_stock_report_template.ipynb) | Parameterized single-stock risk report (cell tagged `parameters`) |
| [`03_run_all_stocks.py`](03_run_all_stocks.py) | Runs the template for every ticker with papermill and builds the portfolio summary |
| [`reports/2026-10-30/`](reports/2026-10-30/) | Sample reports (IBM, NVDA, MRK), `portfolio_summary.csv`, the risk/return plot and `failed_stocks.txt` (empty: all 30 reports succeeded) |
| [`requirements.txt`](requirements.txt) | Dependencies |

## How to run

1. **Environment.** From the repository root, run `uv sync --locked`. It installs everything in `requirements.txt`. With pip instead: `pip install -r Homeworks/HW2/Nirmit_Ghughu/requirements.txt`.
2. **Data.** Download `data.db` from bCourses to `Lectures/Lecture 4/data/data.db`. The cleaning notebook copies it to `Homeworks/HW2/Nirmit_Ghughu/data/data.db` (git-ignored) and writes only to that copy. The course source file is never modified.
3. **Part 1.** Run all cells of `01_data_cleaning.ipynb`. This recreates `ohlc_hw_cleaned`, plus three audit tables, in the local copy. It takes a few seconds.
4. **Part 2 (optional, one stock).** Open `02_stock_report_template.ipynb` and run it, or run it with papermill:
   ```bash
   .venv/bin/papermill Homeworks/HW2/Nirmit_Ghughu/02_stock_report_template.ipynb out.ipynb \
       -p ticker NVDA -p start_date 2024-06-01 -p end_date 2024-11-30 --cwd Homeworks/HW2/Nirmit_Ghughu
   ```
5. **Part 3.** Run:
   ```bash
   .venv/bin/python Homeworks/HW2/Nirmit_Ghughu/03_run_all_stocks.py            # all 30 tickers, ~1 minute
   .venv/bin/python Homeworks/HW2/Nirmit_Ghughu/03_run_all_stocks.py --tickers AAPL NOT_A_TICKER   # failure demo
   ```
   Output goes to `reports/{run date}/`. All paths are resolved from the script's own location, so it works from any working directory.

Only three sample reports are committed. The folder's `.gitignore` excludes the other 27 generated reports and the database.

## Key design decisions

### Part 1: cleaning

- **Diagnose before repairing.** Every detector writes to one Data Quality Report (`ticker | issue_type | issue_date | description | severity`, several issues allowed per row) *before* anything is changed. Each repair is then recorded in a flag column saved with the cleaned data: `price_repair`, `adj_close_repair`, `volume_filled`, `extreme_return_flag`, `spike_reversal_flag`, `adj_close_spike_flag`, `zero_volume_flag` and `n_raw_copies`.
- **Tickers are normalized first.** HTML tags and entities are stripped, whitespace removed and case upper-cased, which turns 150 raw labels into 30 tickers. This happens before any grouping, deduplication or return calculation. Every original label is kept in `ohlc_hw_ticker_audit`.
- **Duplicates.** 29,460 exact duplicate copies are removed. Conflicting `(ticker, ts)` keys would have *all* their versions quarantined instead of keeping the first arbitrarily. The real data has none, so this is shown on an injected example.
- **Missing days vs. missing cells.** The reference calendar is each ticker's dates in the supplied `ohlc` table, as HW2 permits, because business weekdays include holidays. The real data has **zero** missing days, which is a correct zero result. The detector does find a deliberately deleted date in the injected example. Missing days are reported, never filled with invented prices.
- **Adjusted close.** `adj_close / close` (the adjustment factor) is checked against its 11-day neighbourhood. About 10,400 values were inflated 4–24×, and 18,374 were missing. They are rebuilt as *validated raw close × trusted date-specific factor* from the matching `(ticker, ts)` row of `ohlc`. One row on a repaired spike day instead uses neighbouring factors that agree within 1%. No factor is ever carried across an unverified corporate action, and the raw prices are never replaced with reference values.
- **Unrepairable rows are quarantined, not guessed.** Five rows have `open = 0` on a ticker's very first day, with no earlier price to carry forward. They are saved with a reason in `ohlc_hw_quarantine`, and the notebook reconciles retained, removed and quarantined rows back to the raw count.
- **Validation afterwards.** No missing, non-finite or nonpositive raw or adjusted prices, unique keys, valid `low ≤ open/close ≤ high`, finite nonnegative volume, ISO `YYYY-MM-DD` dates, and the raw table proven unchanged by a hash.

### How outliers were handled, and why

- **Impossible price relationships** (`high < close`, etc.): **Option 1**, replace the whole bar with NaN and forward-fill within the ticker. The data can't tell which of the four prices is wrong, and a copied bar is always internally consistent. When only some prices are filled, high/low are widened to contain them, and this is flagged `+range_widened`.
- **Extreme returns (3σ):** simple raw-close returns within each ticker, sample std (`ddof=1`), both tails, undefined returns excluded. About 5,000 returns are flagged. Daily returns are fat-tailed, so the screen is a **review flag, not a deletion rule.** I investigated the largest moves individually:
  - **TRV on 1976-10-28, 1978-12-20 and 1981-08-10 are confirmed errors.** The price doubles for one day and fully reverses the next, with O = H = L = C (and zero volume in 1981). These are linearly interpolated between the valid neighbouring days.
  - **MCD on 1968-01-02 (−75%) is kept and flagged.** The new level persists and fits an unadjusted ~4:1 split, but there are no corporate-action records to confirm it, so I don't claim it is one and don't change it.
  - **Everything else is kept with `extreme_return_flag = 1`**: genuine moves that persist on the following days, often on heavy volume.
- **Zero volume** (74 rows, mostly early history) is kept and flagged as possibly stale.

### Part 2: report template

- Uses **adjusted close** and daily **log returns**. Volatility is the 20-day sample std × √252. Drawdown is per-date, and **maximum drawdown** is its minimum. Sharpe is `(mean × 252 − 2%) / (std × √252)`, shown as `N/A` when it is undefined.
- Each metric function has a docstring and is checked against a tiny hand-computable example (e.g. 80 → 100 → 70 → 85 has a worst drawdown of −30% but a final drawdown of −15%).
- **Validation:** read-only SQL with placeholders. The report fails clearly on an unknown ticker, an empty window, or fewer than 21 prices (20 returns need 21 prices). It also re-checks the cleaning rules on the loaded slice.
- **Warm-up gaps are never filled.** The price line is gray until 20 returns exist, and the 50-day MA starts at the 50th price. If there are fewer, the chart says so.
- Results are exported with **scrapbook** as fractions (never formatted strings), with `None` for undefined values.

### Part 3: batch

- **Window:** the six complete calendar months *before* the month of the latest cleaned observation (2024-12-06 → **2024-06-01 to 2024-11-30**). This is anchored to the data, not the computer clock, and the same window is used for every ticker.
- Every ticker runs inside `try/except`, so one failure never stops the batch. Failures are written to `failed_stocks.txt` (tab-separated ticker and error), and the script prints `Successfully generated N/M reports`. If every ticker fails, it exits without building a summary.
- **Only reports that succeeded in this run** are collected. Scrapbook values are read from the exact output paths returned by papermill, so failed or stale notebooks in the folder are never picked up. A failed run still leaves its partially executed notebook on disk for debugging, but it is not collected.
- The script registers a temporary Jupyter kernel pointing at the interpreter that launched it. Reports then always run in the same environment, rather than whatever `python` is first on `PATH`.
- `portfolio_summary.csv` has the seven required columns in fractions, sorted by Sharpe. Stocks with an undefined Sharpe would be left out of the top/bottom-10 ranking. With 30 stocks the two lists don't overlap.

## Assumptions

- The supplied `ohlc` table is used **only** as a trading calendar and as a source of trusted date-specific adjustment factors. It is never used to replace rows wholesale.
- 252 trading days per year, a 2% annual risk-free rate, and the course's simple Sharpe approximation.
- **Unequal coverage.** In the 2024-06-01 to 2024-11-30 window, 22 of the 30 tickers have data only through **2024-10-18**. Only AMZN, CVX, DIS, IBM, KO, MRK, NKE and PG run to 2024-12-06. Each report prints its actual first/last dates and a coverage warning. The script lists the short-history stocks, and the scatter plot marks them with squares. Comparisons between the two groups cover different periods, roughly 4.5 vs. 6 months, so returns aren't directly comparable. No prices are extrapolated into the gap.
- Reading volatility colors against the report's own mean volatility is retrospective. It describes the window; it is not a trading signal.

## How to interpret the reports

Each report has four panels and a summary table:

- **Price and 50-day MA.** Red segments mean 20-day volatility was above the report's mean, green at or below it, and gray means the estimate isn't available yet. The price crossing its 50-day MA shows trend changes.
- **Daily returns.** Green bars are up days and red bars are down days. Clusters of tall bars are volatility clustering.
- **Rolling volatility.** Shaded red where it is above the mean, green where it is below.
- **Drawdown.** Distance below the running peak. The black dot is the maximum drawdown, the worst loss for someone who bought at the peak.
- **Summary table.** Total return over the observed window, the compounded annualized rate, whole-period volatility, Sharpe, maximum drawdown, best/worst day, share of positive days, and the average up/down day. The annualized return scales a short window up to a year, so read it as a rate, not a forecast.

**Portfolio results (2024-06-01 to 2024-11-30).** IBM, HD, SHW, WMT and MCD have the highest Sharpe ratios. MRK, BA and NKE are the only stocks with negative total returns. NVDA's mean volatility (about 61%) is roughly three times its peers', and its −27% drawdown came in August 2024. A six-month Sharpe ratio is noisy, so these rankings describe this period only and aren't a recommendation.
