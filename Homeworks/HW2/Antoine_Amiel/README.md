# HW2: Scalable Stock Analysis Pipeline (Antoine Amiel)

Three steps: clean the dirty `ohlc_hw` table, build a parameterized single-stock risk report, and run that report for every stock with papermill.

## How to run

From this folder, with the course environment (`uv sync --locked` at the repository root, `.venv` as the Jupyter kernel):

1. Copy the course database from bCourses to `data/data.db` in this folder. It is a local working copy; it is gitignored and never committed.
2. Run `01_data_cleaning.ipynb` top to bottom. It writes `ohlc_hw_cleaned` (plus quarantine, quality report and ticker audit tables) into `data/data.db`, and leaves `ohlc_hw` unchanged.
3. Optional: run `02_stock_report_template.ipynb` directly; its defaults are AAPL over 2024.
4. Run `python 03_run_all_stocks.py`. It writes one report per ticker, `portfolio_summary.csv`, `risk_return_profile.png`, `failed_stocks.txt` and `run.log` to `reports/<run date>/`. Use `--tickers AAPL MSFT` for a quick test on a subset.

Dependencies are listed in `requirements.txt`. Everything was tested with the course lockfile versions (pandas 2.3.3, papermill 2.6.0, scrapbook 0.5.0).

## What is submitted

| Path | Content |
|---|---|
| `01_data_cleaning.ipynb` | Diagnosis, Data Quality Report, cleaning and validation |
| `02_stock_report_template.ipynb` | Parameterized report (cell tagged `parameters`) |
| `03_run_all_stocks.py` | Batch runner and portfolio summary |
| `reports/2026-10-26/` | Three sample reports (AAPL, AMZN, MRK), `portfolio_summary.csv`, `risk_return_profile.png`, `failed_stocks.txt` |

The full batch (30 reports, 30/30 successful) was generated locally; only three samples are committed, so `report_path` in the CSV points to files that exist after a local run.

## Data cleaning: what was found and what was done

| Issue | Rows | Action |
|---|---|---|
| HTML markup in ticker labels (150 labels, 30 tickers) | 39,936 | Normalized before grouping (tags stripped, trimmed, upper case); original labels kept in `ohlc_hw_ticker_audit` |
| Exact duplicate records | 29,460 | Removed. No conflicting `(ticker, ts)` keys remained; conflicts would have been quarantined, not resolved by keeping the first row |
| Missing `adj_close` | 18,374 | Rebuilt (see below) |
| Corrupted `adj_close` (4x to 24x too large) | 10,417 | Detected through the adjustment factor, rebuilt |
| `open = 0` (below `low`) | 5 | First day of the ticker's history, nothing to carry forward: quarantined |
| One-day spikes on TRV (1976, 1978, 1981) | 3 | Confirmed bad prints, interpolated |
| MCD drop of 75.4% on 1968-01-02 | 1 | Suspected unadjusted split: the 375 earlier MCD rows are quarantined |
| Zero volume | 75 | Kept and flagged as possibly stale |
| Missing raw prices, missing volume, `high < close`, `low > close`, missing trading days | 0 | Detectors and repairs demonstrated on a small injected example |

Result: 397,711 raw rows = 29,460 duplicates removed + 380 quarantined + 367,871 retained. After cleaning, every validation check is zero: no missing, non-positive or non-finite prices, all OHLC bounds hold, unique keys, finite non-negative volume, no adjustment-factor jumps, no missing trading days.

## Key design decisions

- Diagnosis before repair. The Data Quality Report (`ticker | issue_type | issue_date | description | severity`, 103,328 rows) is built on the data before any repair, with one row per issue and per observation.
- Severity. High means the value would distort prices or returns if left in place. Medium means information is missing or doubtful but repairable. Low means cosmetic or review-only.
- Calendar. Missing trading days are checked against each ticker's dates in the supplied `ohlc` table, within its own history span. Business days would count exchange holidays as gaps.
- Adjusted close. The factor `adj_close / close` changes slowly with dividends; a factor more than 2x its local median (11 surrounding days) marks a corrupted value. Flagged rows are 4x to 24x, all others are below 1.53x, so the threshold is not fine-tuned. Missing or corrupted values are rebuilt as validated raw close times the date-specific factor from the matching `(ticker, ts)` row of `ohlc`, only where the raw close matches. Factors are never forward-filled across dates. For the 3 interpolated TRV closes, the factor is interpolated between the neighbors, which agree within 1%. The source of each value is stored in `adj_source` (`vendor`, `reference_factor`, `interpolated_factor`).
- Repair flags (`price_filled`, `price_interpolated`, `volume_filled`, `zero_volume`, `extreme_return_flag`, `adj_source`) are saved with the cleaned data; quarantined rows are kept in `ohlc_hw_quarantine` with their reason.
- Validation reruns after every repair step and is logged in the notebook.

## How outliers were handled, and why

The 3-standard-deviation screen on raw-close simple returns (sample std, both tails, per ticker) flags about 5,000 days. The raw close in this vendor feed is already split-adjusted (AAPL shows no jump at its 2020 4:1 split), so a large move is not assumed to be a split. Each flag is classified with explicit rules:

- Spike and reversal (move above 25% undone the next day, within 5%): the three TRV cases are flat bars at twice the price, one with zero volume. They are data errors, and the close is interpolated between the neighboring days.
- Suspected unadjusted split (move above 45%, not reversed, price ratio within 2% of a split ratio): only MCD on 1968-01-02, ratio 0.246, close to 1/4. The vendor's adjusted close shows the same jump, so the adjustment does not handle it. Without a corporate-action source this cannot be confirmed, and neither side can be rescaled defensibly. The earlier block is quarantined because the later block is the one continuous with current prices.
- Everything else is kept and flagged as a review flag, with no claim about its cause. When five or more tickers are flagged on the same day (for example 1987-10-19), the description records a market-wide move.

## Assumptions

- The supplied `ohlc` table is trusted as the trading calendar and as the source of adjustment factors for the rows that need one. It is not used to replace the dirty data wholesale: 339,101 adjusted closes remain the vendor's own.
- The batch window is the six calendar months before the month of the latest cleaned observation (2024-12-06), so June 1 to November 30, 2024, the same for every ticker. Output folders are named by run date, independent of the window.
- 22 of the 30 histories end on 2024-10-18, so their reports cover June 3 to October 18 instead of the full window. Every report prints its observed coverage, the CSV includes `first_date`, `last_date` and `trading_days`, and the scatter plot draws these stocks as triangles. Their metrics are not strictly comparable with full-window stocks, and nothing is extrapolated past the last observation.

## How to read a report

- Price panel: adjusted close, colored red when the 20-day volatility is above its mean over the report, green otherwise, gray during the 20-day warm-up; the dashed line is the 50-day moving average, unavailable before 50 prices. The coloring is retrospective.
- Daily returns: daily log returns, green up, red down.
- Rolling volatility: annualized 20-day volatility with its mean; red and green shading mark high- and low-volatility regimes.
- Drawdown: decline from the running peak; the black dot marks the maximum drawdown.
- Summary table: total return is last over first adjusted close minus 1; annualized return compounds the mean log return; annualized volatility uses the whole period (it differs from the mean rolling volatility used in the portfolio summary); the Sharpe ratio uses a 2% annual risk-free rate and shows N/A when undefined.
- Portfolio summary: sorted by Sharpe ratio; undefined Sharpe ratios are excluded from the ranking. Values are fractions (0.25 = 25%).
