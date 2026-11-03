# HW2 - Stock Analysis Pipeline (Louis Maurice)

## How to run
From the repo root: `uv sync && source .venv/bin/activate` (or `pip install -r requirements.txt`).
Put `data.db` from bCourses in `data/data.db` inside this folder (gitignored), then from this folder:

```bash
papermill 01_data_cleaning.ipynb 01_data_cleaning.ipynb   # writes ohlc_hw_cleaned
python 03_run_all_stocks.py                                # runs 02_ for every ticker
```
`02_stock_report_template.ipynb` can also be run alone for one stock (its `parameters` cell is tagged).

## Data issues found and how they were handled
| Issue | Count | Handling |
|---|---|---|
| Tickers wrapped in HTML tags (150 names for 30 stocks) | 39,936 rows | Strip tags |
| Exact duplicate ticker/date rows | 29,460 | Drop |
| Missing `adj_close` | 18,374 (after dedup) | Rebuilt from close × interpolated adjustment factor |
| Corrupted `adj_close` (one-day spikes, raw OHLC fine) | ~10,400 | Same as above |
| `open = 0` (first row of CVX, JNJ, MCD, MRK, PG) | 5 | Open set to that day's close, clipped into [low, high] |
| Zero volume | 75 | Kept, flagged |
| Missing trading days (1972-1989 only) | 445 | Not filled; outside any report window |

## Key design decisions
- **Outliers**: almost every >3σ return was in `adj_close` only, not in the raw OHLC. So the error is in the adjustment factor `adj_close / close`, not in the price. A factor more than 10% away from its 11-day rolling median is treated as a data error. It is interpolated within the ticker, and `adj_close = close × factor`. This keeps one consistent adjustment convention, and real splits and dividends are kept because they change the factor for good, not for one day.
- **Large returns left after the repair** are treated as real market moves. They are kept and flagged (`extreme_return`).
- Repair flags (`open_repaired`, `adj_repaired`, `extreme_return`, `zero_volume`) are saved in `ohlc_hw_cleaned`.
- Validation: there are no NaN prices, all OHLC bounds hold, and 99.99% of the cleaned `adj_close` is within 1% of the course's clean `ohlc` table.

## Assumptions
- Report window = last 6 complete months before the latest date in the database (2024-12-06 → 2024-06-01 to 2024-11-30).
- Most tickers stop on 2024-10-18, so their reports cover June 1 to Oct 18 only. Only tickers with data through December cover the full window.
- Returns are log returns, with 252 trading days a year and a 2% risk-free rate for Sharpe.

## Reading the reports
Each `reports/2024-11-30/{TICKER}_report.ipynb` shows four charts:
- price, coloured by volatility (red = high), with its 50-day average
- daily returns
- 20-day annualized volatility against its mean
- drawdown from the running peak, with the worst point marked

A summary table follows. `portfolio_summary.csv` ranks all stocks by Sharpe ratio, and `risk_return.png` plots volatility against total return.
