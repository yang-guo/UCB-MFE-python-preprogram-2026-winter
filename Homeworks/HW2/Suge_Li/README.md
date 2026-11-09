# Homework 2 — Stock analysis pipeline

Suge Li. Working directory: `Homeworks/HW2/Suge_Li`.

## How to run

This project uses [uv](https://docs.astral.sh/uv/). The dependencies are already installed in the repository-level `.venv`; `uv run` reuses it automatically, so no manual `pip install` is required.

1. Place a local copy of the course database at `Homeworks/HW2/Suge_Li/data/data.db` (download from bCourses). Do not commit it. If the course file `Lectures/Lecture 4/data/data.db` is present, `01_data_cleaning.ipynb` will copy it into `data/` on first run.

2. Open `Homeworks/HW2/Suge_Li/01_data_cleaning.ipynb` in Jupyter, select the kernel that points to the repository-level `.venv`, and **Run All**. This creates `ohlc_hw_cleaned` in `data/data.db`.

3. From the repository root, generate reports for every cleaned ticker:

   ```bash
   uv run python Homeworks/HW2/Suge_Li/03_run_all_stocks.py
   ```

   Steps 2 and 3 must use the same interpreter. `03_run_all_stocks.py` registers the papermill kernel with `sys.executable`, so the interpreter that runs the script is the one used inside every generated report. `uv run` keeps both steps on the repository-level `.venv`.

   Outputs go to `Homeworks/HW2/Suge_Li/reports/{YYYY-MM-DD}/`. The folder name is the **run date**. The analysis window is the six complete calendar months before the month of the latest cleaned `ts` (for a last observation on 2024-12-06 that is 2024-06-01 through 2024-11-30).

   To recreate the environment from scratch (only if needed):

   ```bash
   uv venv --python 3.12.9 .venv
   uv pip install --python .venv/bin/python -r Homeworks/HW2/Suge_Li/requirements.txt
   ```

## Design decisions

- Ticker labels in `ohlc_hw` include HTML markup. They are normalized (tags stripped, whitespace removed, uppercased) **before** grouping or deduplicating. `ticker_raw` is stored on the cleaned table as an audit trail.
- Exact duplicate rows are dropped. After normalization there were no conflicting `(ticker, ts)` values; a conflict would stop the pipeline rather than keeping the first row.
- Missing trading days are measured against the supplied `ohlc` table's dates per ticker (the reference calendar allowed by the assignment), over that ticker's supplied history. Any absent `ohlc_hw` date is reported as a medium-severity `missing_session` issue.
- Impossible OHLC (including open = 0) is repaired from the **same day's** remaining valid prices. Yesterday's close is not copied when it would fall outside today's high/low.
- Extreme simple close returns (`|r − mean| > 3 sd`, `ddof=1`, both tails) are **review flags**, not deletions. In this file the raw close matches the supplied `ohlc` table, so those days are retained.
- Missing or inconsistent `adj_close` is rebuilt as `close × (ohlc.adj_close / ohlc.close)` on the matched `(ticker, ts)`. That is a date-specific vendor factor, not a factor carried across an unverified corporate action. If close itself was repaired and that factor cannot be verified, the row is quarantined. Raw OHLC is never replaced by the reference table.
- Missing, non-finite, or negative volume is set to 0 and flagged `volume_unknown`. Observed zero volume is kept and flagged separately as `zero_volume`; it is not treated as evidence of no trading.
- Rows that still cannot be repaired are written to `ohlc_hw_quarantine` (an empty table is written when there are none) rather than borrowing another ticker's price.

## Interpreting the reports

- Returns, volatility, Sharpe, and up/down statistics use **daily log returns of adjusted close**.
- Total return is `last_adj / first_adj − 1`. Maximum drawdown is the worst per-date drawdown from a running peak (`drawdown.min()`).
- Rolling 20-day volatility is annualized with `sqrt(252)`. The first 20 return observations are left blank (not zero-filled). The 50-day average needs 50 prices.
- Sharpe uses the course approximation `(mean × 252 − 0.02) / (std × sqrt(252))`. Undefined Sharpe (zero vol or < 2 returns) is excluded from rankings.
- Tickers do not share the same last date. The batch requests one common window; each notebook prints its **actual** first and last observation. Do not treat a shorter history as a full-window return.

## Submitted sample outputs

`Homeworks/HW2/Suge_Li/reports/2026-10-29/` contains only:

- `AAPL_report.ipynb`, `MSFT_report.ipynb`, `NVDA_report.ipynb`
- `portfolio_summary.csv`


The local batch generated all 30 reports; the other 27 notebooks are intentionally not committed. CSV metrics are fractions, not percents. Do not commit `data/data.db`.

