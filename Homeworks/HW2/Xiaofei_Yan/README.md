# HW2: Automated Stock Analysis

## Submission files

This folder contains the data-cleaning notebook, the parameterized single-stock report template, and the Papermill batch runner. The local database belongs in `data/data.db`; it is provided through bCourses and must not be committed.

## Environment and dependencies

The notebooks and script use Python 3.12.9. From this folder, create and install into a homework-specific environment:

```bash
uv venv .venv --python 3.12.9
source .venv/bin/activate
uv pip install -r requirements.txt
```

On Windows PowerShell, activate with:

```powershell
.venv\Scripts\Activate.ps1
```

Keep `.venv/` and `data/data.db` local. Do not use the course-wide environment for this homework.

## Run order

1. Download `data.db` from bCourses and place it at `data/data.db`.
2. Activate the HW2 environment and start Jupyter from this folder.
3. Open `01_data_cleaning.ipynb` and run its cells from top to bottom. It reads `ohlc_hw`, performs the checks and repairs, validates the retained rows, and recreates `ohlc_hw_cleaned` in the local database. The raw `ohlc_hw` table is left unchanged.
4. Open `02_stock_report_template.ipynb`. Its top cell is tagged `parameters`. Set `ticker`, `start_date`, and `end_date` as needed, then run the notebook from top to bottom. The report uses adjusted close for log returns and drawdown.
5. From this folder, run the all-stock batch:

   ```bash
   python 03_run_all_stocks.py
   ```

   The script reads tickers and the latest observation date from `ohlc_hw_cleaned`. It analyzes the six complete calendar months before the month containing that latest date and creates `reports/YYYY-MM-DD/`, where the folder date is the batch run date. It writes one notebook per ticker, `failed_stocks.txt`, `portfolio_summary.csv`, and `portfolio_summary.png` when at least one report succeeds.

The Part 1 notebook must be rerun whenever the cleaned table needs to be recreated. The batch script runs the Part 2 template independently for each ticker and collects metrics only from reports that completed successfully in that run.

## Data cleaning decisions

- Ticker labels are normalized before duplicate checks. Exact duplicate records are removed; conflicting `(ticker, ts)` keys are not arbitrarily resolved.
- Missing raw prices are forward-filled within ticker after sorting by ticker and date. Rows with all four raw OHLC prices missing are removed before filling. Missing volume is filled with zero and marked as unknown; that repair is not evidence that no trading occurred.
- For a matched `(ticker, ts)`, the `ohlc` table provides a date-specific adjusted-price factor. Missing or inconsistent adjusted closes are rebuilt as validated raw close times that factor, and repair flags are saved. The factor is never forward-filled across dates. Unresolved rows are quarantined.
- For impossible OHLC relationships, the offending field is replaced with `NaN` and forward-filled within the same ticker (Option 1). Rows that remain invalid after repair are quarantined.
- The three-standard-deviation return screen uses simple raw-close returns and sample standard deviation. These flags are retrospective review signals, not automatic deletion rules. The database does not provide enough event or vendor evidence to label every extreme return as an error, split, merger, or news event, so those returns are retained and flagged rather than interpolated without evidence.
- Retained raw and adjusted prices, OHLC bounds, keys, volume, and matched adjustment factors are validated before `ohlc_hw_cleaned` is written.

## Assumptions and coverage

- For missing-trading-day checks, the supplied `ohlc` dates for each ticker are the reference calendar over that ticker's supplied history. Missing rows are reported; prices are not invented for absent dates.
- The matched `ohlc` adjustment factor is treated as the trusted, date-specific convention only where a valid ticker/date match exists. No factor is carried across an unmatched date.
- Part 3 uses one requested six-month calendar window for all tickers, based on the latest cleaned observation. Individual ticker histories may start later or end earlier. Each report states its actual first and last observation dates; metrics for shorter histories use only available observations and are not full-window comparisons. No trailing prices are extrapolated.

## Interpreting the reports

- Daily returns are log returns from adjusted close. Percentage formatting is for display; metrics in the portfolio CSV remain fractions.
- Rolling volatility is annualized from 20 daily log returns and remains unavailable during its warm-up period. The report's mean rolling volatility is not the same as whole-period annualized volatility in the summary table.
- Drawdown measures the decline from the running adjusted-price peak. It can recover toward zero; maximum drawdown is the most negative point in that series.
- Sharpe ratio uses the course approximation with a 2% annual risk-free rate. Undefined Sharpe values are omitted from rankings.
- The portfolio scatter plot compares mean rolling volatility with total return and colors defined Sharpe values. Stocks with shorter observation coverage should be interpreted with that limitation in mind.

## Sample outputs and submission

The script generates the full batch locally. For submission, include only 2–3 example report notebooks from one run folder and that folder's `portfolio_summary.csv`. Do not stage every generated notebook, `data.db`, another database copy, credentials, or the virtual environment. Keep the report path in the CSV relative to this homework folder.
