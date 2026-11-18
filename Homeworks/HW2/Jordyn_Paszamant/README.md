# Stock reports from SQLite

**SHUBHAM GUPTA**

This project cleans the course stock history, runs a report for each ticker,
and collects the results in a CSV and a risk-return chart. The reporting
window comes from the latest date in the cleaned data.

## Run the project

Use Python 3.12 or another compatible Python 3 environment. From this folder:

```bash
python -m pip install -r requirements.txt
python -m jupyter kernelspec list
```

If there is no `python3` kernel, register this environment:

```bash
python -m ipykernel install --user --name python3 --display-name "Python 3"
```

Place the supplied database at `data/data.db`. The database is excluded
from the submission. Open `01_data_cleaning.ipynb` and run all its cells.
Alternatively, execute it from the project root:

```bash
python -m papermill 01_data_cleaning.ipynb 01_data_cleaning.executed.ipynb -k python3
python 03_run_all_stocks.py
```

In a notebook opened from this directory, `%run 03_run_all_stocks.py` runs
the same batch. Change `db_path` and `kernel_name` near the top of the script
when using a different database location or kernel. The template's first code
cell is tagged `parameters` and takes `ticker`, `start_date`, `end_date`
and `db_path`.

## Cleaning decisions

I read `ohlc_hw`, which is the homework table. The other source tables are
kept intact. Tickers are stripped of HTML tags, trimmed, and uppercased. Dates
and numeric fields are parsed before validation. The exclusions below are
sequential, so each removed row is counted once.

| Stage | Rows removed | Rows remaining |
| --- | ---: | ---: |
| Original homework records | 0 | 397,711 |
| Invalid date or ticker | 0 | 397,711 |
| Weekend observations | 413 | 397,298 |
| Missing or nonfinite numbers | 19,836 | 377,462 |
| Nonpositive price or invalid volume | 5 | 377,457 |
| Inconsistent OHLC bounds | 0 | 377,457 |
| Identical duplicate quotes | 27,952 | 349,505 |
| Conflicting duplicate ticker/date keys | 0 | 349,505 |
| Extreme adjustment-factor deviations | 10,407 | 339,098 |

HTML normalization reduces 150 ticker strings to 30 stock symbols.
The adjustment screen compares `adj_close / close` with its centered
21-observation median, requiring at least five observations. Values below
one-third or above three times that median are excluded. For example, the
raw UNH record for June 3, 2024 had a close of 497.44 and an adjusted close
of 10,442.51, which would create a misleading return spike.

This screen is a heuristic. It can exclude genuine adjustment changes and
uses later neighbors, so the cleaned data is for retrospective analysis.
I preserved ordinary large price moves when the adjustment factor is
consistent. Missing adjusted prices are excluded rather than replaced with
raw closes, and no observations are filled. Zero volume is permitted.
An exchange holiday calendar and verified corporate-action feed would
improve the checks.

The notebook creates a staging table and replaces `ohlc_hw_cleaned` in a
transaction, with a unique ticker/date index. Re-running it refreshes the
cleaned table from the source. `cleaning_audit.json` stores the counts.

## Return and risk conventions

All return calculations use adjusted close, with no separate dividend credit.
The date grid is the union of cleaned observation dates. A return requires
valid prices on adjacent grid dates, which avoids calling a multi-session
gap a daily return. There is no forward filling.

- Daily return is `log(price / previous_price)`.
- Rolling volatility is the sample standard deviation of 20 consecutive valid
  daily log returns multiplied by `sqrt(252)`. Prices before the reporting
  window supply the warm-up. An incomplete window remains missing.
- Annualized volatility in the ten-row table uses all valid in-window daily
  log returns. `avg_volatility` in the CSV is the mean rolling volatility.
- Sharpe is `sqrt(252) * (mean_daily_log_return - log(1.02)/252) / std_daily_log_return`.
  It is undefined when daily standard deviation is zero.
- Total return is the last observed adjusted close divided by the first,
  minus one. Annualized return compounds this change over the number of
  shared-grid intervals between those observations, using 252 per year.
- Drawdown is adjusted close divided by its in-window running peak, minus
  one. Maximum drawdown is its minimum.
- Best day, worst day, average up day and average down day use simple returns
  converted from the valid daily log returns. Positive days is their fraction
  above zero. Empty up/down groups are shown as undefined.

Dates with missing prices can leave few complete 20-day windows. Each report
shows how many daily returns and rolling volatility observations were usable.
Missing data can bias these risk estimates if the omissions are systematic.
Drawdowns also omit movements on dates without a quote.

## Batch outputs and findings

The latest cleaned observation is **December 6, 2024**, so the selected window
is **June 1 through November 30, 2024**. The batch run on **November 18, 2026**
completed **30 of 30 tickers**, with no failures.

Coverage differs across tickers. AAPL and HD end on October 18, and BA ends on
October 17 in this cleaned source. IBM and several others extend into December,
so they cover more of the requested six-month window. The reports show their
actual first/last dates. The cross-stock ranking therefore compares different
observed coverage within the requested window.

HD has the highest Sharpe in this run, about **3.15**, with a total return of
**27.29%**. BA has the lowest Sharpe, about **-1.70**, with a total return of
**-17.66%**. AAPL returned **21.25%** over its observed coverage. These figures
depend on the exclusions and missing observations described above.

`reports/2026-11-18/` contains the portfolio CSV, scatter plot, empty failure
list, batch log, and three representative executed notebooks for AAPL, IBM
and BA. AAPL illustrates a familiar large stock, IBM covers the full requested
window, and BA illustrates negative return and drawdown. The full batch
generates a notebook for every ticker. Only the three samples are packaged
to keep the submission compact; the CSV paths for the other tickers will
resolve after rerunning the batch.

The scatter uses average rolling volatility on the x-axis, total return on
the y-axis, and Sharpe as color. The five highest-return stocks are annotated.
AMGN has no complete 20-return window, so its mean rolling volatility is
undefined. It remains in the CSV and Sharpe rankings, but is omitted from
the scatter because it has no x-coordinate. The chart has 29 plotted stocks.
Undefined Sharpe values are excluded from the printed rankings and plotted
in grey when their risk/return coordinates exist.

The CSV is a comparison of individual stocks. No combined portfolio weights,
portfolio return series or correlation model are assumed.

## Checks

The run verified ticker/date uniqueness, finite positive prices and the
number of rows written to SQLite. I checked the six-month window calculation
at year boundaries and across February in a leap year, and independently
reconciled AAPL's CSV metrics to a fresh query of the cleaned table.
All 30 Scrapbook dictionaries matched the saved CSV. A controlled failure
in the first report still allowed the other 29 reports to be collected, and
recorded the failed ticker. The batch log includes the top and bottom ten
Sharpe rankings. The script
catches report errors independently, records failed symbols, and writes full
tracebacks to `batch_errors.log`; one failure does not stop later reports.

The code was written for this assignment using pandas, Matplotlib,
Papermill, Scrapbook and IPython. No external price source or API is required.
