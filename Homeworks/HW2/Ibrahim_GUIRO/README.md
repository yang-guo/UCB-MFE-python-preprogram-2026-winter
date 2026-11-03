# HW2 — Market Data Cleaning and Stock Risk Reports

Author: Ibrahim GUIRO

## Files

- `01_data_cleaning.ipynb`: data inspection, cleaning, audit report,
  validation, and database export.
- `02_stock_report_template.ipynb`: parameterized single-stock risk report.
- `03_run_all_stocks.py`: batch execution and cross-stock comparison.
- `requirements.txt`: Python dependencies and installed versions.

## Setup

Use Python 3.12 and run the following commands from this directory:

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
```

Place a local copy of the supplied database at `data/data.db`.
The database is excluded from version control.

Select the `.venv` interpreter as the notebook kernel. The batch script
runs the default `python3` kernel of the environment it is launched from,
so start it with the virtual environment activated.

## Execution Order

1. Run all cells in `01_data_cleaning.ipynb`.
2. Run `02_stock_report_template.ipynb` to inspect a single-stock report.
   Its input cell is tagged `parameters`.
3. Save the template, then execute:

```bash
python 03_run_all_stocks.py
```

The batch uses the six complete calendar months preceding the month
containing the latest date in the cleaned database.

Outputs are written to `reports/YYYY-MM-DD/`:

- One executed notebook per stock.
- `portfolio_summary.csv`: metrics, actual coverage, and report paths,
  sorted by Sharpe ratio.
- `failed_stocks.txt`: tickers that failed during the current execution.
- `execution.log`: execution details and error tracebacks.
- `risk_return_profile.png`: risk-return comparison.

The terminal also displays the top and bottom ten Sharpe ratios.
A failed stock does not prevent the remaining reports from running.

Only three sample reports are committed (AAPL, MRK, TRV) together with
`portfolio_summary.csv`. The full batch is generated locally. AAPL ends
on 2024-10-18, MRK covers the whole window and has the lowest Sharpe
ratio, and TRV is the ticker with quarantined observations.

### Collecting results without Scrapbook

Instead of Scrapbook, the template ends with one named JSON output
(`output_name = "stock_report_results"`). The script reads that output
from the notebook returned by `papermill.execute_notebook` during the
current run, checks that the ticker and dates match the parameters, and
only then adds it to the summary. Old or failed notebooks left in the
folder are never read.

## Cleaning Decisions

Ticker labels are normalized before duplicate detection. Exact repeated
records are removed while retaining one observation per ticker and date.

Missing adjusted closing prices are reconstructed using date-specific
reference adjustment factors after validating the reference prices and
checking that raw closing prices match. Existing adjusted-price
discrepancies are corrected using the same reference convention.
Original values and separate repair flags are preserved.

Five observations with unresolved nonpositive opening prices and three
TRV observations with unexplained price spikes and zero reported volume
are quarantined. These exclusions are documented rather than treated as
confirmed market events.

Zero reported volume is flagged; it is not sufficient on its own to
justify removing an observation.

Impossible price relationships use Option 2: an open or close outside
`[low, high]` is replaced with the same ticker's previous close, the
bounds are checked again, and `adj_close` is rebuilt with that day's
adjustment factor. Rows that still fail (no previous day, `low > high`,
or a previous close outside today's range) are quarantined. The function
is demonstrated on an injected example because the retained data has no
remaining case; the only real ones were the first-day zero opens.

Missing trading days are detected against the supplied `ohlc` dates for
each ticker, including its first and last reference date, so a truncated
history would also be reported. The comparison is made on (ticker, ts)
keys. Before quarantine, no reference key is absent from `ohlc_hw`, and no
`ohlc_hw` key is outside the reference: zero missing days. After cleaning,
exactly the 8 quarantined keys are absent, by design; the notebook checks
that the missing set equals the quarantined set. The detector is tested on
a toy history with a date removed in the middle and at the start.

Missing raw OHLC values are handled by forward filling within each
ticker, after removing rows with all four raw prices missing. Missing
volume is filled with zero and flagged as imputed. This step required
no additional changes to the retained dataset.

Extreme raw returns are flagged using a ticker-specific, two-sided
three-standard-deviation rule (sample standard deviation). A flag alone
does not trigger correction or deletion. The largest flags were checked
one by one:

- TRV (3 days): one-day doubling with zero volume and immediate reversal,
  same values in the reference → quarantined.
- MCD 1968-01-02 (−75%): price drops to about 1/4 and stays there while
  volume rises about tenfold. An unadjusted split is one possible
  explanation, but it is an unconfirmed hypothesis (a vendor error is also
  possible), so no correction is applied: the row is kept and flagged.
- AAPL 2000-09-29 (−52%): the drop persists and volume is about three
  times higher, consistent with the September 2000 profit warning (news
  not verifiable from this dataset). Treated as a likely real move, kept
  and flagged.

No return was interpolated, because none was a confirmed error with
valid neighbours on both sides.

## Validation and Audit

The source `ohlc_hw` table is preserved. Results are saved separately as:

- `ohlc_hw_cleaned`
- `ohlc_hw_quarantined`
- `ohlc_hw_quality_report`

Final row reconciliation:

397,711 original rows = 29,460 removed duplicates
+ 8 quarantined observations + 368,243 retained observations.

Validation checks cover unique ticker-date keys, valid dates and tickers,
finite positive prices, consistent OHLC bounds, and finite nonnegative
volume. Saved prices are compared with the in-memory results.

Returns following excluded sessions are left undefined to avoid treating
a return across multiple sessions as a daily return.

## How to Interpret the Reports

Each report has four panels:

- **Price and 50-day MA**: red segments mark days when the 20-day
  volatility is above its average for the report, green below, gray
  before 20 returns are available. The moving average starts after 50
  prices.
- **Daily returns**: green and red bars for up and down days. Clusters
  of tall bars show stressed periods.
- **Rolling volatility**: shaded red when above the report average.
- **Drawdown**: how far the price is below its previous peak; the dot is
  the maximum drawdown.

The summary table gives total and annualized return, full-period
volatility, Sharpe ratio, maximum drawdown and the best/worst days. A
Sharpe ratio above 1 means the excess return was large relative to the
volatility over this short window; with about six months of data it is
noisy and should not be read as a forecast.

In `portfolio_summary.csv`, compare stocks with the same `actual_end`
first: 22 tickers stop on 2024-10-18 and 8 continue to 2024-11-29, so
their returns cover different lengths of time.

## Metric Conventions and Limitations

Risk and performance calculations use adjusted closing prices.
Daily returns are logarithmic. Volatility uses sample standard deviation
and 252 trading sessions per year. The annual risk-free rate used in the
Sharpe approximation is 2%.

Maximum drawdown is measured relative to the running adjusted-price peak.
Rolling volatility uses 20 returns; the moving average uses 50 prices.
Insufficient-history estimates remain undefined.

The CSV stores returns and volatility as decimal fractions.
`avg_volatility` is the mean annualized rolling volatility, which differs
from full-period annualized volatility.

Reports disclose actual coverage. Stocks can have different available
end dates, so the cross-stock comparison may cover unequal periods.
Annualized returns are calculated statistics, not forecasts.

Agreement with the supplied reference establishes consistency with that
source; it does not independently prove economic accuracy.