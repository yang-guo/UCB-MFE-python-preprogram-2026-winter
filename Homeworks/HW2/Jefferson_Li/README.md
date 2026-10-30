# HW2 — Jefferson Li

Two notebooks and one script implement the stock-analysis pipeline.
[Course assignment](../../../Lectures/Lecture%202/hw2.md)

## Run

From this folder, using Python 3.12.9:

```sh
source .venv/bin/activate
jupyter lab
```

Select `.venv/bin/python` as the notebook kernel. Run `01_data_cleaning.ipynb`
from top to bottom, then run `02_stock_report_template.ipynb` for one stock.
Generate all stock reports with:

```sh
python 03_run_all_stocks.py
```

To recreate the environment:

```sh
uv venv --python 3.12.9 .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

Download the supplied database from Canvas and place a working copy at
`data/data.db`. The cleaning notebook recreates `ohlc_hw_cleaned`; it never
changes the raw `ohlc_hw` or reference `ohlc` tables. The database is not submitted.

## Part 1: cleaning decisions

- Normalize HTML/whitespace/case in ticker labels and retain original labels.
  Remove exact duplicates; quarantine conflicting stock/date records.
- Diagnose before repairs. Drop rows with all raw prices missing before filling.
  Forward-fill missing raw prices within a sorted ticker, then recheck bounds.
  Quarantine unresolved leading gaps and invalid rows. Missing/invalid volume
  becomes zero with an unknown-volume repair flag.
- Use matched stock/date factors (`ohlc.adj_close / ohlc.close`) to reconstruct
  missing or inconsistent adjusted close. Never forward-fill adjustment factors.
  This preserves the dirty dataset's raw observations rather than replacing it.
- Simple raw-close returns beyond three sample standard deviations are review
  flags. In this data, the raw-close pairs agree with the supplied reference;
  they are retained with the cause explicitly unverified. No split/news claim
  is inferred. Only explicitly confirmed errors use interior interpolation.
- Compare missing trading dates against each ticker's own supplied `ohlc`
  history. Quarantined observations remain documented gaps, not invented prices.

The run starts with 397,711 rows, removes 29,460 exact repeats, quarantines five
unresolved zero opening prices, and retains 368,246 rows. It repairs 28,791
adjusted-close values. All retained prices are finite and positive, keys unique,
volumes finite/nonnegative, and raw OHLC bounds valid. Repair flags are saved.
Quality reports, quarantine records, and row accounting are in the local `data/`
folder and audit tables. Synthetic examples verify missing cases and repairs.

## Parts 2–3: interpretation and outputs

The report uses adjusted close, daily log returns, 20-return rolling annualized
volatility, a 50-price moving average, drawdown, and a ten-metric summary.
Annualization uses 252 trading days; Sharpe uses the course 2% annual risk-free
approximation. Rolling warm-up gaps remain missing; unavailable statistics
show N/A. Maximum drawdown is the minimum of the per-date drawdown series.

The batch uses the six calendar months before the latest observation's month.
With a latest observation of 2024-12-06, it requests 2024-06-01 to 2024-11-30.
The run-date folder is separate from that historical window. Each stock has its
own actual coverage, which is recorded in the report and CSV. Shorter histories
are not extrapolated, so comparisons should consider these different periods.

`reports/YYYY-MM-DD/` contains all executed reports, `portfolio_summary.csv`,
`risk_return_scatter.png`, `failed_stocks.txt`, and a short `run_info.json`.
Only successful reports from the current run enter the summary. Failures are
logged without stopping later stocks; an all-failed run has an empty CSV and
no misleading chart. Undefined Sharpes are excluded from rankings.

CSV returns/volatility/drawdown values are fractions, not percentage strings.
`avg_volatility` is the mean rolling volatility, distinct from the report's
whole-period annualized volatility. The scatter plots this rolling-volatility
mean against total return and colors points by Sharpe.

Verified run on 2026-10-29: all 30 reports succeeded, with no failed tickers.

The full batch is kept locally. Git includes only AAPL, IBM, and MRK sample
notebooks plus `portfolio_summary.csv`, as requested by the assignment. Other
run outputs stay local.
