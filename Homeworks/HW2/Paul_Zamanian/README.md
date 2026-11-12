# HW2: stock analysis pipeline

Paul Zamanian

Files:
- `01_data_cleaning.ipynb`: part 1, finds the problems in `ohlc_hw` and builds `ohlc_hw_cleaned`
- `02_stock_report_template.ipynb`: part 2, the report for one stock (the parameters cell is tagged `parameters`)
- `03_run_all_stocks.py`: part 3, runs the template for every ticker with papermill and makes the portfolio summary
- `reports/2026-10-29/`: three example reports (IBM has the best Sharpe, MRK the worst, AAPL has a shorter history) and `portfolio_summary.csv`

## How to run

From this folder:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
mkdir -p data
cp /path/to/data.db data/data.db     # the bCourses database, not committed

jupyter execute --inplace 01_data_cleaning.ipynb   # or open it and run all cells
python 03_run_all_stocks.py                        # writes reports/<today>/
```

To run one report on its own: `papermill 02_stock_report_template.ipynb out.ipynb -p ticker MSFT -p start_date 2024-01-01 -p end_date 2024-06-30`

The cleaning notebook only reads `ohlc_hw` and `ohlc`. It writes `ohlc_hw_cleaned`, `ohlc_hw_quarantine` (rows I couldn't fix, with the reason), `ohlc_hw_ticker_audit` and `ohlc_hw_quality_report` into `data/data.db`.

## What I found in the data and what I did

- Tickers: 150 different labels for 30 stocks, because some are wrapped in HTML. I remove the tags and spaces and upper-case them before anything else, and keep the original in `raw_ticker`.
- 29,460 exact duplicate rows, dropped (`n_source_rows` keeps the count). There are no conflicting duplicates; if there were, both rows would go to quarantine.
- 18,374 missing `adj_close` values and 10,417 that are 4x to 24x too high. I find the spikes by comparing `adj_close / close` with the median of the days around it. Both cases are rebuilt as `close * (adj_close / close)` with the factor from the same ticker and date in `ohlc`. The spikes I found are exactly the rows where `ohlc_hw` and `ohlc` disagree.
- 384 rows dated on a Sunday (16 tickers, early 1974 and early 1975). `ohlc` has the same 384 Sunday rows, so this comes from the source data, not from the dirty copy. Most of those weeks have no Friday, so it's probably Friday's data with the wrong date, but 52 of these rows have the Friday as well. I quarantined all of them instead of guessing. Keeping them with a flag would also have been reasonable; either way no report is affected, since they're all from 1974-75.
- 5 rows with `open = 0`. They are all the first row of a ticker, so there's nothing to forward-fill from. Quarantined.
- 3 one-day glitches in TRV where the price doubles on zero volume and comes back the next day. Interpolated from the days around them.
- 35 smaller `adj_close` glitches, mostly DIS in 1974-75 (the factor switches between 0.621 and 0.663) plus single days like AAPL on 1990-10-19 (`adj_close` equals `close`). `ohlc` has the same numbers, so I have nothing to fix them with. They're all before 1991, so I kept them and they're listed in the quality report.
- 75 zero-volume days (1962 to 1982). 72 of them have open = high = low = close and 65 repeat the previous close, so they look like stale quotes. Kept and flagged.
- No missing raw prices, no `high < close` or `low > close`, no missing volume and no missing days compared to `ohlc`. Those repairs are tested on a small fake dataset at the end of the notebook.

In total: 397,711 raw rows = 367,862 kept + 29,460 duplicates + 389 quarantined. All validation checks pass and `ohlc_hw` is unchanged.

## Design decisions

- Tickers are cleaned first, then duplicates are removed, then everything else. Anything over time is done sorted by date and within one ticker.
- `ohlc` is only used as a calendar and to look up the adjustment factor of a specific date. It never replaces a row.
- For impossible prices I used option 1 (set to NaN, then forward-fill). If only the close is outside `[low, high]`, the close is the wrong value, and the same for the open. If both are outside, or `high < low`, I can't tell which one is wrong, so the row goes to quarantine. Forward-filling alone can put a price outside the day's range, so a filled open/close is clipped into `[low, high]` and a filled high/low is widened.
- `adj_close` is never forward-filled and a factor is never carried from one day to another, except for the 3 TRV bars, where both neighbouring days have the same factor.
- Missing volume becomes 0 with a flag, because an unknown volume doesn't mean nothing traded.
- Every repair is flagged in the cleaned table (`ohlc_repaired`, `adj_close_repaired`, `volume_repaired`, `zero_volume`, `extreme_return`, plus text columns saying what was done), and the price checks run again after each step.
- Severity in the quality report: high for impossible or missing values in the price columns and impossible dates, medium for things that may be real or are about the structure (extreme returns, duplicates, missing days, volume), low for cosmetic things (ticker markup, zero volume).
- The batch script reads the results back with scrapbook, only from the reports that succeeded in that run, and writes failures to `failed_stocks.txt` and `run.log`.

## Outliers

The 3 std screen on daily close returns (per ticker, both sides) flags 5,060 days once the Sunday rows are removed. The flags are for review; nothing is deleted because of them. For each one I checked:

- Splits: they can't be the cause, because `close` is already split-adjusted in this data (AAPL's 2020 split and NVDA's 2024 split show no jump).
- Data errors: 6 flags, the 3 TRV glitches (the jump and the way back). `ohlc` has the same glitch, so the evidence is that the price comes straight back. Interpolated.
- Factor changes: 7 flags where the `adj_close / close` factor moves more than 5% on the same day. Six are the factor glitches above. The seventh is MSFT on 2004-11-15, which matches its $3 special dividend. Kept.
- The rest are treated as real market moves and kept with a flag. MCD on 1968-01-02 (-75%) looks like a split that was never adjusted for, but I can't confirm that from the data. It's far outside the report window anyway.

## Assumptions

- The expected trading days of a ticker are its dates in `ohlc` (weekdays alone would count holidays as gaps). Missing days are reported, never filled.
- 252 trading days a year and a 2% risk-free rate, as in the assignment. Returns and drawdown use the adjusted close; volatility, Sharpe and the daily stats use log returns.
- The batch window is the 6 calendar months before the month of the last date in the cleaned table. That date is 2024-12-06, so the window is 2024-06-01 to 2024-11-30 for every ticker. The output folder is named after the day the script runs.
- A report needs at least 21 prices (20 returns). With fewer it stops with a clear message and the batch moves on.

## Reading the reports

- Price panel: red where the 20-day volatility is above its mean for the period, green below, gray for the first 20 days before there is a volatility value. The mean uses the whole period, so the colors describe what happened; they're not a signal. The 50-day average starts on day 50.
- Returns and volatility: daily log returns, and the annualized 20-day volatility against its mean, shaded above and below.
- Drawdown: how far the price is below its highest point so far. The dot is the maximum drawdown.
- Summary table: total return from the first and last adjusted close, compounded annualized return, volatility over the whole period (not the average of the rolling one), and positive days out of all returns.
- `portfolio_summary.csv`: fractions, sorted by Sharpe, with `first_date`, `last_date` and `trading_days` added. 22 of the 30 tickers stop on 2024-10-18 (97 days) while 8 go to 2024-11-29 (126 days), so the ranking doesn't compare exactly the same period. Sharpe ratios and annualized returns over 4 to 6 months are also noisy, so I'd read them as a description of that period, not a forecast.
- `report_path` points to the full local batch; only the three example reports are committed.
