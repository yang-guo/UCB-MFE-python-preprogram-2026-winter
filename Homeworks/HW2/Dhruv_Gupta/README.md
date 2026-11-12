# Homework 2: Clean the prices, then repeat the risk report

This folder cleans `ohlc_hw`, writes `ohlc_hw_cleaned` to a local working copy, and generates one risk report per ticker. The batch that produced the files in `reports/2026-10-29/` finished 30 of 30 reports. The latest cleaned observation is 2024-12-06, so every report was requested for the six complete months before that month: 2024-06-01 through 2024-11-30.

Run the three files in this order, from an interpreter that can import the packages in `requirements.txt`. Python 3.12 is the version that produced these outputs.

## How to run

Install the dependencies, then download `data.db` from bCourses into `Lectures/Lecture 4/data/data.db`. Leave that course file in place. This project reads it and writes a separate working copy.

```bash
python -m pip install -r Homeworks/HW2/Dhruv_Gupta/requirements.txt
```

1. Open `01_data_cleaning.ipynb` from inside the course repository and run all cells. The notebook finds the course database, copies it to `Homeworks/HW2/Dhruv_Gupta/data/data.db` when that working file is absent, and replaces only `ohlc_hw_cleaned` and `ohlc_hw_quarantine` in the copy. `ohlc_hw` in the course database stays as delivered. The working database is local and is not part of the submission. Running the notebook again recreates `ohlc_hw_cleaned`.

2. Open `02_stock_report_template.ipynb` and run all cells to check one stock. The first code cell is tagged `parameters`. Its defaults are `ticker = "AAPL"`, `start_date = "2024-01-01"`, `end_date = "2024-12-31"`, and `db_path = "data/data.db"`. A relative `db_path` resolves to this folder's working copy. The batch script later replaces these four values.

3. Generate the full set from this folder, with the same interpreter:

```bash
python 03_run_all_stocks.py
```

The script can also be launched from another working directory. It locates the template and the working database from its own file path. Papermill runs each report in that same interpreter, so a separate Jupyter kernel registration is unnecessary. Output goes to `reports/{YYYY-MM-DD}/`, where the folder name is the day the script runs. That date is independent of the historical window.

The script reads every distinct ticker from `ohlc_hw_cleaned` and passes each one the same `start_date` and `end_date`. A failure is written to `failed_stocks.txt` and `failed_stocks.json`, and the loop continues. If every ticker fails, the script says so and does not write a portfolio summary.

## Design decisions

Ticker text is normalized before any grouping, duplicate check, or return. HTML markup, whitespace, and case are removed from the symbol used as a key. `ticker_raw` keeps the delivered label. Rows are sorted by ticker and date before returns or fills.

Exact duplicate rows are removed. Conflicting `(ticker, ts)` keys are quarantined together. The first conflicting row is not kept by default.

Missing sessions are dates present in the supplied `ohlc` table inside that ticker's own first and last `ohlc_hw` date. A missing session is reported and is left without a price. Business-day calendars are not used, because they count exchange holidays as gaps. On this delivery the gap count is zero, and the detector is also shown on a four-date example with January 3 removed.

A row whose open, high, low, and close are all missing is dropped before any fill. A remaining missing price is repaired inside one ticker, in date order. When today's low and high form a valid range, the previous close is used if it sits inside that range. Otherwise the midpoint of today's low and high is used. A missing price with no usable range is forward-filled from the last valid price of the same ticker. A fill that is nonpositive or that breaks the bar is discarded. A leading gap with no earlier price is quarantined. No value is taken from a later date or from another ticker.

Missing, negative, or non-finite volume becomes 0 and `volume_repaired` is set. That 0 means the volume is unknown. An observed 0 stays 0, `zero_volume_flag` is set, and `volume_repaired` stays false.

Adjusted close uses the same-day factor `ohlc.adj_close / ohlc.close` when both reference prices are finite and positive. The repaired value is the validated raw close times that factor. The factor is taken from that date only. Raw open, high, low, close, and volume are not copied from `ohlc`.

Dates in the cleaned table are `YYYY-MM-DD`. Repair flags travel with the retained rows. After repair, retained prices are finite and positive, `(ticker, ts)` keys are unique, volume is finite and nonnegative, and raw bounds hold. This run retained 368,251 rows and quarantined none.

Report returns, volatility, drawdown, and Sharpe use adjusted close. Part 1's simple returns use raw close and exist only as a data-quality screen. The batch collects named Scrapbook outputs from notebooks whose execution returned. It does not scan the reports folder, so a partial or older notebook cannot enter `portfolio_summary.csv`.

## How outliers were handled

The statistical screen is `abs(return - mean) > 3 * std` inside each ticker, using the sample standard deviation (`ddof=1`) and both tails. The return is the simple raw-close return. Undefined returns stay out of the mean, the standard deviation, and the flags. A ticker whose standard deviation is zero or undefined gets no statistical flags and a low-severity note.

That screen is a review flag. It is not a deletion rule. A close is interpolated only when the error is confirmed and the point sits between two valid neighbors of the same ticker. On this delivery the extreme closes match `ohlc`. The large raw moves do not come with a small adjusted move, so none is treated as a split. The threshold for that split check is an absolute raw return above 40% together with an absolute adjusted return below 20%. Three large zero-volume reversals are kept, and `extreme_return_note` records that no filing or news source was available. The interpolation function was checked on a small example and was not applied to these closes.

Impossible bars use the previous close when that close is positive and inside the prices that are still valid. Otherwise they use the other prices on the same bar: the midpoint of low and high, or the open/close span when high or low is the broken side. Unconditional forward-fill is avoided because yesterday's close can fall outside today's range.

On this file the only broken bars are five opens recorded as 0, on the first session of CVX, JNJ, MCD, MRK, and PG. High, low, and close on those rows are already a valid bar, and there is no previous close. The supplied `ohlc` table has the same open of 0. The open is set to the midpoint of that day's low and high and flagged `open_repaired`.

Adjusted-close factors that differ from the same-day reference factor by more than 50% are rebuilt from the validated raw close and that same-day factor. In this file those damaged factors are several times the reference factor, and they are not the previous or next reference factor. A smaller disagreement would be quarantined rather than overwritten.

Severity in the quality report follows the damage. A missing raw price, an impossible bar, a nonpositive or non-finite price, negative volume, or a conflicting key is high. A missing adjusted close, missing volume, missing session, extreme return, or zero volume is medium. An exact duplicate, or a zero or undefined return standard deviation, is low.

## Assumptions

- The course `ohlc` table can supply each ticker's session calendar and a same-day adjustment factor. It is not a replacement for the dirty `ohlc_hw` rows.
- A gap count of zero is a valid result when every `ohlc` date inside a ticker's delivered span is present.
- A 3σ move that also appears in `ohlc`, without the raw-versus-adjusted signature of a split, stays in the cleaned table.
- The annual risk-free rate in the Sharpe ratio is 2%. Annualization uses 252 trading days. The ratio is `(mean daily log return * 252 - 0.02) / (std daily log return * sqrt(252))`. Fewer than two finite returns, or zero volatility, makes the ratio undefined.
- The batch window is the six calendar months before the month that contains the latest cleaned observation, including when that observation falls on a month-end. December 6, 2024 selects June 1 through November 30, 2024.
- Every ticker receives that same requested window. A history that ends earlier keeps its last cleaned price. Later dates stay empty.

## How to read a report

The header states the requested window and the number of trading days returned, then the first and last dates actually loaded. Those two dates are the coverage. The requested end can fall on a weekend. November 30, 2024 is a Saturday, so a report that reaches the last session in the window ends on 2024-11-29.

In this batch the observed histories split into two groups:

- 22 reports cover 2024-06-03 through 2024-10-18, 97 trading days: AAPL, AMGN, AXP, BA, CAT, CRM, CSCO, GS, HD, HON, JNJ, JPM, MCD, MMM, MSFT, NVDA, SHW, TRV, UNH, V, VZ, WMT.
- 8 reports cover 2024-06-03 through 2024-11-29, 126 trading days: AMZN, CVX, DIS, IBM, KO, MRK, NKE, PG.

Return, volatility, drawdown, and Sharpe for the October group describe that shorter sample. Placing one of those names next to a name that continues through 2024-11-29 compares two different sets of sessions. The price series is not extended through the trailing gap.

The four panels are a 16 by 12 inch figure titled `{Ticker} Risk Analysis Report - {start} to {end}`:

- Adjusted close and the 50-day moving average. The line is green at or below the report's mean rolling volatility and red above it. Gray marks the warm-up, before 20 returns exist. The moving average itself starts once 50 prices exist.
- Daily log returns, green when positive and red when negative.
- 20-day annualized rolling volatility, with the mean drawn and the two sides shaded. The first 20 dates stay empty.
- Per-date drawdown from the running peak. The marked point is the minimum of that series, which is the maximum drawdown. Drawdown can recover after that date.

The summary table prints percentages. The stored values are fractions, except the Sharpe ratio, which is the course ratio. Total return is the last adjusted close divided by the first, minus one. Annualized return compounds the nonmissing log returns: `exp(sum * 252 / n) - 1`. Annualized volatility in the table is `std(log returns) * sqrt(252)` for the whole window. That is separate from `avg_volatility` in the portfolio file, which is the mean of the 20-day rolling volatility. Best day, worst day, and the up and down averages use log returns. Positive days divide the strictly positive returns by every nonmissing return, including zero days. An empty up or down subset prints `N/A`.

`portfolio_summary.csv` has one row per successful report: `ticker`, `avg_daily_return`, `avg_volatility`, `max_drawdown`, `sharpe_ratio`, `total_return`, and `report_path`. It is sorted by Sharpe ratio, descending. Undefined Sharpe ratios stay in the file and stay out of the top and bottom rankings. The scatter plot uses average rolling volatility on the x-axis, total return on the y-axis, and Sharpe ratio as the color. The five highest total returns are labeled. The chart title is `Risk-Return Profile of Portfolio`. On this run IBM has the highest Sharpe ratio and MRK the lowest.

The full local batch is in `reports/2026-10-29/`, with `portfolio_summary.csv` and `portfolio_risk_return.png`. The submission keeps two or three of those notebooks plus the summary CSV. AAPL, IBM, and MRK show the short history, the longer history, and the weakest Sharpe ratio.
