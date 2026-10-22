# HW2 - Stock Analysis Pipeline

Farshid Gilak

This folder has my three parts of HW2: cleaning the `ohlc_hw` table, a parameterized single-stock report, and a script that runs the report for every stock and builds a portfolio summary.

```text
Farshid_Gilak/
├── 01_data_cleaning.ipynb          Part 1: diagnose + clean ohlc_hw, save ohlc_hw_cleaned
├── 02_stock_report_template.ipynb  Part 2: one-stock risk report (papermill template)
├── 03_run_all_stocks.py            Part 3: runs the template for all stocks + summary
├── requirements.txt
├── data/data.db                    local copy of the course database (gitignored, not submitted)
└── reports/2026-10-22/
    ├── IBM_report.ipynb            sample reports (3 of the 30)
    ├── MRK_report.ipynb
    ├── AAPL_report.ipynb
    ├── portfolio_summary.csv
    └── risk_return_profile.png
```

## How to run

1. Put the bCourses `data.db` in `Lectures/Lecture 4/data/` (the notebook also looks in `Lectures/Lecture 2/data/`, which is where I had mine). The first time you run Part 1 it copies the database into `data/data.db` in this folder and only ever writes to that copy, so the course file stays untouched.
2. Install the packages. From the repo root, `uv sync --locked` already covers everything. Otherwise:

   ```bash
   pip install -r requirements.txt
   ```

3. Run `01_data_cleaning.ipynb` top to bottom (takes about 15 seconds). It creates `ohlc_hw_cleaned` in `data/data.db`, plus a few extra tables I used for checking (`ohlc_hw_quality_report`, `ohlc_hw_outlier_review`, `ohlc_hw_quarantine`).
4. If you want to look at one stock, open `02_stock_report_template.ipynb`, change the cell tagged `parameters` (ticker, start_date, end_date) and run it.
5. Run the batch from this folder:

   ```bash
   python 03_run_all_stocks.py
   ```

   It takes about a minute for all 30 stocks and writes into `reports/<today's date>/`. You can also run a subset with `--tickers AAPL MSFT`, or see the other options with `--help`.

The order matters: Part 2 and Part 3 read `ohlc_hw_cleaned`, so Part 1 has to run first.

One thing I ran into: my default Jupyter `python3` kernel pointed at an old virtual environment, so papermill kept failing to start. The script now makes a temporary kernel that uses whatever Python you run the script with, so you shouldn't need to set up a kernel yourself.

## What I found in the data

Before fixing anything, I ran all the checks and kept the results in a Data Quality Report (`ticker | issue_type | issue_date | description | severity`). The main findings:

- **Ticker labels.** There were 150 different labels for 30 stocks, because most were wrapped in HTML like `<a><span style="color: red">AAPL</span></a>`. I stripped the tags, trimmed, and uppercased them, and kept the original in a `raw_ticker` column.
- **Duplicates.** There were 29,460 exact duplicate rows. After removing them the table had exactly the same number of rows as the reference `ohlc` table, and there were no conflicting rows (same ticker and date with different values).
- **`adj_close` was the real problem.** 18,374 values were missing and another 10,417 were inflated by about 4x to 24x. The raw open/high/low/close and volume all matched the reference exactly.
- **Missing trading days: zero** for every ticker, checked against that ticker's dates in `ohlc`. The detector itself is tested on a small example where I removed one date on purpose.
- A few vendor quirks: 5 rows with `open = 0` (all on a ticker's first day, in 1962-66) and 75 zero-volume days.

Several cases from the assignment checklist simply don't occur in this database: missing prices, `high < close`, infinite values, missing or negative volume, rows with all prices missing, and conflicting keys. I didn't want to pretend they were there, so section 1.4 of the notebook builds a small fake dataset, injects each of these problems, runs the same cleaning code, and checks the results with asserts.

## Key design decisions

**How I found the bad `adj_close` values.** The raw close in this data is already split-adjusted, so the ratio `adj_close / close` should only change in small steps on dividend dates. I compared each day's ratio with the median of the 11 days around it, and anything more than 10% off gets flagged. This catches every inflated value without looking at the reference table. I then compared against `ohlc` as a separate check, and the two agreed almost exactly.

**How I rebuilt `adj_close`.** The assignment allows using matching rows in `ohlc` to get the adjustment factor, so the rebuilt value is `adj_close = close × factor`. I only used the reference factor if (a) our raw close matches the reference close, and (b) the reference factor itself passes the same 10% check. On 6 dates the vendor's own factor is off (for example AAPL on 1990-10-19), and copying it would just copy the error. For those I used the factor from the days before and after, but only when they agree within 0.5%, which means there was no dividend in between. In total, 28,790 values came from the reference factor and 6 from the neighbouring factors. Each row has `adj_close_repaired` and `adj_close_source` columns so you can see where a value came from.

**Impossible prices (high < close etc.).** I treat the open or close as the wrong field. I use the previous close if it falls inside today's high-low range, otherwise the midpoint of high and low. Plain forward fill can put yesterday's close outside today's range, which is still an impossible row, so this avoids that. The real data had no such rows, so this is only exercised by the injected tests.

**Missing values.** Rows with all four raw prices missing are dropped before filling. Other missing prices are forward-filled within the same ticker and then rechecked against the day's range. Missing volume becomes 0 with a `volume_repaired` flag, because unknown volume doesn't mean nothing traded.

**Quarantine instead of guessing.** The 5 zero-open rows are each ticker's first observation, so there's nothing earlier to fill from, and I didn't want to borrow a later price. I quarantined them (CVX, JNJ, MRK, PG on 1962-01-02 and MCD on 1966-07-05), along with KO's first row, whose adjustment factor is off and has no earlier day to compare with. Final count: 397,711 raw rows = 29,460 duplicates removed + 6 quarantined + 368,245 kept.

The raw `ohlc_hw` table is never modified. The notebook checks that it's unchanged at the end.

## How I handled outliers and why

The 3-standard-deviation screen (per ticker, both tails, sample std) flagged 5,056 daily returns. I didn't delete any of them automatically. I went through them with a few rules based on what the data actually shows:

- **Data errors (3 days, all TRV: 1976-10-28, 1978-12-20 and 1981-08-10).** On each of these the price doubles for one day and comes back the next, on zero volume, and the vendor's adjustment factor is also inconsistent that day. With all three signs together I was comfortable calling it a bad print, so I interpolated those days from the neighbouring days. These are the only prices I changed because of the outlier screen.
- **Dividends (22 days).** The raw price drops but the adjusted return is normal, because the dividend explains it. For example, MSFT on 2004-11-15 was the ex-date of its $3 special dividend. Kept as they are.
- **One suspicious day (AXP 1972-12-14).** It's a zero-volume move that reverses the next day, but there's no other evidence. Kept and flagged, since I can't prove it's wrong.
- **Everything else (5,027 days)** looks like real market moves: Oct 1987, 2000-02, 2008, March 2020 and so on. I kept these. Removing real crashes would make every stock look less risky than it was, which defeats the point of a risk report. I don't claim to know the cause of any of them. MCD on 1968-01-02 (-75%) might be a split the vendor never adjusted, but I can't confirm that from the data, so it's flagged and left alone.

One interesting side effect: after fixing the three TRV prints, TRV's number of flags went from 87 to 206. The fake +100% days had inflated TRV's standard deviation so much that they hid a lot of normal big moves. The notebook re-checks these new flags, and they're all ordinary market moves.

## Assumptions

- The expected calendar for each ticker is its dates in the supplied `ohlc` table (the assignment allows this). So "no missing days" means complete compared to the vendor history, not to the exchange calendar.
- `ohlc` is trusted as the source of adjustment factors, but only after passing the same consistency check.
- 252 trading days a year, a 2% risk-free rate, and the course's Sharpe approximation using daily log returns.
- Part 3 window: the six full calendar months before the month of the latest cleaned date. The latest date is 2024-12-06, so the window is 2024-06-01 to 2024-11-30, the same for every stock.
- The `reports/` folder is named after the date I ran the script (2026-10-22), not the analysis window.

## How to read the reports

Each report starts with a **coverage** table: the requested window next to the first and last date that actually exist. This matters here because 22 of the 30 stocks have no data after 2024-10-18, so their numbers cover about 4.5 months instead of 6. Nothing is extrapolated. These stocks are flagged in the CSV (`covers_requested_end = False`, plus `last_date`) and marked with an x on the scatter plot. Keep that in mind when comparing them with stocks like IBM or MRK that have the full six months.

After that comes a short table of any values in the window that were estimated during cleaning, and then the charts:

- **Price:** adjusted close, red where 20-day volatility is above its average for the period, green where it's at or below, and gray for the first 20 days before there's enough data. The blue dashed line is the 50-day moving average.
- **Daily returns:** green bars for up days, red for down days.
- **Rolling volatility:** 20-day volatility (annualized) against its average, with the high and low periods shaded.
- **Drawdown:** how far the price is below its previous peak, with the worst point marked.

The summary table at the end has the 10 metrics from the assignment. Total return is `last adj_close / first adj_close - 1`. Everything else uses daily log returns. If something can't be calculated (for example Sharpe with zero volatility), it shows N/A. In `portfolio_summary.csv` all metrics are stored as fractions, so `0.2125` means 21.25%.

## Sample output

I kept 3 of the 30 reports:

| Report | Why I picked it |
| --- | --- |
| `IBM_report.ipynb` | Best Sharpe ratio (3.22, +39.9% over the full six months) |
| `MRK_report.ipynb` | Worst Sharpe ratio (-1.93, -19.8%) |
| `AAPL_report.ipynb` | Example of a shorter history: the data ends 2024-10-18, so it covers only 97 trading days |

`portfolio_summary.csv` still lists all 30 stocks. The `report_path` values for the other 27 point to notebooks that aren't committed. Running `python 03_run_all_stocks.py` recreates them. The script also writes `failed_stocks.txt` on every run; all 30 reports succeeded, so it was empty and isn't committed.

## Small notes

- The summary table in each report is a plain pandas table, not a styled one. pandas styling needs `jinja2`, which isn't in the course environment, and I didn't want to add a dependency just for formatting.
- To run the quick tests of the window logic: `python -m doctest 03_run_all_stocks.py`.
- If I had more time, I would check the calendar against a real exchange calendar, and try to confirm the AXP 1972 and MCD 1968 cases from another data source.
