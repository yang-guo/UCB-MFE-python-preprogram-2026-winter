# HW2 – Stock Data Cleaning and Automated Risk Reports (Scott Westvold)

This project cleans a raw daily OHLC price table for 30 Dow stocks, then generates a parameterized
risk report notebook for every stock with papermill.

## Requirements

- **Python** with the packages in `requirements.txt`:

  | Package | Version | Used for |
  |---|---|---|
  | pandas | 3.1.0 | data loading, cleaning, metrics |
  | numpy | 2.5.4 | numeric calculations |
  | matplotlib | 3.11.2 | report charts |
  | papermill | 2.7.0 | running the report template for each ticker |
  | tqdm | 4.70.1 | progress bar in the batch runner |
  | scrapbook | 0.5.0 | saving/reading summary tables and figures from report notebooks |
  | pyarrow | 12.0.1 | required by scrapbook's pandas encoder |

  Standard-library modules used (no install needed): `sqlite3`, `datetime`, `typing`, `math`, `os`, `logging`.

- **A Jupyter kernel (`ipykernel`)** in the environment that runs the notebooks. papermill executes
  `02_stock_report_template.ipynb` with the notebook's `python3` kernel.
- **The database** `data/data.db` (distributed via bCourses, gitignored). It must contain the raw
  `ohlc_hw` table.

Install:

```
pip install -r requirements.txt
pip install ipykernel
```

## How to Run

Run everything from this folder (`Homeworks/HW2/SCOTT_WESTVOLD`), since all paths are relative to it.
Run the three steps in order:

| Order | File | Inputs | Outputs |
|---|---|---|---|
| 1 | `01_data_cleaning.ipynb` | `data/data.db` → table `ohlc_hw` (raw prices) | `data/data.db` → table `ohlc_hw_cleaned`; data quality report and repair summary printed in the notebook |
| 2 | `02_stock_report_template.ipynb` | Parameters (`ticker`, `start_date`, `end_date`, `db_path`); table `ohlc_hw_cleaned` | Validation summary, 2x2 chart grid (price & 50-day MA, daily returns, drawdown, rolling volatility), summary metrics table; scrapbook scraps `summary_table`, `scrap_book_summary_table`, `report_figure` |
| 3 | `03_run_all_stocks.py` | Table `ohlc_hw_cleaned`; `02_stock_report_template.ipynb` | `reports/YYYY-MM-DD/{TICKER}_report.ipynb` (one per ticker), `reports/YYYY-MM-DD/run_log.txt`, `reports/YYYY-MM-DD/failed_stocks.txt` |

1. **Clean the data:** open `01_data_cleaning.ipynb` and run all cells. This writes `ohlc_hw_cleaned`.
2. **(Optional) Try one report:** open `02_stock_report_template.ipynb`, edit the tagged `parameters`
   cell, and run all cells. You don't need to run it manually for the batch.
3. **Generate all reports:**

   ```
   python 03_run_all_stocks.py
   ```

   The script runs the template once per ticker in `ohlc_hw_cleaned`. The report window is the
   latest date in the data minus 182 days through the latest date. One stock failing does not stop
   the others: each error is logged to `run_log.txt` and the ticker is listed in `failed_stocks.txt`
   (one per line; empty if nothing failed).

## Results of the Report Run (2026-11-01)

- **Report window:** 2024-06-07 to 2024-12-06
- **Final run (01:43):** **30 / 30 reports succeeded**; `failed_stocks.txt` is empty.
- **Earlier runs the same day** (kept in `run_log.txt`, which appends):
  - 01:23: 0/30 succeeded – every report failed with `NameError: name 'validate_data' is not defined`
    (the template called `validate_data` before the cell defining it). Fixed in the template.
  - 01:30: 27/30 succeeded – AAPL, AMGN, AMZN failed with the same error before the fix was saved;
    those three were rerun successfully at 01:38.

| Ticker | Report | Ticker | Report | Ticker | Report |
|---|---|---|---|---|---|
| AAPL | ✅ | GS | ✅ | MSFT | ✅ |
| AMGN | ✅ | HD | ✅ | NKE | ✅ |
| AMZN | ✅ | HON | ✅ | NVDA | ✅ |
| AXP | ✅ | IBM | ✅ | PG | ✅ |
| BA | ✅ | JNJ | ✅ | SHW | ✅ |
| CAT | ✅ | JPM | ✅ | TRV | ✅ |
| CRM | ✅ | KO | ✅ | UNH | ✅ |
| CSCO | ✅ | MCD | ✅ | V | ✅ |
| CVX | ✅ | MMM | ✅ | VZ | ✅ |
| DIS | ✅ | MRK | ✅ | WMT | ✅ |

Failed reports: **none** in the final run.

## Tickers and Valid Date Ranges

First and last date with data for each ticker (after merging the HTML-wrapped ticker variants).
The data ends on different dates per ticker, so for tickers ending 2024-10-18 the report window
effectively covers 2024-06-07 to 2024-10-18.

| Ticker | First Date | Last Date |
|---|---|---|
| AAPL | 1980-12-12 | 2024-10-18 |
| AMGN | 1983-06-17 | 2024-10-18 |
| AMZN | 1997-05-15 | 2024-12-06 |
| AXP | 1972-04-03 | 2024-10-18 |
| BA | 1962-01-02 | 2024-10-18 |
| CAT | 1962-01-02 | 2024-10-18 |
| CRM | 2004-06-23 | 2024-10-18 |
| CSCO | 1990-02-16 | 2024-10-18 |
| CVX | 1962-01-02 | 2024-12-06 |
| DIS | 1962-01-02 | 2024-12-06 |
| GS | 1999-05-04 | 2024-10-18 |
| HD | 1981-09-22 | 2024-10-18 |
| HON | 1962-01-02 | 2024-10-18 |
| IBM | 1962-01-02 | 2024-12-06 |
| JNJ | 1962-01-02 | 2024-10-18 |
| JPM | 1969-03-05 | 2024-10-18 |
| KO | 1962-01-02 | 2024-12-06 |
| MCD | 1966-07-05 | 2024-10-18 |
| MMM | 1962-01-02 | 2024-10-18 |
| MRK | 1962-01-02 | 2024-12-06 |
| MSFT | 1986-03-13 | 2024-10-18 |
| NKE | 1980-12-02 | 2024-12-06 |
| NVDA | 1999-01-22 | 2024-10-18 |
| PG | 1962-01-02 | 2024-12-06 |
| SHW | 1968-01-02 | 2024-10-18 |
| TRV | 1972-04-03 | 2024-10-18 |
| UNH | 1984-10-17 | 2024-10-18 |
| V | 2008-03-19 | 2024-10-18 |
| VZ | 1983-11-21 | 2024-10-18 |
| WMT | 1972-04-03 | 2024-10-18 |

## Data Cleaning Choices

All repairs use only the `ohlc_hw` table itself (no external or reference data). Every repair is
marked with a `flag_*` column so changes are traceable, and the raw OHLC bounds and adjusted-price
consistency are rechecked after every step.

1. **Messy tickers:** some tickers were wrapped in HTML (e.g. `<a><span style="color: red">AMGN</span></a>`).
   Tags are stripped and tickers uppercased, merging the variants into 30 clean tickers.
2. **Duplicate (ticker, ts) rows:** 29,460 duplicates removed. In each group the most complete row
   (fewest missing values) is kept; ties keep the first row.
3. **Invalid prices:** non-positive or infinite prices (including `adj_close`) are treated as missing.
4. **Empty rows:** rows where all of open, high, low, and close are missing are dropped before any filling.
5. **Missing prices:** open/high/low/close are forward-filled within each ticker (last known price).
6. **Missing volume:** filled with 0 and flagged (`flag_volume_filled`), since unknown volume is not
   evidence of no trading.
7. **adj_close:** not forward-filled as a price, because that breaks the adjustment factor
   (`adj_close / close`). Missing, impossible (`adj_close > close`), and spiked values are rebuilt as
   `close × factor` from the nearest valid days, **only if** the factors before and after the gap agree
   within 0.5% (no dividend or split inside the gap). Otherwise the row is quarantined.
8. **Missing trading days:** the trading calendar is every date that appears for any ticker. A missing
   day inside a ticker's history is inserted as a flat bar at the previous real day's close and factor
   (`flag_date_filled`).
9. **Impossible price relationships** (high below open/close/low, or low above open/close):
   **my choice was Option 1 – replace the bar with NaN and forward-fill** from the previous day's bar,
   so the replacement comes from one internally consistent bar (`flag_hl_ffilled`).
10. **Extreme returns** (more than 3 standard deviations from the ticker's mean daily return) are
    classified with recorded evidence (`outlier_type`, `outlier_evidence`):
    - **Confirmed data error:** a one-day spike that reverses the next day while the market is calm,
      or an otherwise-unexplained move larger than 10 std → interpolated from the neighboring days,
      only between two valid neighbors (never extrapolated).
    - **Split:** price changed by a round multiple (×1.5, ×2, ×3, …) while `adj_close` stayed continuous → kept, flagged.
    - **VIX high day:** at least 30% of the other tickers also moved more than 2 std that day (a
      market-wide proxy; no VIX data used) → kept, flagged, regardless of size.
    - **Possible merger:** volatility collapsed afterward or the series ends soon after → kept, flagged as unconfirmed.
    - **No error detected:** remaining moves, labeled by size (3–5 std or 5–10 std) → kept, flagged.
11. **Quarantine:** rows that can't be repaired defensibly (no earlier value to fill from, an
    `adj_close` gap spanning a possible corporate action, or a confirmed error without two valid
    neighbors) are removed from the cleaned table with a recorded `quarantine_reason`, and counts are reported.

## Known Limitations

- `ts` is stored in `ohlc_hw_cleaned` as `YYYY-MM-DD 00:00:00` text; date filters compare against
  `YYYY-MM-DD` strings, so an `end_date` equal to a ticker's last date excludes that day.
- In the template, the `volatility` column is computed with `calculate_volatility(df_tx['adj_close'])`
  (prices rather than returns), so `avg_volatility` in the scrapbook summary reflects price-level
  variation, not return volatility.
- The remaining small day-to-day wobbles in `adj_close / close` (under the 2% spike threshold) are
  treated as source-data precision rather than errors.
