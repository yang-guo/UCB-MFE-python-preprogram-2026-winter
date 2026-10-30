# Homework 2: Scalable Stock Analysis Pipeline

This homework cleans the supplied OHLC data, builds a parameterized single-stock risk report, and runs the report across all tickers.

## Files and local data

```text
David_MartinsHuber/
├── data.db                         # local source database; ignored by Git
├── data.db.zip                     # optional source archive; ignored by Git
├── data/
│   └── data.db                     # local working copy; ignored by Git
├── 01_data_cleaning.ipynb
├── hw2_cleaning_workers.py         # importable ticker worker for process-pool cleaning
├── 02_stock_report_template.ipynb
├── 03_run_all_stocks.py
├── requirements.txt
└── reports/
    └── YYYY-MM-DD/                 # generated batch; 3 samples and CSV are submitted
```

Run notebooks and scripts from this folder. The cleaning notebook copies the root source database into `data/data.db` when the working copy is missing. It reads `ohlc_hw` and the clean `ohlc` reference table from the working copy, leaves `ohlc_hw` untouched, and writes `ohlc_hw_cleaned` and its audit tables there. The batch runner also reads `data/data.db` by default. Pass `--db-path` to use a different working copy.

## Setup and run order

Install the packages and ensure the `python3` Jupyter kernel is available:

```bash
python -m pip install -r requirements.txt
python -m ipykernel install --user --name python3
```

Then:

1. Run `01_data_cleaning.ipynb` from top to bottom. It creates and validates `ohlc_hw_cleaned` in `data/data.db`.
2. Run `02_stock_report_template.ipynb` for the default AAPL example, or execute it with Papermill parameters.
3. Run the full ticker batch:

   ```bash
   python 03_run_all_stocks.py
   ```

The script writes the current run under `reports/YYYY-MM-DD/`. Full dated runs stay local except for three selected sample notebooks (AAPL, IBM, MRK) and `portfolio_summary.csv`, which are allowed through `.gitignore` in the run-date folder.

Optional runner paths:

```bash
python 03_run_all_stocks.py --db-path data/data.db --output-root reports --kernel-name python3
```

## Data decisions and assumptions

- Tickers are normalized before duplicate checks; original labels are retained. Exact duplicate rows are removed, conflicting normalized ticker/date records are quarantined, and missing dates are compared with the supplied ticker-specific `ohlc` calendar.
- Cleaning runs ticker groups in parallel with `max(1, os.cpu_count() - 2)` workers. Repair flags are retained in the cleaned table. Missing raw prices are forward-filled only within ticker, with no backfill of leading gaps. Missing or invalid volume is set to zero only with a repair flag.
- For a nonpositive or non-finite observed open with a valid same-row range (`low <= close <= high`), open is estimated as `(low + close) / 2`. This is explicitly flagged as an estimate. An inverted `high < low` range marks both bounds as suspect; the other impossible inequalities mark the offending high or low. Those fields are filled only from earlier values of the same ticker, then raw bounds are rechecked. Unresolved rows are quarantined.
- Daily simple-close returns more than three sample standard deviations from their ticker mean are review flags. The dataset has no split, merger, or news-event metadata, so these flags are not automatically interpolated or deleted. An event should be corrected only when evidence supports it.
- Adjusted close is repaired from a same-date clean-reference factor when available; factors are never carried across dates. The cleaned data is checked again for positive finite prices, valid OHLC bounds, unique keys, and finite nonnegative volume.
- The single-stock report uses adjusted-close daily log returns. Annualization uses 252 trading days; Sharpe uses a 2% annual risk-free rate. Historical 95% one-day VaR and expected shortfall use a positive-loss convention and need at least 20 returns. Skewness and unbiased Fisher excess kurtosis are computed from daily log returns.
- The batch window is the six complete calendar months before the month containing the latest cleaned observation, not the computer clock. Each report and CSV disclose actual ticker coverage. Prices are never extrapolated through a ticker's trailing gap, so comparisons should account for different observation counts and end dates.

## Reading the reports

The single-stock notebook shows adjusted price, a 50-day moving average, daily returns, rolling 20-day annualized volatility, drawdown, and a summary table. Warm-up gaps remain missing. The portfolio CSV keeps metrics as fractions; charts and notebook tables format percentages for display. `avg_volatility` in the portfolio CSV is the mean rolling volatility over the ticker's available portion of the shared window.

Generated artifacts include per-ticker executed notebooks, `portfolio_summary.csv`, `risk_return_scatter.png`, `failed_stocks.txt`, and `manifest.json`. The report stores machine-readable metrics in a visually empty, named notebook output. Only reports successfully executed in the current run contribute metrics; a failure for one ticker does not stop the remaining reports.

## Current batch run

The local batch dated `2026-10-30` processed all 30 tickers successfully. The latest cleaned observation was `2024-12-06`, so the shared analysis request was `2024-06-01` through `2024-11-30`. Some supplied histories end earlier than November; each report and the CSV show the actual first/last dates and observation count. The submitted examples are AAPL, IBM, and MRK, alongside the 30-ticker portfolio CSV.

The outlier audit also compares every flagged raw close with the same ticker/date in the supplied clean `ohlc` reference. In the current data, all 5,056 flagged closes agree with that reference. This supports retaining them as review flags; the dataset still cannot establish the event cause.
