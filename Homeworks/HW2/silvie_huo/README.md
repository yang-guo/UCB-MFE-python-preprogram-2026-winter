# Stock Data Cleaning and Automated Risk Reports

Pipeline: **raw `ohlc_hw` → cleaned tables → one risk notebook per ticker → portfolio summary.**

## 1. How to run

Requires Python 3.12.3 and `data/data.db` containing the raw table `ohlc_hw` and the reference table `ohlc`.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m ipykernel install --user --name python3   # if the kernel name differs

# Step 1: run 01_data_cleaning.ipynb top to bottom (writes ohlc_hw_cleaned etc. into data.db)
# Step 2 (optional): open 02_stock_report_template.ipynb and edit ticker/dates
# Step 3: all tickers
python 03_run_all_stocks.py
python 03_run_all_stocks.py --tickers AAPL MSFT --run-date 2025-01-02   # subset
```

Outputs land in `reports/<run date>/`: one `<TICKER>_report.ipynb` per stock, `portfolio_summary.csv`,
`risk_return_profile.png`, `failed_stocks.txt` and `run.log`. If the notebook kernel is not found, pass `--kernel python3`.

## 2. Key design decisions

- **The raw table is never modified.** Every raw row ends up in exactly one place: cleaned, removed or quarantined. Counts are asserted to reconcile (raw = retained + removed + quarantined).
- **Repairs are flagged, never silent.** Each repair has a boolean `flag_*` column, and a final check confirms that unflagged rows equal their raw values.
- **Order matters.** Keys are normalized (HTML stripped, upper-cased, ISO dates) before any deduplication, and raw prices are repaired before extreme returns are measured.
- **The `ohlc` table is the independent witness.** It resolves conflicting duplicates, confirms or refutes outliers, and supplies the adjustment factor.
- **Adjusted close is rebuilt as `clean_close × same-date factor`.** The factor is never forward-filled, so a corporate action cannot leak into other days.
- **One common analysis window for all tickers.** It is the six complete calendar months before the month of the database's latest observation, never the computer clock. This makes runs reproducible and tickers comparable.
- **Robust orchestration.** The script reads metrics through scrapbook and checks that each report's ticker and window match the request. It deletes stale notebooks before each run, and one failing ticker never stops the batch. The database is opened read-only.
- **Units.** Calculations use fractions, and percentages are formatted only for display.

## 3. How outliers were handled and why

A 3-standard-deviation z-score alone cannot tell a bad tick from real news, and deleting real moves would distort risk. So every hit is classified with evidence, in this order:

| Class | Evidence | Action |
|---|---|---|
| `data_error` | Close disagrees with the `ohlc` reference | Interpolate (inside valid neighbours only) and rebuild adj_close |
| `split_like` | Volume jumps by a near-rational ratio (2, 3, 1/2, 3/2…) | Keep, flag |
| `ref_confirmed` | Close matches the reference, so the move is real | Keep, flag |
| `revert_of_data_error` | Inconclusive move right after a data error (the reversion) | Keep, flag |
| `inconclusive` | No evidence either way | Quarantine with a stated uncertainty |

The reference check runs before the volume check, because an external source is stronger evidence than a heuristic. Quarantine is preferred over guessing, and nothing is extrapolated. Z-scores are recomputed after repair, so some extreme days legitimately remain in the cleaned data.

## 4. Assumptions

- The `ohlc` table is trustworthy where it has a single, non-conflicting row per key.
- **Vendor `adj_close` that disagrees with `close × reference factor` is overwritten**, treated like a missing value. The assignment was silent on this. Not overwriting gave unreadable, jagged charts, so I inferred it was intended. The change is flagged (`flag_adj_vendor_inconsistent`) and is easy to switch off (`reconstruct_inconsistent=False`).
- Missing volume is filled with 0 and flagged. This is not evidence of no trading, so downstream code never treats flagged volume as real.
- Tunable thresholds: 3σ for outliers, 0.1% price tolerance against the reference, 2% split-ratio tolerance.
- 252 trading days a year, a 2% annual risk-free rate, a 20-day volatility window and a 50-day moving average.
- A report needs at least 21 prices. Tickers with less raise a clear error and appear in `failed_stocks.txt`.

## 5. How to interpret the reports

**Dashboard (per ticker)**
- *Price:* the line is green when 20-day volatility is at or below its period mean, red when above, and gray during the 20-day warm-up. The colouring is **retrospective** because the mean uses the whole window. The dashed line is the 50-day MA.
- *Returns:* daily log returns. *Volatility:* annualized (×√252), with shading above and below the mean.
- *Drawdown:* the percentage fall from the running peak, with the maximum marked.

**Summary statistics**
- *Annualized volatility* is the whole-period figure. It differs from the *mean rolling volatility* used in the portfolio plot.
- *Sharpe* is a course approximation: `(mean log return × 252 − rf) / annual vol`. Use it to rank, not as an exact figure.
- *Total return* uses `adj_close`, so it includes dividends and splits.

**Portfolio summary**
- Tickers are sorted by Sharpe, with the top and bottom 10 printed.
- In the scatter plot, x is average volatility, y is total return and colour is Sharpe. Triangles mark **shorter-than-window history**, so their return, volatility and drawdown cover fewer days and are not like-for-like with circles. Gray crosses mark an undefined Sharpe.
- Check `n_days` and `coverage_pct` before comparing tickers.
- Six months is a short sample, so treat rankings as descriptive rather than predictive.