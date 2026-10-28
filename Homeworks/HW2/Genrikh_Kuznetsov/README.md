# HW2: Stock analysis pipeline (Genrikh Kuznetsov)

`ohlc_hw` (dirty) → **`01_data_cleaning.ipynb`** → `ohlc_hw_cleaned` → **`02_stock_report_template.ipynb`** (one stock) → **`03_run_all_stocks.py`** (every stock + portfolio summary)

## Files

| File | Purpose |
|---|---|
| `01_data_cleaning.ipynb` | Inspect, diagnose, clean, and validate `ohlc_hw`; writes `ohlc_hw_cleaned` |
| `02_stock_report_template.ipynb` | Papermill template (tagged `parameters` cell) for one ticker's risk report |
| `03_run_all_stocks.py` | Runs the template for every ticker, collects results, saves the summary CSV and plot |
| `requirements.txt` | Packages used |
| `reports/2026-10-28/` | Sample output: `AAPL`, `IBM`, `MRK` reports, `portfolio_summary.csv`, `risk_return.png`, `failed_stocks.txt` (empty: 30/30 succeeded) |

## How to run

All commands are run from the repository root. The repo's uv environment already has every package in `requirements.txt`.

```bash
uv sync --locked                                   # once: create/refresh .venv

# 1. Put the bCourses database in this folder (it is gitignored):
mkdir -p Homeworks/HW2/Genrikh_Kuznetsov/data
cp "Lectures/Lecture 4/data/data.db" Homeworks/HW2/Genrikh_Kuznetsov/data/data.db

# 2. Clean the data (creates ohlc_hw_cleaned in that working copy). Takes about 30 s.
cd Homeworks/HW2/Genrikh_Kuznetsov
uv run papermill 01_data_cleaning.ipynb 01_data_cleaning.ipynb --cwd .
#    (or open it in Jupyter with the .venv kernel and Restart & Run All)

# 3. Generate all reports and the portfolio summary (about 2 minutes for 30 tickers).
uv run python 03_run_all_stocks.py
```

The notebooks use paths relative to this folder, so run them with this folder as the working directory. The script finds its paths from its own location and works from anywhere. To try a single report, open `02_stock_report_template.ipynb` and change `ticker`/`start_date`/`end_date` in the first cell.

**Run order matters:** the report and the script read `ohlc_hw_cleaned`, which only exists after step 2. Nothing writes to `ohlc_hw` or `ohlc`.

## What I found in the data and what I did

| Problem | Count | Handling |
|---|---|---|
| Ticker labels wrapped in HTML (`<a><span style="color: blue">AAPL</span></a>`) | 150 labels → 30 tickers | Tags removed, stripped, upper-cased **before** any grouping; label→ticker audit saved as `ohlc_hw_ticker_audit` |
| Exact duplicate records | 29,460 | Dropped |
| Conflicting `(ticker, ts)` keys | 0 | Would be quarantined (not "keep first") |
| Missing `adj_close` | 18,374 | Rebuilt as `close × factor`, using the reference `ohlc` table's `adj_close/close` **for the same ticker and date** |
| `adj_close` 4–24× too large | 10,417 | Detected without the reference: the `adj_close/close` factor is > 2× the median of the surrounding 11 days. Cross-check against `ohlc`: 10,417/10,417 caught, 0 false alarms. Rebuilt the same way |
| One-day 2× price spikes on zero volume (TRV 1976, 1978, 1981) | 3 | Confirmed errors (see below): OHLC and factor interpolated from neighboring days |
| `open = 0` on a ticker's first day (1962/1966) | 5 | Quarantined: no earlier price exists to estimate from |
| Zero-volume days | 74 | Kept, `zero_volume_flag` (the reference has them too) |
| Missing trading days, missing OHLC, OHLC bound violations, missing volume | 0 | Correctly zero; each detector is demonstrated on injected faults |

Row accounting: 397,711 raw = 29,460 duplicates + 5 quarantined + **368,246 retained**. Quarantined rows are saved in `ohlc_hw_quarantine` with a reason.

## Key design decisions

* **Normalize, then de-duplicate, then sort** by `ticker, ts`. Every fill, return, and rolling calculation is done within a ticker, so a value never leaks from one stock into another.
* **Diagnose before repairing.** The full Data Quality Report (`ticker | issue_type | issue_date | description | severity`) is built from the unrepaired data. After cleaning, the same detector runs again, and the notebook shows before/after counts.
* **The reference `ohlc` table is used only for two things**, as the assignment allows: (1) each ticker's dates are its trading calendar, and (2) the same-date adjustment factor is used to rebuild `adj_close`. It is never used to replace the dirty table. Raw OHLC in `ohlc_hw` matches it exactly, which is what makes our raw close "validated".
* **Adjustment factors are never forward-filled across dates**: a factor changes at dividends and corporate actions.
* **Every repair is flagged.** The cleaned table has `price_repaired`, `volume_repaired`, `adj_close_repaired` (+ note columns) and review flags `extreme_return_flag`, `zero_volume_flag`. `ts` is stored as `YYYY-MM-DD` text.
* **Unknown volume → 0 *with a flag***, because unknown is not the same as no trading. (None occurred, but the rule is implemented and tested.)

## How outliers were handled, and why

**Impossible prices** (`high < close`, `low > close`, open outside the range, zero/negative prices). I used **Option 1 plus a range check**. The close is treated as the suspect field because it's what the report uses, and the day's high/low are kept as the range. The bad value is set to `NaN`, then filled with the previous close **only if that lies inside today's `[low, high]`**, otherwise with the midpoint of today's range. A plain forward fill can create a new bounds violation (shown in the notebook). With real data we can't know which field is wrong, so this is an assumption, and the row is flagged. Leading gaps with no earlier price are quarantined instead.

**Extreme returns** (`|r - mean| > 3 × std` per ticker, sample std, both tails). About 5,000 days are flagged. Over decades of fat-tailed returns, that is expected. Before deciding anything, I looked for evidence:

* none of the flagged closes disagree with the reference;
* about 29% fall on dates when five or more stocks were flagged together (Oct 1987, Oct 2008, Mar 2020), which is a market-wide move rather than a bad quote;
* some coincide with a step in the adjustment factor and an ordinary *adjusted* return (e.g. MSFT 2004-11-15), so the adjustment explains them.

Those are **kept and flagged, not changed**. A flagged return is a **confirmed error** only if (a) the row's OHLC is internally consistent but its close disagrees with the trusted reference, or (b) it is a **one-day round trip**: a flagged move larger than 40% whose next return is also flagged, with the next close within 2% of the previous one. Only the 3 TRV spikes meet these rules (exactly 2× the prior close, zero volume, all four prices equal, fully reversed next day, and present in the reference too). They are linearly interpolated between valid neighbors (`limit_area='inside'`, never extrapolated).

**Uncertain cases, kept as-is and documented:** MCD 1968-01-02 (raw *and* adjusted fall 75% and stay there: either a 4:1 split that was never adjusted for, or an error, and the data can't tell which); DIS 1974–75 and IBM 1975 (single-day 6–7% factor steps, two DIS rows dated on Sundays). **Known limitation:** AAPL 1990-10-19 has `adj_close = close` for one day (a 25% factor anomaly, also in the reference). That is below the 2× scale threshold, so it is not repaired. None of these are near the 2024 report window.

## Assumptions

* Each ticker's dates in `ohlc` are its exchange trading calendar over its own history. Weekdays alone would wrongly count holidays (and the 9/11 closure) as gaps.
* `close` in this dataset is already split-adjusted by the vendor; `adj_close/close` reflects dividends and other distributions.
* Report window = the six full calendar months **before the month of the latest cleaned observation** (2024-12-06 → **2024-06-01 to 2024-11-30**), anchored to the data, not the computer clock. The output folder name is the *run* date.
* Risk-free rate 2%/year and 252 trading days/year (course convention).

## How to read the reports

Each `reports/<run date>/<TICKER>_report.ipynb` shows the requested window and the **actual first/last dates and row count**, then:

* **Price and 50-day MA**: adjusted close, red where the 20-day volatility is above the report's mean and green at or below it, gray during the first 20 days when no volatility estimate exists yet. The coloring uses the whole window's mean, so it is retrospective.
* **Daily returns**: log returns, green up / red down.
* **Rolling volatility**: annualized 20-day volatility vs. its mean, with high/low regimes shaded.
* **Drawdown**: how far below its running peak the stock was on each day; the worst point (maximum drawdown) is marked.
* **Summary table**: total return (`last/first − 1`), compounded annualized return, whole-period annualized volatility, Sharpe, max drawdown, best/worst day, share of positive days, average up/down day. `N/A` means undefined (e.g. zero volatility).

`portfolio_summary.csv` holds **fractions** (0.25 = 25%), sorted by Sharpe. `avg_volatility` is the *mean of the rolling 20-day volatility*, which is different from the whole-period volatility in each report. `first_date`, `last_date`, `trading_days` record each stock's coverage.

**Comparison caveat:** all stocks get the same requested window, but **22 of 30 histories end on 2024-10-18** (97 trading days) while 8 run to 2024-11-29 (126 days). The rankings compare different lengths of time, so a stock with a shorter history isn't directly comparable to one with the full window. Prices are never extrapolated past a stock's last date.

In the latest run, IBM had the highest Sharpe (3.2, +40%), and MRK, BA, and NKE were the only stocks with negative returns. NVDA had by far the highest volatility (≈61%), with a modest Sharpe (0.78).
