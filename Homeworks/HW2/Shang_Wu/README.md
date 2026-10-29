# HW2: Stock Analysis Pipeline (Shang Wu)

A three-step pipeline: clean the dirty `ohlc_hw` table, produce a parameterized risk report for one stock, then run that
report for every stock and summarize the portfolio.

| File | What it does |
|---|---|
| `01_data_cleaning.ipynb` | Detects data-quality issues, repairs or quarantines them, writes `ohlc_hw_cleaned` and `ohlc_hw_quarantine` |
| `02_stock_report_template.ipynb` | Papermill-parameterized risk report for one ticker and date range |
| `03_run_all_stocks.py` | Runs the report for every ticker and builds the portfolio summary |
| `requirements.txt` | Pinned dependencies (tested with Python 3.12) |
| `reports/2026-10-29/` | Sample reports, `portfolio_summary.csv`, `risk_return_profile.png` |

---

## How to run

**1. Set up.** From this folder:

if use `uv`

```bash
uv pip install -r requirements.txt
```

or if use raw python

```bash
pip install -r requirements.txt
```

Download `data.db` from bCourses and place a **working copy** at `data/data.db` (relative to this folder). It is
gitignored and is never committed. The cleaning step writes new tables into this copy and never modifies `ohlc_hw`.

**2. Clean the data.** Run `01_data_cleaning.ipynb` top to bottom (Jupyter, or `papermill 01_data_cleaning.ipynb out.ipynb`).
It creates:
- `ohlc_hw_cleaned`: 367,874 rows, 30 tickers, with every repair and review flag as columns
- `ohlc_hw_quarantine`: 377 rows that could not be repaired defensibly, each with a reason

**3. (Optional) One report.** Open `02_stock_report_template.ipynb` and edit the first code cell (tagged `parameters`),
or run it with papermill:

```bash
papermill 02_stock_report_template.ipynb AAPL_2024.ipynb -p ticker AAPL -p start_date 2024-01-01 -p end_date 2024-12-31
```

**4. All reports.**

```bash
python 03_run_all_stocks.py                      # every ticker in ohlc_hw_cleaned
python 03_run_all_stocks.py --tickers AAPL MSFT  # a subset (useful for testing)
python 03_run_all_stocks.py --help               # other options: --db-path, --template, --output-root, --kernel
```

Output goes to `reports/{run date}/`. It contains one `{TICKER}_report.ipynb` per ticker, `failed_stocks.txt`,
`portfolio_summary.csv`, `risk_return_profile.png`, and `run.log`. The full run takes under a minute and generated
30/30 reports. Only three sample reports are committed (the summary CSV still lists all 30 paths); rerun the script to
regenerate the rest.

**Sample reports in `reports/2026-10-29/`:**

| Report | Why it was chosen |
|---|---|
| `PG_report.ipynb`: **typical** | Covers the full window (126 trading days), and every headline metric is near the 30-stock median: Sharpe 1.04, return +10.2%, volatility 16.6%, drawdown −8.9%. |
| `NVDA_report.ipynb`: **very strange** | Farthest from the median overall. Volatility is 61%, about twice the next stock. A +20% return gives a Sharpe of only 0.78, and its −27% drawdown ties for the worst. |
| `HON_report.ipynb`: **lots of missing data** | Data ends 2024-10-18, so 29 of the window's 126 trading days are absent. 10 of its 97 remaining `adj_close` values were missing in the raw data and rebuilt (2 more were corrupted), the most of any stock except NVDA. Its Data notes section shows these repairs. |

Run order matters: 02 and 03 read `ohlc_hw_cleaned`, which 01 creates.

---

## Key design decisions

**Cleaning (Part 1)**
- **The raw table is never modified.** All work happens on copies. The notebook checks that `ohlc_hw` is unchanged
  after writing.
- **Repair, quarantine, or flag. Never silently delete.**
  - Rows that can be repaired get a flag column and keep their original value (`*_vendor` columns).
  - Rows that cannot be repaired defensibly go to `ohlc_hw_quarantine` with a reason.
  - Plausible but unusual data (extreme returns, zero volume) is kept and flagged.
  - Every raw row is accounted for: 397,711 raw = 29,460 duplicates removed + 377 quarantined + 367,874 retained.
- **Tickers are normalized before anything else.** HTML markup and whitespace are stripped and labels upper-cased, so 150
  raw labels become 30 tickers. Duplicates are found *after* this step. Exact duplicates are dropped, and any
  conflicting `(ticker, ts)` rows would stop the pipeline for investigation (there were none). The original label is
  kept in `ticker_raw`.
- **The diagnosis comes before the repair.** The Data Quality Report (`ticker | issue_type | issue_date | description | severity`)
  is built on the uncleaned data and kept. After cleaning, the same detectors run again on the saved table.
- **Every detector is demonstrated on a small injected example**, including the checks that find nothing in the real
  data, so a result of zero is a real finding and not a silent failure.
- **Missing trading days are checked against the supplied `ohlc` table's dates** for each ticker, within that ticker's
  own history, rather than against business days (which would wrongly flag every exchange holiday). Result: 0 gaps.
- **Missing `adj_close` (18,374) is rebuilt as `validated raw close × date-specific adjustment factor`.** The factor is
  `adj_close / close` from the matching `ohlc` row, and is used only after confirming that row's raw OHLC and volume
  match ours. Factors are never forward-filled across dates.
- **Severity:**
  - *high*: impossible or unusable value
  - *medium*: usable, but biases results until fixed (missing or duplicate data, returns ≥ 10 std)
  - *low*: plausible but worth a look (3–10 std returns, zero volume)

**Report (Part 2)**
- **Prices and calculations:** adjusted close for prices, and daily log returns for every return-based metric. Values
  stay as fractions in calculations and are formatted as percentages only for display.
- **Undefined values stay `NaN` and display as `N/A`.** This covers zero volatility, too few returns, and the
  rolling-window warm-up, which is never filled with 0.
- **Clear errors for bad input.** Bad inputs (unknown ticker, reversed dates, fewer than 2 prices) raise a clear error,
  so the batch run can record the failure. Fewer than 21 prices still produces a report, with rolling volatility shown
  as unavailable.
- **The database is opened read-only.**
- **Key results are saved as Scrapbook outputs** so the batch script can collect them.

**Batch run (Part 3)**
- **The window comes from the data, not the clock.** It is the six calendar months before the month of the latest
  cleaned observation. With a latest date of 2024-12-06, that gives **2024-06-01 to 2024-11-30**. The latest month is
  always excluded.
- **The output folder is named by run date**, independent of the analysis window.
- **Results come only from reports that succeeded in this run.** Failed or old notebooks in the folder are never read.
  Undefined Sharpe ratios are excluded from the rankings.
- **Every ticker runs inside `try/except`.** An all-failed run writes `failed_stocks.txt`, prints no summary, and exits
  with code 1.

---

## How I handled outliers, and why

### Impossible price relationships

The only impossible prices were **5 zero opens**, each on a ticker's first day of data. The `ohlc` reference has the
same zeros, so they are a vendor placeholder. I compared the three suggested repairs on 20,000 simulated corruptions of
valid bars, since the true values of the real 5 are unknown:

| Option | Bad open: median error | Bad open: valid bar | Bad close: median error |
|---|---|---|---|
| 1. NaN, then forward-fill | 0.89% | 46% | 0.92% |
| 2. Previous close | 0.26% | 73% | 0.92% |
| 3. Midpoint of high/low | 0.50% | 100% | **0.46%** |
| 2, clipped to [low, high] | **0.09%** | 100% | 0.77% |

**Chosen rule:**
- **Bad open:** previous close clipped to the day's range. This uses option 2's information with option 3's guarantee
  of a valid bar. On a ticker's first day, where there is no previous close, it falls back to the midpoint.
- **Bad close:** the midpoint.
- **Ambiguous bars** (inverted range, or both open and close out of range): quarantined.
- Closes are repaired before opens, so an open never uses yesterday's bad close.

All 5 real cases used the first-day midpoint. Neither option 1 nor option 2 could have repaired them.

### Extreme returns (> 3 std)

5,056 days were flagged. Each was investigated using four pieces of evidence: whether the move reversed the next day,
volume relative to its previous 20 days, whether the adjustment factor changed (a split or dividend), and how many other
tickers were flagged that day.

| Verdict | Days | Action |
|---|---|---|
| Data error | 3 (all TRV) | 2 interpolated in time from the neighboring days; 1 quarantined |
| Day after a data error | 3 | Fixed together with the error |
| Corporate action | 5 | Kept, flagged (e.g. MSFT 2004-11-15: raw −8.6%, but the adjusted return is normal) |
| Series break | 1 (MCD 1968-01-02) | The 376 earlier MCD rows quarantined |
| Market-wide event | 572 | Kept, flagged (1987-10-19, Oct 2008, Mar 2020) |
| Genuine, unconfirmed | 4,472 | Kept, flagged (90% traded above normal volume) |

**Why these calls:**
- **TRV data errors.** Each one was a doubled price on a flat, zero-volume bar that reversed the next day, with a
  one-day jump in the vendor's adjustment factor. Real news does not look like that.
- **The quarantined TRV day.** Its neighbors' adjustment factors differ by 0.2%, so interpolating across them could
  cross a small distribution. I quarantined it instead.
- **The MCD series break.** The price fell about 4× and normal volume rose about 4× permanently, with no change in the
  adjustment factor. That is consistent with a ~4-for-1 split never applied to the earlier history. A one-day
  interpolation cannot fix a permanent level shift, and rescaling would require guessing the ratio, so the earlier
  history is quarantined.
- **The data-error rule had to be tightened.** It now requires volume below half its usual level *and* a day that is
  not market-wide. Without this, UNH on 2008-09-29 (a real crash day that partly bounced back) would have been
  "repaired".

After repair, the screen flags 5,200 days. The increase is entirely TRV and MCD: removing their huge errors shrinks their
standard deviation, so more ordinary large days cross 3 std. These remain review flags.

### A second adjusted-close problem

Comparing the vendor `adj_close` with `close × reference factor` revealed **10,417 values that were present but wrong**.
They were 4–24× too large, about 3% of every ticker, and 99% sat above the day's raw high. None of the Part 1.2 checks
catch these, because they are positive and finite. They were rebuilt with the reference factor and labelled
`adj_close_source = rebuilt_inconsistent`, with the original kept in `adj_close_vendor`.

---

## Assumptions

- **The supplied `ohlc` table is a trustworthy reference** for the trading calendar and the adjustment factors. Its raw
  OHLC and volume match `ohlc_hw` exactly on every row, after duplicates are removed. It is used only for those two
  purposes and never replaces our data.
- **Zero-volume days (75) are genuine no-trade days with stale prices.** They are kept and flagged rather than deleted.
  A *missing* volume (none in this data) would be filled with 0 and flagged as unknown, not treated as no trading.
- **The repaired opens are estimates**, not recovered values. The open is not used in any Part 2 metric.
- **Mergers and specific news events cannot be identified** without an external events feed. "Genuine, unconfirmed"
  means only that there is no evidence of an error.
- **The classification thresholds are judgment calls.** They are a 15% move, 90% reversal, volume under 0.5× normal,
  under 50% of tickers flagged, and a 1% factor step, and are documented in the notebook.
- **Annualization and rates:** 252 trading days per year, and a 2% annual risk-free rate for Sharpe (the course
  approximation).
- **22 of 30 tickers' data ends on 2024-10-18**, while the other 8 run to the window's last trading day (2024-11-29). I
  treat this as a data-delivery cutoff and do not extrapolate. Each report states its actual coverage.

---

## How to interpret the reports

**Each single-stock report** (`{TICKER}_report.ipynb`) shows:
1. **Coverage.** The requested window, the actual first and last dates, and a warning if the data is shorter. Always
   read the metrics against the actual dates.
2. **Data notes.** How many days in the window were repaired during cleaning or flagged for review.
3. **Charts:**
   - *Price*: red or green when rolling volatility is above or below its mean for the period (retrospective), gray during
     the warm-up, plus the 50-day moving average.
   - *Daily returns*: green up, red down.
   - *Rolling volatility*: the shading shows high and low regimes.
   - *Drawdown*: the loss from the running peak, with the deepest point marked.
4. **Summary table:**
   - *Total return*: start to end of the window.
   - *Annualized return* and *volatility*: what that pace would mean over a year (noisy for short windows).
   - *Sharpe*: excess return per unit of volatility. Above about 1 is strong for one stock over a few months.
   - *Maximum drawdown*: the worst peak-to-trough fall.
   - *Positive days*: the share of up days, with flat days counted in the total.

**`portfolio_summary.csv`** has one row per successful report, sorted by Sharpe ratio. All values are fractions
(0.05 = 5%).
- `avg_volatility` is the **mean of the 20-day rolling volatility**. It is not the same as the whole-period
  volatility in each report's summary table.
- `first_date`, `last_date`, `trading_days`, and `full_coverage` show what each row actually covers. Compare tickers
  with `full_coverage = False` (about 4.5 months of data) against full-coverage tickers (about 6 months) with care,
  because their total returns are over different lengths of time.

**`risk_return_profile.png`.** Points up and to the left are better risk-adjusted performers (higher return for less
volatility), and the color shows the Sharpe ratio. In this run, IBM, HD, and SHW had the highest Sharpe ratios, and
MRK, BA, and NKE the lowest. NVDA stands out at about 61% volatility.

These are descriptive statistics for one historical period, not forecasts or investment recommendations.

---

## Using Claude Code

I used Claude Code (Anthropic's coding agent) in the terminal throughout this assignment. I gave it the assignment one
requirement at a time, and it wrote and ran the code. I reviewed each result, asked follow-up questions, and decided
what to keep. Some decisions from that process:

- **Work one requirement at a time, and run everything.** Each step was executed and its output inspected before
  moving on, rather than generating the whole pipeline at once. Every detector and repair got a small injected
  test case with `assert`s, so a result of zero could be trusted.
- **Choose methods with evidence, not by default.** For the impossible-price repair, instead of picking one of the three
  suggested options, we benchmarked all of them on simulated errors. The result (option 2, clipped to the day's range) was
  better than any single option on its own.
- **Allow scope beyond the literal request when the data demands it.** Rebuilding missing `adj_close` exposed 10,417
  corrupted values that no required check catches. I accepted fixing them in the same step, because leaving them would
  have broken every adjusted return in Part 2. They are labelled separately so the decision stays visible.
- **Keep human judgment where rules fall short.** The MCD 1968 series break is a documented manual verdict. A generic
  rule both missed it and misfired on a real crash (AAPL 2000-09-29), so the evidence is shown and the call is explicit.
- **Check the AI's output instead of trusting it.** Several mistakes were caught by inspecting results, not by the
  first draft:
  - a rule that labelled a real crash day (UNH 2008-09-29) as a data error;
  - a repair order that could borrow a bad close;
  - a coverage warning that treated a Saturday as a missing trading day;
  - a chart axis with repeated labels.
  Each was fixed and re-verified.
- **Verify the environment, not just the code.** `requirements.txt` was tested by installing it into an empty
  environment and running the pipeline there, confirming that papermill's kernel used that environment.
- **Disclose rather than disguise.** The code and explanations were produced with AI assistance. The decisions above,
  and the responsibility for the submission, are mine.
