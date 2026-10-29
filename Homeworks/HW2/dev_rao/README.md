# HW2 - stock analysis pipeline (Dev Rao)

Clean `ohlc_hw` -> parameterized risk report for one stock -> run it for every ticker with papermill and collect a summary.

## files

| file | what it is |
|---|---|
| `01_data_cleaning.ipynb` | inspection, quality report, repairs, validation, saves `ohlc_hw_cleaned` |
| `02_stock_report_template.ipynb` | papermill template (tagged `parameters` cell: `ticker`, `start_date`, `end_date`, `db_path`) |
| `03_run_all_stocks.py` | runs the template for all tickers over the last 6 complete months, builds the summary |
| `reports/2026-10-30/` | sample output: AAPL, IBM, MRK reports + `portfolio_summary.csv` + `risk_return_profile.png` + `failed_stocks.txt` (empty, nothing failed) |
| `requirements.txt` | packages used |

The full batch makes 30 reports. I only committed 3: AAPL (history ends early, shows the coverage warning), IBM (best Sharpe) and MRK (worst Sharpe). The rest are gitignored.

## how to run

Setup, from the repo root:

```
uv sync --locked          # or: pip install -r Homeworks/HW2/dev_rao/requirements.txt
```

Put `data.db` from bCourses either at the repo root or in `Lectures/Lecture 4/data/`. Both work, and it's gitignored either way.

Run order:

1. `01_data_cleaning.ipynb`, run all. The first time, it copies the db to `Homeworks/HW2/dev_rao/data/data.db` (my local working copy, gitignored). Then it writes `ohlc_hw_cleaned` (plus `ohlc_hw_quarantine` and `ohlc_hw_quality_report`) into that copy. The original db and the raw `ohlc_hw` table are never written to.
2. `02_stock_report_template.ipynb` (optional): run it by itself to see one report. The defaults are AAPL 2024-01-01 to 2024-12-31, and `db_path` is `data/data.db` relative to this folder.
3. `python Homeworks/HW2/dev_rao/03_run_all_stocks.py` with the venv's python. It works from any folder since paths are based on the script location. Options: `--tickers AAPL MSFT` for a quick test, `--db <path>` for another db. Heads up, a `--tickers` test writes into the same date folder, so it overwrites that day's summary csv. Rerun without it after testing.

Output goes in `reports/{today's date}/`. That's the run date, not the analysis window.

## what I found in the data and what I did

Checked before cleaning (numbers are from the notebook, not hardcoded):

- tickers: 150 raw strings (html like `<a ...><span ...>AAPL</span></a>`, spaces, case) -> 30 real tickers. I normalized them **before** any grouping/deduping, otherwise one stock gets split into 5. The original string is kept in `raw_ticker`.
- duplicates: 29,460 exact duplicate rows dropped. 0 conflicting (ticker, ts) keys, so I didn't need a "which row wins" rule.
- missing cells: only in `adj_close` (18,374 after dedup). Raw OHLC and volume have none.
- missing trading days: 0, using each ticker's dates in `ohlc` as the calendar. I tested the detector on an injected gap (removed one AAPL date) and it finds it.
- corrupted adj_close (~10.4k rows): not on the checklist but it's the biggest problem in the data. adj_close was 4x to 24x too big on random rows (describe() shows max adj_close ~10,442 vs max close ~605). I caught them with the adjustment factor `adj_close / close`. It should be flat apart from steps at splits/dividends, so a row whose factor is way off the rolling median of its neighbours (log diff > 0.5) gets flagged. I checked this against `ohlc` afterwards: it caught all of them with no false positives.
- open = 0 on 5 rows: the very first row of PG, CVX, JNJ, MRK (1962-01-02) and MCD (1966-07-05). These also break `low <= open`.
- zero volume: 75 rows, all 1962-1982 (mostly WMT).
- high < close / low > close, negative or inf prices, missing volume: none in the real data. Each check is shown working on a small made-up example in the notebook.

### repairs

- adj_close (missing or corrupted): rebuilt as validated raw close x the factor from the **same (ticker, ts)** row in `ohlc` (`ohlc.adj_close / ohlc.close`). I only did this where the ohlc close matches our raw close, so the factor belongs to the same price. Never forward-filled a factor, since that could carry it across a split. This is a per-row fix of flagged rows, not a swap of the whole table. Flag: `adj_close_repaired`, `adj_close_source = ohlc_factor_x_close`.
- invalid raw price: set to NaN, then use the last known close if it's inside today's [low, high], otherwise the midpoint of high/low (option 1 with a bounds check, because plain ffill can land outside today's range). The 5 open=0 rows are all first rows, so there's nothing earlier to use. They're **quarantined** (saved to `ohlc_hw_quarantine` with a reason) instead of making up a price.
- missing/negative volume: would go to 0 with `volume_repaired = True`, because unknown volume isn't the same as no trading. There were none, so it's only shown on a demo. The real zero-volume days are kept as they are with `zero_volume_flag`.
- dropping rows where all OHLC are missing is done first (0 rows).

Row accounting: 397,711 raw = 29,460 duplicates + 0 all-missing + 5 quarantined + 368,246 kept.

After repairs I rerun all the checks (no missing/nonpositive/inf prices, OHLC bounds hold, unique keys, volume >= 0, no factor jumps left) and the table only saves if they pass. Then I read it back from the db and check again.

## outliers, and why I handled them this way

The 3 sigma screen (simple returns on raw close, per ticker, `ddof=1`, both tails, first return excluded) flags ~5,000 days. Over 60 years of daily data that's expected because returns have fat tails. So it's a **review flag** (`extreme_return_flag`), not a delete rule.

I went through the biggest ones by |z|:

- TRV 1976-10-28, 1978-12-20, 1981-08-10: close exactly doubles, then drops right back the next day. O=H=L=C and volume is 0 on all three. A split doesn't undo itself the next day, so I treat these as data errors and interpolate O/H/L/C/adj_close as the midpoint of the day before and after (`spike_repaired`). The rule is |return| > 25% and the next close back within 5% of the pre-spike close, which catches exactly these 3. The reference `ohlc` table has the same spikes, so I couldn't just look up the right value there.
- AAPL 2000-09-29 (-52%), NVDA 2000-03-07 (+42%): big volume and the price stays at the new level. Nothing says data error, so I kept them and they stay flagged. I'm not claiming a specific news event, I just didn't find evidence that it's wrong.
- MCD 1968-01-02 (-75%): also persists with high volume. Could be a split the vendor didn't adjust for, but I don't have a source to confirm that, so I kept it flagged and noted the uncertainty. It's way outside the report window anyway.

Basically: only fix what I can actually show is wrong, flag the rest, quarantine what can't be repaired.

## assumptions

- `ohlc` dates per ticker are the trading calendar (allowed by the hw). I didn't use weekdays, since holidays would show up as fake gaps.
- `ohlc` adjustment factors are trusted per date. The only spots where they looked bad were the 3 TRV spikes, which I handled separately.
- Window = the 6 calendar months before the month of the latest date in `ohlc_hw_cleaned`. Latest is 2024-12-06, so the window is **2024-06-01 to 2024-11-30**. Same window for every ticker.
- **22 of the 30 tickers end on 2024-10-18** (only AMZN, CVX, DIS, IBM, KO, MRK, NKE, PG go to Dec). Those reports cover about 97 days instead of 126, which isn't the same period. Each report prints its actual first/last date (in red in the figure subtitle if it's short), the CSV has `first_date`, `last_date`, `trading_days` and `short_coverage`, and the scatter uses triangles for them. I didn't extend prices past where the data ends.
- Report metrics follow the hw: adjusted close, log returns, sample std, 252 days, 2% risk-free rate, NaN/"N/A" when something is undefined, warm-up NaNs not filled.
- The summary CSV has the 7 required columns plus 4 coverage columns at the end. All values are fractions (0.21 = 21%).

## how to read the reports

Each report has the coverage line, input checks, a 2x2 chart and the 10-metric table.

- price panel: red where the 20-day vol was above its average for the window, green where it was below, gray at the start where there aren't 20 returns yet. The dashed line is the 50-day MA, which only starts after 50 prices.
- returns panel: daily log returns. Look for clusters of big bars.
- volatility panel: red shading = riskier than usual for that stock, green = calmer.
- drawdown panel: how far below its previous high the stock was on each day. The dot is the max drawdown.
- table: total return is just last/first price. Annualized return roughly doubles a 6-month number, so it looks big. Sharpe uses 2% risk-free, and on 6 months of data it's pretty noisy, so I wouldn't rank stocks on it too seriously. That goes double for the ones with shorter coverage.

In the batch, IBM had the best Sharpe (~3.2, +40%) and MRK the worst (~-1.9, -20%). NVDA is the obvious high-vol outlier (~61% avg vol) on the risk/return plot.
