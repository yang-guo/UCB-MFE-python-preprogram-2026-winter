# Homework 2: Scalable Stock Analysis

**Author:** Sterling Couts

This pipeline cleans the supplied market-data delivery, produces parameterized single-stock risk reports, and runs those reports for every cleaned ticker. The assignment in `Lectures/Lecture 2/hw2.md` defines the conventions.

## Setup and run order

1. Download `data.db` from bCourses/Canvas and place it at `Lectures/Lecture 4/data/data.db` in the course repository. Extract it first if downloaded as a ZIP. The source must contain `ohlc_hw` and `ohlc`.
2. Use the course Python environment (the repository recommends `uv sync --locked`) or install this homework's dependencies into an environment with Python 3.12 or later:

   ```text
   python -m pip install -r requirements.txt
   ```

   Development used Python 3.14.8, pandas 2.3.3, NumPy 2.3.5, Matplotlib 3.10.7, Papermill 2.6.0, and Scrapbook 0.5.0. Select the same environment as the notebook kernel and for script execution.
3. Set the notebook working directory to `Homeworks/HW2/Sterling_Couts`. Restart the kernel and run all of `01_data_cleaning.ipynb`. It locates the course source relative to the checkout, opens it read-only, creates a separate local `data/data.db`, and recreates `ohlc_hw_cleaned`. It also writes local audit CSVs.
4. Run `02_stock_report_template.ipynb` to inspect the default AAPL report. Its tagged `parameters` cell supplies `ticker`, `start_date`, `end_date`, and `db_path`. Relative `db_path` resolves from the notebook working directory. A report needs at least 21 prices; 21–49 prices can support volatility but not the 50-price moving average.
5. From this homework directory, run:

   ```text
   python 03_run_all_stocks.py
   ```

   The script locates files relative to its own location, so it also works when invoked by absolute path from another directory. It supplies an absolute database path to each report and uses the script's interpreter for child kernels. Outputs go to `reports/YYYY-MM-DD/`, using the computer's local run date.

The batch deliberately refuses to overwrite a nonempty same-date output directory. Preserve a previous run by moving its folder under local `work/` before rerunning. For the included sample date, the samples also make that directory nonempty. Do not delete prior outputs merely to bypass the check. No source database or lecture notebook is modified.

## Cleaning decisions and audit

- Strip the observed simple HTML tags, trim whitespace, and uppercase ticker labels before grouping or deduplication. Preserve original labels and a normalization mapping. AMGN and AMZN remain distinct securities.
- Remove exact duplicates using the eight original fields with normalized ticker; exclude the audit label from equality comparisons. Investigate conflicting ticker-date keys separately. This delivery had no remaining conflicts.
- Use each ticker's supplied `ohlc` dates as its expected calendar. The original normalized delivery has no missing reference sessions. This cannot detect dates absent from both sources.
- Assess missing, nonfinite and nonpositive prices, raw OHLC bounds, volume, and within-ticker two-sided three-sigma simple raw-close returns (`ddof=1`). Preserve a multi-issue quality report before and after cleaning. Severity labels distinguish invalid required values from review flags.
- Quarantine five leading zero-open observations. These same invalid opens occur in the reference, and no prior observation supports forward filling. Save reasons rather than invent prices.
- Reconstruct only affected adjusted closes from validated raw close times the same-date reference factor `ohlc.adj_close / ohlc.close`. This explicitly trusts the supplied reference adjustment convention. It does not verify market truth independently or carry factors across unverified corporate actions. Preserve original adjusted values and repair flags.
- Retain statistically extreme raw returns with review flags. Finite raw OHLC agrees with the supplied reference; the inspected AMZN jump is internally consistent. No news/split explanation or raw-close error is asserted without evidence. Statistical flags alone do not justify interpolation or deletion.
- Synthetic cases demonstrate sorted within-ticker forward fill, removal of all-raw-missing rows before fill, missing volume replaced by zero with a flag, unresolved leading gaps, and invalid OHLC after filling. A separate confirmed-error interpolation example uses interior valid neighbors, does not extrapolate edges, and checks OHLC and date-specific adjusted-price consistency. These examples do not modify the real delivery.

Observed reconciliation:

| Category | Rows |
|---|---:|
| Original delivery | 397,711 |
| Extra exact duplicates removed | 29,460 |
| Quarantined | 5 |
| Retained cleaned observations | 368,246 |
| Missing adjusted closes repaired | 18,374 |
| Inconsistent adjusted closes repaired | 10,417 |

The two repair counts describe retained-row changes, not additional row removals. The final report contains five explained reference-date gaps, 74 reported zero-volume review flags, and 5,057 extreme-return flags. Original flags numbered 5,056; recalculation after exclusions changes the MCD classification on June 21, 1989. Raw data remains unchanged. SQLite dates are stored as `YYYY-MM-DD`.

## Risk conventions and interpretation

- Reports use adjusted close. Daily risk and daily summary statistics use log returns; Part 1's simple raw-close returns are diagnostic.
- Rolling volatility is the sample standard deviation of 20 log returns times `sqrt(252)`; it needs 21 prices. The 50-price average needs 50 prices. Warm-up gaps remain missing.
- Total return is last/first adjusted close minus one. Annualized return is `exp(mean(log_returns) * 252) - 1`. Whole-period annualized volatility differs from the mean rolling volatility exported to the portfolio summary.
- Sharpe follows the course approximation `(252 * mean_log_return - 0.02) / (sqrt(252) * sample_std)`. Insufficient returns or numerically zero volatility produce N/A. Roundoff-scale standard deviations are treated as zero.
- Drawdown is price/running peak minus one; its minimum is maximum drawdown. The plotted series can recover.
- Best/worst days and average up/down days are log returns. Positive-day proportion includes zero-return days but excludes undefined returns. Empty conditional subsets display N/A.
- CSVs and named exports retain fractional units. Percentage formatting is for presentation only. Price colors use the report's mean rolling volatility retrospectively; they are not trading signals.

The latest cleaned observation is December 6, 2024. The batch therefore requests June 1–November 30, 2024: the six calendar months before the month containing the latest observation, even if that observation falls on month-end.

Actual coverage starts June 3 and ends either October 18 (97 prices) or November 29 (126 prices). Do not interpret Sharpe or return rankings as strictly matched-period comparisons. The scatter compares individual stocks, not the return/risk of a weighted investment portfolio. The displayed estimates describe historical observed samples, not forecasts.

## Automation, outputs, and tests

Each ticker runs in a fresh Papermill kernel. Scrapbook stores named results, actual dates and observation counts. A report is collected only after execution and required metric extraction succeed. The batch logs failures, continues, and produces no fabricated summary if every report fails. Undefined Sharpe values are excluded from rankings; top and bottom groups may overlap in small successful batches.

The full local run produced 30/30 successful reports. Independent calculations from SQLite reproduced all 150 numerical entries in the seven-column portfolio CSV. Tests cover hand-worked risk calculations, constant-return Sharpe, rolling warm-up, six-month windows across year/month-end boundaries, real invalid-ticker failure followed by a successful stock, and a simulated all-failed batch. Local `00_pipeline_checks.ipynb` records development checks but is not required to regenerate the submission. Scrapbook renderer notices in VS Code do not imply missing exports; saved metrics were read back successfully.

The submission includes three executed examples: AAPL (shorter coverage), IBM and NKE (fuller supplied coverage), plus the full-universe `portfolio_summary.csv`. The complete batch remains local. CSV report paths for the other 27 stocks are populated when the batch is regenerated; those notebook files are intentionally not submitted.

Generated `report_coverage.csv`, `portfolio_summary.png`, failure logs and manifest are available locally and recreated by the script. The local `.gitignore` limits submitted reports to the three examples and summary CSV, and excludes databases, audits, temporary kernels, and development outputs. Never commit database copies or credentials.
