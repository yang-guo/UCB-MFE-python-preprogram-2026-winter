# HW2 — Scalable stock analysis pipeline

**Yixiao Yang · UCB MFE Python Pre-Program · Fall 2026**

This submission has five main files and a `reports/` directory. All functions are
defined directly in the notebooks or the batch script; no local helper modules,
external test folder or generated source files are needed to run the assignment.

| File | Role |
|---|---|
| `01_data_cleaning.ipynb` | Create a working database copy; inspect, diagnose, clean, validate, demonstrate edge cases, and persist results |
| `02_stock_report_template.ipynb` | Tagged parameters, SQL loading, documented risk functions, four-panel visualization, summary and Scrapbook outputs |
| `03_run_all_stocks.py` | Optional cleaning run; all-ticker Papermill batch, failure handling, rankings, CSV and portfolio chart |
| `requirements.txt` | Third-party Python dependencies |
| `README.md` | Run order, design decisions, assumptions and interpretation |
| `reports/2026-10-28/` | AAPL, IBM and NVDA executed examples plus the complete 30-stock summary CSV |

The three sample notebooks include their function definitions and actual outputs.
The full batch is generated locally; only the three requested examples and the
summary CSV are included in this submission. CSV paths for the other 27 stocks
resolve after generating the full batch.

## Run on the course database

Use Python 3.10+ (verified with Python 3.12). Put this directory at
`Homeworks/HW2/Yixiao_Yang/` in the course repository. Download the course database
from bCourses to `Lectures/Lecture 4/data/data.db`. From the homework directory:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python 03_run_all_stocks.py --clean --source-db "../../../Lectures/Lecture 4/data/data.db"
```

On Windows, activate with `.venv\Scripts\activate`. An alternative original database
location can be passed to `--source-db`. The original is opened read-only and copied
using SQLite backup to `data/data.db` if that working copy is absent. Existing local
working copies are reused, not overwritten by a new source. Source and target must
be different files. To start again with a different source, select a new local
`--db-path` within the homework directory.

`--clean` runs and saves the cleaning notebook, recreates `ohlc_hw_cleaned`, and
then generates every stock report. To repeat with the existing working copy:

```bash
python 03_run_all_stocks.py --clean
```

Alternatively, open `01_data_cleaning.ipynb` in Jupyter with this homework directory
as the working directory. Adjust its `source_db` parameter if needed, then Restart
& Run All. It creates the working copy itself. Run the report template interactively
or generate all reports with:

```bash
python 03_run_all_stocks.py
```

The script defaults to standard Papermill/Jupyter execution. No API key, external
market download or synthetic data fallback is used. Exit status is 0 for all
successful reports, 1 for partial failure and 2 for no successful reports. An empty
cleaned table fails before the batch starts.

## Observed cleaning results

| Check | Actual result |
|---|---:|
| Input rows / raw ticker labels | 397,711 / 150 |
| Normalized ticker identities | 30 |
| Exact duplicate records removed | 29,460 |
| Conflicting ticker/date keys | 0 |
| Rows quarantined | 5 |
| Validated rows retained | 368,246 |
| Missing adjusted prices repaired, after deduplication | 18,374 |
| Nonmissing inconsistent adjusted prices repaired, after deduplication | 10,417 |
| Total adjusted-price reconstructions | 28,791 |
| Raw OHLC prices changed in retained rows | 0 |
| Observed zero-volume rows retained | 74 |
| Missing sessions before cleaning, relative to per-ticker reference | 0 |

Before deduplication there were 19,855 missing adjusted-close cells and 11,265
reference-factor inconsistencies. The different counts reflect duplicate records.
The five quarantined rows are the first supplied observation for PG, CVX, MRK and
JNJ on 1962-01-02, and MCD on 1966-07-05. Their raw open is zero in both the homework
and reference tables, with no earlier same-ticker donor. No defensible open was
invented. They become five disclosed leading gaps after cleaning and do not affect
the 2024 report window.

The cleaned table has no missing/nonpositive/nonfinite prices, invalid OHLC bounds,
negative/nonfinite volume, duplicate keys or adjustment inconsistencies. Statistical
return and observed-zero-volume flags remain for review. Low-severity normalization
entries in the post-cleaning report preserve the label audit; they do not indicate
that retained ticker keys are still unnormalized. Input accounting reconciles:
397,711 = 368,246 retained + 29,460 removed + 5 quarantined.

## Cleaning decisions and assumptions

**Identity and duplicates.** Strip HTML/entities, whitespace and case before grouping.
Keep original labels/dates and a source-row identifier. Payload signatures are computed
before numeric parsing, so different invalid strings cannot become alleged exact
duplicates merely because both parse as NaN. Remove only exact canonical observations.
Quarantine all different payloads sharing a ticker/date key; do not arbitrarily keep
the first. An audit records every input row, including duplicates and quarantine reasons.

**Missing sessions.** Use each ticker's dates in the supplied `ohlc` table over its
full supplied history. This avoids inventing exchange-holiday gaps and covers entirely
missing tickers. It does not establish that the vendor itself contains every exchange
session. Tickers without a reference calendar are labeled unassessed. Missing dates
become issue rows; no rows or prices are fabricated. The notebook includes a deliberately
removed-date example and reports zero gaps when that is what the data show.

**Prices.** Drop all-raw-OHLC-missing rows before any fill. Replace missing/invalid scalar
raw prices using the preceding validated retained observation of the same ticker.
Reject unresolved leading gaps and values outside today's range. For a contradictory
close, same-day H/L midpoint repair is allowed only when the trusted same-date reference
independently corroborates O/H/L. Otherwise quarantine. Recheck positive finite prices
and both raw open/close bounds after each repair. Forward fills and midpoints are
estimates, flagged as such; they can reduce measured volatility.

**Adjusted prices.** The reference is assumed trustworthy for a matched date-specific
factor `ohlc.adj_close / ohlc.close`. Rebuild adjusted close as validated raw close × this
factor, including after a close repair. Never carry a factor across dates or a possible
corporate action. Conflicting reference keys supply no factor. If there is no trusted
factor, retain an originally valid vendor pair only when raw close is unchanged and
label its factor unverified. Otherwise quarantine. Do not substitute the reference
dataset for the dirty delivery.

**Outliers.** Diagnose simple raw-close returns within sorted tickers with
`abs(return - mean) > 3 * sample_std`, both tails, `ddof=1`. No implicit price fill is
used for diagnostic returns. Undefined/zero standard deviation gives no flag. A
statistical flag alone is not a deletion or interpolation rule. If the raw close agrees
with the trusted same-date reference, retain it, without claiming a particular split
or news event. A reference disagreement is quarantined because the exact repair is
not justified. With no reference evidence, retain only if structural validation passes
and label the event uncertain. `outlier_review.csv` records the evidence. The small
confirmed-error example demonstrates interpolation within valid neighbors and no edge
extrapolation; the real pipeline does not interpolate unverified extremes.

**Volume.** Missing, negative and infinite volume becomes zero with `volume_unknown`.
An unknown placeholder is distinct from an observed zero. Genuine zero volume stays
as a review flag. Volume does not affect adjusted-close risk metrics here.

**Accounting and validation.** Retained + removed + quarantined = input rows. Save
repair flags with `ohlc_hw_cleaned`; audit/issue tables and CSVs preserve the initial
diagnosis and post-cleaning checks. SQL `ts` is always `YYYY-MM-DD`. Re-read persisted
data, validate it, and check the source table is unchanged. All retained prices are
finite and positive, volume is finite/nonnegative, and keys are unique.

## Financial conventions

| Output | Definition |
|---|---|
| Daily return | `log(adj_close / previous_adj_close)` |
| Rolling volatility | 20-return sample SD × √252; needs 21 prices |
| Drawdown | Adjusted price / running peak − 1 |
| Maximum drawdown | Minimum per-date drawdown; not a cumulative-return minimum |
| Sharpe | `(252 × mean(log return) − 0.02) / (sample_sd × √252)` |
| Total return | Last / first adjusted close − 1 |
| Annualized return | `expm1(sum(log_returns) × 252 / number_of_returns)` |
| Annualized volatility | Whole-period sample SD × √252 |
| `avg_volatility` in batch CSV | Mean of 20-day rolling annualized volatility |
| Positive days | Strictly positive returns / all nonmissing returns, including zero |
| Best/worst and up/down days | Daily log-return statistics; empty sign subsets are N/A |

Fractions stay numeric in calculations and CSVs; percentages are display formatting
only. Undefined/zero-volatility Sharpe is NaN, displayed as N/A and excluded from
rankings. The first missing return is never zero-filled. MA50 needs 50 prices. Prices
may bridge an unresolved missing session, so the 252-session annualization assumption
and comparisons can be distorted by gaps; inspect the cleaning diagnostics.

## Batch outputs and interpretation

The window is the six calendar months **before** the month containing `MAX(ts)` in
`ohlc_hw_cleaned`, even when that latest observation is month-end. December 6 or
December 31, 2024 both imply June 1–November 30, 2024. The computer clock does not choose
this historical window. The output folder uses the actual run date in Asia/Shanghai:
`reports/YYYY-MM-DD/`.

The report titles and CSV show requested and actual dates/counts. Histories need not
end together. A stock with fewer than 21 prices fails clearly; a stock with 21–49
prices is shown with no invented MA50. No trailing extrapolation is performed.
Raw returns may jump at a split while adjusted returns remain smooth.

The batch collects `report_metrics` through Scrapbook only after execution and
identity/run-ID checks succeed. Same-day output directories are archived before a
new run. No wildcard collection of old notebooks is performed. Every ticker has a
try/except boundary. If all fail, the manifest and failure logs are written, but no
pretend portfolio summary is created.

Files in the dated directory include successful/failed executed notebooks,
`portfolio_summary.csv`, `portfolio_dashboard.png`, `failed_stocks.txt`,
`failed_stocks.json`, `batch.log`, and `manifest.json`. The CSV's relative report paths
resolve from its own directory. Top/bottom 10 use defined Sharpe only and may overlap.
The scatter X-axis is **mean rolling** volatility; Y is total return; color is Sharpe.
Undefined Sharpe is gray, and up to five highest-return stocks are labeled. This is
a cross-stock dashboard, not a calculation of a weighted portfolio's risk.

## Verification and execution environment

The compact implementation was rerun against the supplied real database. All 30
reports completed; all 210 independently checked numeric summary values matched
the earlier verified run. All 26 behavioral tests passed: the original 24 checks now exercise the
embedded functions, and two additional checks verify source-file protection and
working-copy reuse. The cleaning notebook's 14 injected demonstrations passed. The test harness is a development tool and is not needed to run this
self-contained submission. Report notebooks also execute hand-calculated checks.

This preparation environment prohibits local Jupyter kernel sockets. These saved
notebooks therefore ran through a Papermill engine executing real cells in IPython
and capturing actual outputs, including figures and Scrapbook records. They are
labeled accordingly in notebook metadata. Standard Jupyter transport was not
verified in this environment; the submitted batch defaults to standard Jupyter.

The source database SHA-256 remained unchanged. The requested report window is
2024-06-01–2024-11-30: 22 stocks have 97 prices ending 2024-10-18; eight have 126
ending 2024-11-29. Each matches its supplied reference calendar in this window.
This unequal coverage limits direct rankings.

## Submission

Submit these five main files and the four files in `reports/2026-10-28/` under
`Homeworks/HW2/Yixiao_Yang/`, via a pull request to the course repository. The three
examples are `AAPL_report.ipynb`, `IBM_report.ipynb` and `NVDA_report.ipynb`; the
fourth file is `portfolio_summary.csv`. Preserve the folder structure.

Do not commit databases, credentials, `.venv/`, `outputs/`, report archives or the
other 27 generated reports. Rerunning creates local artifacts that are deliberately
excluded from this prepared package. Course source files are unchanged.

## Course sources

Implementation was checked against the uploaded `hw2.md` and the repository at commit
`acb047c8367794102c31ccbea8efc19ad0f54940` (retrieved 2026-10-28). The uploaded assignment
and repository assignment differ only by a trailing blank line.

- [HW2 requirements](https://github.com/yang-guo/UCB-MFE-python-preprogram-2026-winter/blob/acb047c8367794102c31ccbea8efc19ad0f54940/Lectures/Lecture%202/hw2.md)
- [Lecture 2: pandas and portfolio analysis](https://github.com/yang-guo/UCB-MFE-python-preprogram-2026-winter/tree/acb047c8367794102c31ccbea8efc19ad0f54940/Lectures/Lecture%202)
- [Lecture 4: cleaning, SQL, report and Papermill notebooks](https://github.com/yang-guo/UCB-MFE-python-preprogram-2026-winter/tree/acb047c8367794102c31ccbea8efc19ad0f54940/Lectures/Lecture%204)
- Lecture 4's guide and optional reference notebooks were also inspected. Where an
  older optional example differs, the current HW2 contract takes precedence (e.g.
  no arbitrary key deduplication or automatic clipping of 3σ returns).
- The supplied course transcript's HW2 discussion confirms the Lecture 2 + Lecture 4
  scope and the use of one parameterized executed notebook per stock.

## 中文说明

提交目录为五个主要文件加 `reports/` 文件夹。辅助函数已经直接合并到 Notebook
和批量脚本中，不再依赖单独的 Python 模块或 tests、validation 文件夹。
先安装 requirements.txt 中的依赖，再执行上面的 `--clean --source-db` 命令，
即可从原始数据库副本重新完成清洗和全部股票报告。

只上传本包中的文件；本地运行生成的数据库和完整结果不需要提交。
报告目录中的日期是运行日期，分析区间由数据库最新日期自动推导。
