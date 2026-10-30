# October 15: Clean the data, explain the risk, repeat the report

Yang · 9am–noon Pacific · second data-analysis session. Plan for **150 minutes including a ten-minute break**, leaving the remainder of the scheduled slot for questions and setup delays.

The deferred visualization/cleaning module from October 8 is **first**. The sequence is now `1_visualize_and_clean`, `2_sql_and_data_quality`, `3_stock_analysis_template`, `4_running_analysis_at_scale`. Four short modules preserve the moved notebook and the existing independent report template; do not introduce a fifth notebook.

## Teaching goals and constraints

The syllabus emphasizes Python proficiency, pandas/Matplotlib/SQL, independent development, and the ability to read, test, and improve AI-generated code. Earlier course planning established a beginner-to-moderate MFE audience, finance-relevant examples one step beyond toy syntax, and notebooks as the complete material for a cell-by-cell walkthrough. Preserve that approach: students should explain what a table or chart means, change an input, and check the result.

Worked examples stay in the explanation; exercises with different inputs stay at the end. Student notebooks have no timetable or instructor delivery notes. Timing and teaching choices live here. Advanced Python and productionization are later modules, so kernel configuration, framework design, concurrency, and elaborate plotting machinery are not learning objectives tomorrow.

## Live route: 2.5 hours

| Time (PT) | Module | Focus and boundary |
|---|---|---|
| 9:00–9:55 | [1. Can we trust this price chart?](1_visualize_and_clean.ipynb) | Returns and normalized plots; vendor text; issue table; calendar gaps; defensible repairs; adjusted prices |
| 9:55–10:15 | [2. Ask the database a precise question](2_sql_and_data_quality.ipynb) | Inspect actual HW data, SELECT/WHERE/ORDER BY, parameters, one pandas check, save/read a table |
| 10:15–10:25 | **Break** | **10 minutes** |
| 10:25–11:05 | [3. Turn prices into a report you understand](3_stock_analysis_template.ipynb) | Simple vs log returns, rolling windows, drawdown, Sharpe, summary metrics, four panels |
| 11:05–11:25 | [4. One report becomes three](4_running_analysis_at_scale.ipynb) | Date window, parameterized execution, failure handling, collect/rank successful results |
| 11:25–11:30 | HW2 handoff | Three deliverables, actual data quirks, conventions, and run order |

**140 instructional minutes + 10-minute break = 150 minutes.** This is a selective walkthrough, not time to dissect every helper or complete every exercise. All required HW2 patterns are in the four main notebooks; students revisit the longer examples when implementing their own notebooks/script. The scheduled 11:30–noon remainder is useful contingency, not an assumed extra lecture block.

### What to emphasize, and where to save time

1. **Cleaning gets the largest block.** Start with “Does the apparent winner survive checking?” Run the generator as supplied; discuss its inputs and output, not its stochastic construction. Spend about 10 minutes on plots/returns, 10 on vendor labels and duplicate keys, 10 on gaps and statistical flags, 20 on repairs/adjustment/validation, and 5 on interpretation. Explain the issue-report function as masks → selected rows → concatenation; do not dictate it line by line. Keep string cleaning, but explain what the regex extracts rather than every metacharacter. The small missing-volume recap and full post-repair chart can be run with a brief interpretation.
2. **SQL is a bridge into analysis.** Show that `ts` is the date column and `adj_close` is actually missing. Preserve SELECT/WHERE/ORDER BY, placeholders, the SQL-versus-pandas check, and `to_sql` round-trip. Briefly identify GROUP BY; the dated JOIN is a worked reference if the clock is tight, since joins were already taught on October 8 and HW2 does not require a market-cap join. No CTE/cursor tour.
3. **Teach the report's meanings before its styling.** Check one return and one drawdown by hand; explicitly distinguish simple/log returns and whole-period/rolling volatility. Sharpe and annualization are now explained, not unexplained homework additions. Trace one panel's conditions and labels; run the rest as variations on the same Figure/Axes pattern. Preserve gray warm-up segments and honest coverage labels. Yang–Zhang is optional.
4. **Keep automation to a short data-analysis workflow.** Trace one call, then the loop and the deliberate failure. Inspect the named metrics and summary, not interpreter registration. The three-stock batch is enough to understand the all-stock assignment; the quarterly batch remains off. If report execution is slow, inspect the preflight's saved reports while the live batch runs.
5. **Close with the homework contract.** `ohlc_hw` → validated `ohlc_hw_cleaned` → parameterized report → all-stock script → successful-results summary. Students must explain their cleaning decisions and reproduce their outputs. Do not spend this block on new APIs.

If running behind, shorten detailed vendor parsing, the SQL join explanation, and chart cosmetics first. Preserve grouped/time-ordered operations, missing-cell versus missing-row reasoning, adjusted prices, log-return/Sharpe conventions, and the deliberate batch failure. Assign final exercises for independent practice. A student explaining one changed input is stronger evidence than a successful Run All.

## Setup and actual data findings

Run `uv sync --locked` at the repository root and select `.venv`. Download `data.db` from bCourses into `Lectures/Lecture 4/data/`. The opening module uses seeded synthetic data and needs no database, API, or credentials. The other modules open the source read-only; examples write to ignored `outputs/`.

The local supplied database was inspected for this review:

- `ohlc_hw` has 397,711 rows and 150 raw ticker strings. Stripping markup/whitespace and normalizing case produces 30 ticker identities.
- After normalization, 29,460 rows are exact duplicates. Removing them leaves unique ticker/date keys with the same date coverage as `ohlc`.
- The 19,855 null cells in the raw homework table are in `adj_close`; raw OHLC has no nulls. These counts are before deduplication and should not be used as grading constants.
- No dates are absent relative to each ticker's supplied `ohlc` keys after normalization/deduplication. Students must still implement the check and test a deliberately removed date. Holidays must not become invented price observations.
- Histories differ: the overall database ends December 6, 2024, while AAPL/MSFT/NVDA end October 18. The six-month assignment window is June 1–November 30, but requested coverage is not observed coverage. Keep actual dates/counts visible and qualify cross-stock comparisons.

These observations motivated clarifications in HW2: adjusted-price repair, reference-calendar assumptions, exact versus conflicting duplicates, and zero-issue results. Do not tell students that every listed fault must appear. The provided `ohlc` table may supply reference dates and trusted adjustment factors; replacing the entire dirty dataset is not a cleaning analysis.

The live report/batch retains January–June 2024 so all three examples have sufficient history. Q4 2024 contains only 14 observations for these examples and cannot support a 20-return window. The optional quarterly extension uses complete 2023 quarters.

## HW2 coverage audit

**Before this revision, Lectures 2 + 4 did not fully support HW2.** Calendar detection, the required issue table, the precise 3σ screen, adjusted-price repairs, Sharpe/annualized return, several plotting requirements, and the executable six-month rule were missing or only mentioned. The following map identifies the repaired coverage. No advanced reference notebook or later lecture is a prerequisite for a required HW2 deliverable.

| HW2 requirement | Where the student learns the pattern | Transfer to the homework |
|---|---|---|
| 1.1 Load, inspect types/statistics, dates, tickers | Module 2: “Inspect the homework delivery” and parameterized queries; Lecture 2 pandas basics | Load full `ohlc_hw`, describe columns, normalize keys before grouping |
| 1.2 Missing cells versus missing sessions | Module 1: `.isna()`, `missing_sessions`, explicit session example; module 2 reference keys | Compare per-ticker expected dates; return zero honestly; test an injected gap |
| 1.2 Simple returns and 3σ flags | Module 1: grouped `pct_change`, mean/std transforms in `price_issue_report` | Both tails, within ticker; review rather than automatically delete |
| 1.2 OHLC, nonpositive/nonfinite values, volume | Module 1 issue masks; module 3 `quality_flags` | Check all raw bounds plus adjusted-close validity and zero-volume review |
| 1.2 Required quality-report schema | Module 1: `quality_report` and `gap_issues` | Extend descriptions, severities, and issue types; retain original diagnosis |
| 1.3 Clean identifiers and duplicate records | Module 1: vendor export, normalized keys, exact versus conflicting duplicates | Apply to actual HTML-wrapped ticker strings before any time-series step |
| 1.3 Drop all-missing OHLC, grouped fill, interpolate | Module 1: “Make the repair explicit” and “Choose a repair” | Preserve flags; quarantine unresolved edge gaps; use future neighbors only retrospectively |
| 1.3 Adjusted prices and validation | Module 1: split/factor example and assertions; module 3 raw/adjusted checks | Use trustworthy factors, validate repaired OHLC, account for removed rows |
| 1.3 Before/after and persisted table | Module 1 raw/clean plots and flags; module 2 `to_sql`/read-back | Compare statistics, save `ohlc_hw_cleaned` in a local copy, retain audit columns |
| 2.1 Tagged parameter cell | Module 3 parameters and metadata explanation; module 4 one run | Use ticker/start/end and a portable database path |
| 2.2 Filter cleaned data, validate, show head/tail and coverage | Module 2 SQL; module 3 loading and quality checks | Change table to `ohlc_hw_cleaned`; reject insufficient history clearly |
| 2.3 Log returns, 20-day volatility, drawdown, Sharpe | Module 3 explicit functions, tiny-price checks, Sharpe explanation | Preserve units and window warm-up; `.min()` gives maximum drawdown |
| 2.4 All four panels and styling | Module 1 Figure/Axes; module 3 16×12 chart | 50-day mean; volatility-colored line; signed bars; mean/shading; gradient and worst marker; grids/labels |
| 2.5 All ten summary metrics | Module 3 `summary_metrics` and conditional-return examples | Annualized compounded return, whole-period volatility, nonmissing denominators, N/A cases |
| 3.1 All tickers and six complete months | Module 4 DISTINCT ticker, MAX date, `six_complete_months` | Query the cleaned table; use one common requested window and disclose actual coverage |
| 3.1–3.2 Script, folders, progress, failures | Module 4 runner, `tqdm`, try/except, both failure logs, “Move the batch into a Python script” | Adapt local paths/names; guard every call; handle all-failed batch |
| 3.3 Seven summary columns, ranking and CSV | Module 3 named exports; module 4 successful-path collector and `portfolio_summary` | Use fractional metrics; rank defined Sharpe values; exclude failed/stale reports |
| 3.3 Risk/return scatter, Sharpe colorbar, top-five labels | Module 4 comparison figure | Apply to all successful stocks; qualify unequal observed coverage |
| Submission: dependencies, README, examples | HW2 checklist; module 4 script/run-order guidance; Lecture 1 Git/PR workflow | Reproduce cleaned table; commit 2–3 reports and CSV, no database/credentials |

HW2 now explicitly distinguishes simple cleaning returns from log report returns, per-date drawdown from maximum drawdown, whole-period volatility from mean rolling volatility, and output run date from historical analysis dates. Its three deliverables and **October 29, 2026** due date remain unchanged.

## Formative checks to preserve

- Why can normalized prices change the apparent ranking?
- Why does cleaning ticker strings have to precede grouped returns and duplicate detection?
- Can a table contain no nulls and still omit a trading session?
- Why is a statistical outlier not automatically a data error? What source supports this repair?
- Why can raw close halve with no adjusted-price loss?
- For 80 → 100 → 70 → 85, why are the worst drawdown (−30%) and final drawdown (−15%) different?
- Why do 20 returns require 21 prices, and why is the first missing return not a zero day?
- Does NVDA still run after `NOT_A_TICKER` fails? Why is a saved file not proof of success?
- Does requesting June–November guarantee that AAPL has November observations?

An optional AI-tutor prompt can ask for an explanation or a changed-input example; students should predict and verify it themselves. This carries the syllabus's AI fluency goal without depending on unpublished Lecture 3 material.

## Optional references

The [advanced outlier reference](reference/1_advanced_outlier_detection.ipynb), [ingestion reference](reference/2_data_ingestion_and_sql.ipynb), [full cleaning case](reference/3_real_world_data_challenge.ipynb), Yang–Zhang appendix, and quarterly batch remain optional. The main notebooks now contain the required HW2 preparation.
