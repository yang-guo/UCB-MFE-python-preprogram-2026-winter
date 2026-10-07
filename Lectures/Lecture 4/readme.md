# October 15: SQL, data quality, and automated reports

Yang · 9am-noon Pacific · second of two data-analysis sessions.
The course schedule calls this **Data Analysis III · SQL**.

## Live modules

Each notebook runs independently from a fresh kernel; no notebook depends on variables left over from another module.

| Time (PT) | Module | Main outcome |
|---|---|---|
| 9:00-10:25 | [SQL and data quality](1_sql_and_data_quality.ipynb) | Read CSV/JSON, query and join SQL tables, validate OHLCV, flag outliers, document repairs |
| 10:25-10:35 | **Break (10 minutes)** | |
| 10:35-11:15 | [One stock report](2_stock_analysis_template.ipynb) | Parameters, adjusted returns, rolling volatility, drawdown, plots, Scrapbook outputs |
| 11:15-11:45 | [Run, collect, compare](3_running_analysis_at_scale.ipynb) | Papermill, three-stock batch, failures, summary and comparison |
| 11:45-noon | Homework clinic in module 3 | Failure exercise, six-month window, deliverables and questions |

**Total: 170 instructional minutes + 10-minute break = 180 minutes.** Exercises, report execution, and debriefs are included in these times.

## Setup and data

From the repository root, run `uv sync --locked` and select the project's `.venv` interpreter. Download `data.db` from bCourses into `Lectures/Lecture 4/data/`; the file is intentionally gitignored. No credentials or network access are required to run these notebooks.

- `ohlc`: supplied price histories for 30 stocks, ending December 6, 2024.
- `ohlc_hw`: intentionally dirty homework data.
- `market_caps`: dated market-cap observations.
- CSV and JSON teaching snapshots are already included in `data/`.

The notebooks locate data from the repository root, lecture folder, or reference folder. They open the source database read-only. Demonstration writes go to `outputs/cleaning_demo.db`; batch reports, logs, manifests, CSV summaries, and figures go to `outputs/`. A missing database produces an actionable error instead of silently creating an empty file.

Papermill child reports use the same Python interpreter as the running notebook. The runner writes a temporary kernel specification inside its own output directory, so a machine-specific registered kernel is not required.

## Teaching choices

Start with SQL so students understand the source before quality checking it. Compare SQL and pandas aggregations, and use a dated join to demonstrate why the full key matters. Reuse October 8's cleaning workflow on controlled faults; do not clean the homework for students. Distinguish impossible values from statistical review flags, and retrospective repairs from information available at the time.

After the break, focus on the report's contract and the simple rolling volatility students already know. Walk through the plots as an application of the prior session, rather than reteaching every plotting command. The default batch is three stocks, each run once. Use execution time to explain isolated kernels, parameter injection, and Scrapbook results. Collect only this run's successful report paths.

If behind, shorten the comparison-chart walkthrough and work the failure exercise as a class. Preserve the parameter tag, validation, one successful report, and result collection because homework needs all four.

## Reference library (optional)

- [Advanced outliers](reference/1_advanced_outlier_detection.ipynb): rolling/global comparisons, OHLC rules, return and spike/reversal flags, Prophet, alternative retrospective repairs.
- [Ingestion and SQL in detail](reference/2_data_ingestion_and_sql.ipynb): CSV options, nested JSON, cursors, CTEs, read/write examples.
- [Full multi-asset cleaning case](reference/3_real_world_data_challenge.ipynb): the original end-to-end case with text/sector normalization, duplicates, repairs, validation, correlations, and cleaning log.
- Yang-Zhang OHLC volatility remains an executable appendix in module 2. The live report and batch summary use close-to-close volatility.
- Module 3 retains the four-period batch extension; set `RUN_EXTENSIONS = True` to run it after class. It uses complete 2023 quarters; the selected stocks have insufficient observations for a 20-day return window in Q4 2024.

## Where the original material went

| Original material | Live treatment | Extended treatment |
|---|---|---|
| Lecture 2 Jupyter tutorial | 10-minute opening check | Prework tutorial retained in Lecture 2 |
| Lecture 2 Series + DataFrames | October 8 module 1, one portfolio workflow | Both expanded notebooks in Lecture 2/reference |
| Lecture 3 visualization | October 8 module 2, chart choice and Figure/Axes | Full notebook in Lecture 2/reference |
| Lecture 3 cleaning fundamentals | October 8 module 2, methods and small case | Full notebook in Lecture 2/reference |
| Lecture 3 full case study | Integrated short cases on both days | Full case in this folder's reference library |
| Lecture 4 outliers | October 15 module 1, OHLC/rolling/return/reversal flags | Prophet and alternative repairs in reference |
| Lecture 4 ingestion/SQL | October 15 module 1, retrieve then validate | Cursors/CTEs/options in reference |
| Lecture 4 template | October 15 module 2, simpler volatility matching homework | Yang-Zhang appendix |
| Lecture 4 batch reports | October 15 module 3, three-stock batch and summary | Multi-period extension |

The original roughly 7-8 hours of teaching becomes 5 hours 40 minutes of live instruction by consolidating repeated examples and moving deeper variants to optional study, not by speeding through every old cell. No required homework skill depends on completing an optional notebook.

## Homework and verification

[Homework 2](../Lecture%202/hw2.md) is due **October 29, 2026**, two weeks after this final data-analysis session. Keep database files local; submit the requested code and small sample reports.

From the repository root, run `uv run python scripts/validate_data_analysis.py`. It executes the five live modules, prework, and seven reference notebooks in fresh kernels against a temporary SQLite backup. It also checks report error cases, numerical invariants, and the multi-period extension. Use `--core-only` for a shorter check of the five live modules. Validation outputs are temporary unless `--output-dir PATH` is supplied.
