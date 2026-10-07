# October 15: Retrieve data and build a report you can explain

Yang · 9am-noon Pacific · second of two data-analysis sessions.
The schedule calls this **Data Analysis III · SQL**.

The class is designed for MFE students learning to work with Python, including students who have not built software professionally. SQL is introduced as another way to select and summarize a table. Report calculations are checked on tiny examples before students see a full price history. Automation follows understanding one report.

## Live modules

| Time (PT) | Notebook | Student's result |
|---|---|---|
| 9:00-10:10 | [Ask the database a precise question](1_sql_and_data_quality.ipynb) | A filtered, ordered extract and a note about what was checked |
| 10:10-10:20 | **Break (10 minutes)** | |
| 10:20-11:25 | [Turn prices into a report you understand](2_stock_analysis_template.ipynb) | Explained returns, rolling volatility, drawdown, and a four-panel report |
| 11:25-noon | [One report becomes three](3_running_analysis_at_scale.ipynb) | Three successful reports, a recorded failure, and a comparison table |

**170 instructional minutes + 10-minute break = 180 minutes.** All exercises, discussion, report execution, and the homework handoff are included. The report gets 65 minutes because interpreting and modifying Python is the primary goal.

## Setup and data

Run `uv sync --locked` at the repository root and select the project's `.venv` interpreter. Download `data.db` from bCourses into `Lectures/Lecture 4/data/`; it stays gitignored. There is no live API or credential requirement.

The supplied database contains `ohlc` (30 stocks), intentionally dirty `ohlc_hw`, and dated `market_caps`. Coverage differs by ticker; the overall database ends December 6, 2024, but the three report examples have only 14 observations in Q4 2024. The optional quarterly batch uses complete 2023 quarters.

Each module runs from a fresh kernel and locates the repository from the root or its lecture/reference folder. The source database is read-only. Demonstration SQL writes and report outputs go into the ignored `outputs/` folder. A missing database gives a clear download/path message rather than creating an empty file.

Supplied batch setup creates a kernel definition inside that run's output directory so child reports use the current Python interpreter. It changes only the running notebook's kernel lookup; it does not install a user-level kernel. Students can rerun `run_report` interactively after the lesson. Restarting and running setup creates a fresh output folder.

## What students should be able to do afterwards

- Translate a table question into SELECT, WHERE, and ORDER BY, then inspect the result in pandas.
- Check a SQL selection against pandas and recognize the keys needed for a dated join.
- Use a short Python function and check it on three or four hand-computable values.
- Explain daily versus cumulative return, rolling warm-up rows, and drawdown from a running peak.
- Change a report parameter, predict the effect, and verify it in the output.
- Trace a loop over report inputs and exclude failed or stale outputs from a comparison.

Students do not need to implement kernel registration, a database abstraction, a reusable plotting framework, or a volatility estimator from a paper. All instructional code is visible and runnable in order. Supplied setup and formatting are labeled so you can introduce them briefly, then spend time on the analysis. Only interpreter-registration plumbing lives in the adjacent support utility; report logic stays in the notebook.

## Teaching sequence and formative checks

Each notebook is your complete course material: read the question, explain the example, run the cell, and discuss the output. The lesson is continuous; formal exercise prompts, starter cells, and feedback are collected in a final Exercises section. All instructional code is visible. Worked examples use different inputs or questions from the exercises: AAPL above $180 versus MSFT above $400, NVDA versus JPM extracts, different price paths for returns and drawdown, and a drawdown comparison versus a volatility comparison. Automatic feedback checks mathematics and table shape; discussion checks interpretation.

| Checkpoint | Evidence of understanding | Common difficulty and response |
|---|---|---|
| SQL recall | Student identifies the ticker/date key and explains why ordering matters | Draw two tickers with interleaved dates before discussing syntax |
| E: MSFT above 400 | SQL and pandas select the same rows, values, and order | Build SELECT/FROM first, then add one WHERE condition |
| Dated join | Student predicts preserved left rows and unmatched nulls | Show that joining only on ticker combines several different dates |
| JPM transfer | Student distinguishes checked rows from verified calendar coverage | Ask which question needs an external calendar |
| F: return function | Student predicts 50 → 55 → 44 as +10%, −20%, with a leading missing value | Name the input, operation, and returned Series explicitly |
| G: drawdown | Student separates the worst drawdown (−25%) from the final one (−10%) | Write the running peaks beside four prices |
| Parameter change | Student connects a shorter window with fewer warm-up rows and a changed estimate | Inspect the first non-null row before discussing the plot |
| H: comparison | Student selects, sorts, states units, and limits the conclusion to the period | Read one row aloud; compare a fractional metric with its percent display |

Use the final Exercises section after the walkthrough or for independent practice. Hints and stretch questions are included there. The parameter-change task is also at the end of the report notebook. A quiet student or a successfully executed notebook is not by itself evidence of understanding; ask students to explain a changed result.

### The deliberate batch failure

The default batch requests AAPL, MSFT, `NOT_A_TICKER`, and NVDA. **Three successful reports and one logged failure are the expected outcome.** Ask students to predict whether NVDA still runs, locate the failure record, and explain why the invalid ticker is missing from the summary. The failed notebook is retained for inspection and excluded by the successful-run manifest. The supplied collector also rejects duplicate keys and empty input.

This short demonstration makes the idea of checking results concrete without turning the lesson into an error-handling or infrastructure lecture. Later advanced-Python and production classes develop those subjects.

## Pacing and the full course

Keep the SQL section to table questions students can compare with pandas. CSV/JSON variants, cursors, CTEs, and exhaustive SQL syntax are reference study. Use saved intermediate output to discuss a student's misunderstanding rather than adding another query operator.

In the report, teach a tiny return function and `cummax` before showing a complete summary. The four-panel Figure/Axes code applies October 8's plotting pattern. Walk through one panel, then identify how the other panels reuse it; formatting details are available directly in the notebook for later study. Pause on units, warm-up rows, and the difference between a daily metric and a whole-period metric.

If behind, reduce plotting customization and assign the final exercises for later practice. Preserve the return/drawdown hand checks, the SQL-versus-pandas comparison, and the report comparison walkthrough. The final batch has supplied setup so class time is spent reading the loop and using the result.

Course connections:

- **Lecture 1:** dictionaries, lists, loops, and simple functions become table operations and report inputs.
- **October 12 LLM session:** no unpublished LLM lesson content is assumed. The same habit of inspecting a result applies regardless of whether a person or an AI helped write the code.
- **Advanced Python (October 19/22):** classes, scope, context managers, concurrency, and program structure come later; they are not prerequisites here.
- **Modeling and backtesting:** sorted dates, `shift`, rolling statistics, and information availability prepare students for time-series features and leakage discussions. A retrospective repair is not automatically usable as a contemporaneous feature.
- **Productionization:** simple assertions and inspecting a recorded failure establish a reason for later testing practices.

## Reference library and original coverage

| Original content | Current home |
|---|---|
| Jupyter setup and magics | Optional [prework](../Lecture%202/1_jupyter_tutorial.ipynb); brief in-class start |
| Series/DataFrame API detail | October 8 module 1 plus its two expanded references |
| Visualization variants and diagnostics | October 8 module 2 plus its expanded visualization reference |
| Cleaning methods, string cleanup and parsing | October 8 price repairs and vendor-text example/exercise, plus its cleaning reference |
| Full multi-asset case | [Full cleaning case study](reference/3_real_world_data_challenge.ipynb) |
| Rolling/global outliers, reversal flags, Prophet | [Advanced outlier reference](reference/1_advanced_outlier_detection.ipynb) |
| CSV/JSON, cursor, CTE and SQL detail | [Ingestion reference](reference/2_data_ingestion_and_sql.ipynb) |
| Report, risk calculations, and plots | Live module 2; Yang–Zhang remains its optional executable appendix |
| Papermill/Scrapbook and multi-period reports | Live module 3; `RUN_EXTENSIONS = True` enables 12 additional reports after class |

The live core concentrates on transferable Python operations, interpretation, and practice. The references preserve the wider original material without requiring a beginner to study it all before succeeding in class.

## Homework and verification

[Homework 2](../Lecture%202/hw2.md) remains due **October 29, 2026**, with the same three deliverables. The in-class examples establish the required patterns; students still implement the specified additional metrics (including Sharpe ratio), styling, cleaning choices, and all-stock/six-month expansion themselves. Do not submit the raw database or credentials.

Run `uv run python scripts/validate_data_analysis.py` from the repository root. It executes five live modules, prework, and seven reference notebooks in fresh kernels against a temporary database backup. It checks the lecture results before any exercise variables exist, then checks default blank exercises, correct learner attempts, and rejection of incorrect attempts. It also tests numerical invariants, report failure cases, and the quarterly extension. Use `--core-only` for a shorter check. `--output-dir PATH` keeps the disposable outputs for inspection.
