# October 8: Pandas, portfolio analysis, and trustworthy data

Yang · 9am-noon Pacific · first of two data-analysis sessions.
The course schedule calls this **Data Analysis I & II**. Lecture 3 is Aneesh's October 12 LLM session.

The three notebooks form a progression: learn how pandas identifies and combines rows, apply those operations to a portfolio decision, then investigate the prices and labels that make the analysis possible. Each notebook runs independently and contains visible worked examples. Exercises and feedback stay at the end and use changed inputs or questions.

## Live modules and pacing

| Order | Notebook | Focus | Target |
|---|---|---|---|
| 1 | [Pandas basics](2_pandas_basics.ipynb) | Series/DataFrames, label lookup, joins and indices, aggregation | 60–65 minutes |
| | **Break** | | **10 minutes** |
| 2 | [From pandas to portfolio decisions](3_pandas_to_portfolio.ipynb) | Multiple lots/books, weighted cost, P&L, concentration and a cash-aware rebalance | 40–45 minutes |
| 3 | [Can we trust this price chart?](4_visualize_and_clean.ipynb) | Generated stock histories, return comparisons, vendor text and data repairs | 50–55 minutes |

**150–165 instructional minutes + 10-minute break = 160–175 minutes (2h40–2h55).** This leaves room within the three-hour session for questions or setup delays. The estimates include explanation and discussion of the worked examples. The final exercises are available for selected review or independent practice; completing every exercise live would require additional time.

One full-length route is basics 9:00–10:05, break 10:05–10:15, portfolio 10:15–11:00, and cleaning 11:00–11:55, with the final five minutes for questions. Timings belong here, not in the student notebooks.

## Before class

From the repository root, run `uv sync --locked` and select the project's `.venv` Python interpreter. The [Jupyter tutorial](1_jupyter_tutorial.ipynb) is optional preparation. Begin class with pandas basics; the tutorial is not a fourth live module.

The three live notebooks work offline from fresh kernels. The basics and portfolio modules contain synthetic execution and holdings tables. The cleaning notebook defines `generate_stock_data`, adapted from the previous visualization material: geometric Brownian price paths, seeded local random generators, positive and internally consistent OHLC, and volume. It creates five assets with different annual drift/volatility inputs and 252 observations each. Business weekdays approximate sessions; the dates are not an exchange calendar. No stock CSV or bCourses database is needed for Lecture 2.

The optional Jupyter tutorial keeps its historical AAPL sample. Live API access there is opt-in and requires a local `AV_API_KEY`. Credentials and generated output files stay outside Git.

## 1. Pandas basics: one execution is one row

Use the trade blotter as the common example rather than presenting a catalogue of APIs.

- **Series and DataFrames:** create a labeled Series, then a table of executions; show that selecting one column returns a Series and a list of columns returns a DataFrame.
- **Labels and positions:** the default index starts at zero, so `.loc[0]` initially looks positional. After sorting, label 0 still identifies T101 while `.iloc[0]` identifies the largest execution. Show `set_index`, `reset_index`, inclusive label slices, and exclusive positional slices.
- **Alignment:** multiply shuffled ticker-indexed shares and prices. Connect this to Series assignment and column concatenation.
- **Joins:** XOM exists only in the traded universe; TSLA exists only in the reference table. Compare inner, left, right, and outer results, including `_merge`. Column merges ignore source row indices; index joins require meaningful matching labels. `join(on=...)` matches the left column to the right index.
- **Cardinality and aggregation:** repeated ticker executions need a many-to-one reference join. Then distinguish executions, unique tickers, and dollar activity; keep unknown sectors with `dropna=False` so missing coverage does not erase activity.

The final exercises change the sort/filter request, the reference coverage, and the grouping question. If time is tight, demonstrate the append/concat cell briefly and save the aggregation stretch for practice. Preserve the sorting example, all four join types, index matching, and the repeated-key check.

## 2. Portfolio analysis: apply the basics

Start directly with eight lots across Core and Tactical books. The notebook uses the earlier pandas patterns; it does not repeat the Series/DataFrame, lookup, or join-type explanations.

The extra challenge is choosing the correct level of aggregation and denominator:

- Combine lots before assessing issuer exposure. AAPL's share-weighted entry price differs from the simple average of lot prices.
- Distinguish unrealized P&L, return on cost, and portfolio weight.
- Use `transform('sum')` to align book totals with lot rows; explain this one new operation where it is used.
- Apply illustrative limits of 25% per ticker and 55% in Technology.
- Sell 40 AAPL and buy 20 JPM in the worked scenario. AAPL meets its limit, but Technology still exceeds its limit. Include residual cash in the denominator and reconcile the total portfolio value.

The final exercises use a corrected lot size and a different AAPL/XOM trade. They require the student to apply the workflow and explain the result, rather than copy the worked scenario's numbers.

## 3. Visualization and cleaning: inspect the assumptions

Introduce the generator through its inputs and resulting DataFrame; keep the function's construction brief. It supplies reproducible data for changing scenarios, not a separate stochastic-calculus lesson. The generated frame is retained, and faults are injected into a copy.

- Compare raw prices with normalized performance; compare AAPL/MSFT daily variability and worst daily moves.
- Investigate missing prices, impossible OHLC relationships, invalid volumes, label variants, and repeated records.
- Clean a vendor export for portfolio matching: strip markup/whitespace, standardize case, split exchange/ticker fields, and extract signed numeric quotes. Explain the business consequence of a malformed quote rather than dissecting regex character by character.
- Show why filling a null can violate the current day's range. Make repairs explicit, preserve flags, and validate the resulting table.

The final exercises compare NVDA/JPM distributions, build a combined exception table, repair a new volume fault, and clean a different vendor delivery. Keep plot formatting and the small missing-volume recap brief to make room for string parsing and validation.

## Flow and course connections

Lecture 1 supplies lists, dictionaries, loops, and simple functions. Pandas basics introduces table operations once; the portfolio module uses them for a decision; cleaning shows why matching keys and valid observations matter to that decision. October 15 then translates table selection into SQL and builds reports from checked data.

The previous Lecture 2 reference folder has been retired. The relevant Series/DataFrame content is consolidated into pandas basics, and string cleaning remains part of the main cleaning lesson. Lecture 4 retains its separate references for advanced outliers, ingestion, and the full cleaning case.

[Homework 2](hw2.md) remains due **October 29, 2026**, with its existing deliverables. Verify the sequence with `uv run python scripts/validate_data_analysis.py`; use `--core-only` to execute the six live data-analysis notebooks. Validation runs against disposable outputs and a copy of the Lecture 4 database, leaving student notebooks output-free.
