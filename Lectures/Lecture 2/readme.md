# October 8: Pandas and portfolio analysis

Yang · 9am-noon Pacific · first of two data-analysis sessions.
The course schedule calls this **Data Analysis I & II**. Lecture 3 is Aneesh's October 12 LLM session.

The two covered modules build from pandas table operations to a portfolio decision. The planned visualization/cleaning module was deferred to October 15 and is now [Lecture 4's first notebook](../Lecture%204/1_visualize_and_clean.ipynb). Each notebook runs independently, with worked examples in the explanation and exercises at the end.

## Covered modules

| Order | Notebook | Focus |
|---|---|---|
| 1 | [Pandas basics](2_pandas_basics.ipynb) | Series/DataFrames, label lookup, joins and indices, aggregation |
| 2 | [From pandas to portfolio decisions](3_pandas_to_portfolio.ipynb) | Multiple lots/books, weighted cost, P&L, concentration and a cash-aware rebalance |

## Preparation and continuation

From the repository root, run `uv sync --locked` and select `.venv`. The [Jupyter tutorial](1_jupyter_tutorial.ipynb) is optional preparation. The two main notebooks use synthetic execution/holdings tables and work offline. The tutorial's live API example is opt-in and requires a local `AV_API_KEY`; credentials stay outside Git.

October 15 starts with the deferred seeded stock histories, plotting, vendor-text cleaning, and data repairs, then proceeds to SQL, risk reports, and automated analysis. See the [Lecture 4 guide and HW2 coverage map](../Lecture%204/readme.md) for the revised 2.5-hour route.

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

## Flow and course connections

Lecture 1 supplies lists, dictionaries, loops, and simple functions. Pandas basics introduces table operations once; the portfolio module uses them for a decision. The deferred [visualization and cleaning module](../Lecture%204/1_visualize_and_clean.ipynb) opens October 15 by investigating the prices and matching keys behind those decisions. SQL retrieval and reusable reporting follow.

The Lecture 2 reference folder remains retired. Series/DataFrame foundations are consolidated into pandas basics, and vendor string parsing remains in the main cleaning notebook, now in Lecture 4.

[Homework 2](hw2.md) is due **October 29, 2026**. The two sessions together provide the required patterns; see Lecture 4's requirement-by-requirement coverage audit.
