# October 8: Use Python to explain a portfolio and inspect its data

Yang · 9am-noon Pacific · first of two data-analysis sessions.
The course schedule calls this **Data Analysis I & II**. Lecture 3 is Aneesh's October 12 LLM session.

The aim is to help beginning and moderately experienced MFE students read, change, and check Python. Each module is complete course material for an instructor-led video walkthrough, starts independently, and ends with a small result students can explain. Financial questions give the code a purpose; the arithmetic is small enough to check by hand before applying it to a table.

## Live modules

| Time (PT) | Notebook | Student's result |
|---|---|---|
| 9:00-10:25 | [What is actually in our portfolio?](2_pandas_to_portfolio.ipynb) | Portfolio weights, sector chart, and one-sentence concentration note |
| 10:25-10:35 | **Break (10 minutes)** | |
| 10:35-noon | [Can we trust this price chart?](3_visualize_and_clean.ipynb) | Comparable price chart, documented repairs, and a quality note |

**170 instructional minutes + 10-minute break = 180 minutes.** Instructor explanation, examples, discussion, and time to review the final exercises are included in these times.

## Before class

From the repository root, run `uv sync --locked` and select the project's `.venv` interpreter (`.venv/bin/python` on macOS/Linux; `.venv\Scripts\python.exe` on Windows).
The [Jupyter tutorial](1_jupyter_tutorial.ipynb) is optional preparation: run through the first plot and practice Restart & Run All. The first module includes a setup check and begins with Lecture 1's dictionaries and lists, so missing this prework does not block participation.

Both live notebooks work offline from the repository root or their lecture folder. The first uses a synthetic four-stock snapshot. The second reads `data/classroom_prices.csv`: 120 synthetic business-day observations for each of AAPL, MSFT, and NVDA, generated with NumPy's random generator using seed 42. The ticker labels connect examples; these numbers are not market quotes. Business days here are a teaching approximation, not an exchange calendar. Faults are injected visibly into a copy during the lesson. The live notebooks do not need the bCourses database.

The prework tutorial also offers the existing historical AAPL sample. Live API access is optional, requires `USE_LIVE_API = True`, and uses a locally configured `AV_API_KEY`. Never submit credentials or `.env` files. Generated files go into the ignored `outputs/` folder.

## What students should be able to do afterwards

- Explain a row, column, Series, and DataFrame using an actual table.
- Select rows with a condition, distinguish labels from positions, and add a calculated column.
- Match data on keys, recognize multiplied rows, and summarize values by sector.
- Sort and group observations before calculating changes; explain the first missing return.
- Choose a bar, line, or histogram for a question and label its units.
- Prepare vendor text for portfolio analysis: normalize identifiers and company names, split exchange/ticker fields, parse numeric quotes, and flag values that need review.
- Use masks to locate faults, make a documented repair, and check that it respects data constraints.

A successful notebook execution is a software check. Evidence of learning is a correct change to the code **and** a student's explanation of the resulting numbers.

## How to teach the notebooks

Walk through the lecture cells from top to bottom using **predict → run → change → explain**. All code is visible, and complete worked examples stay beside the concepts they explain. The formal exercises are collected at the end of each notebook, so you can narrate and run the lesson continuously. Use the final Exercises section for practice or review after the walkthrough.

Exercise cells start with `None` or an empty sentence and are followed by their feedback cells. Worked examples use separate `example_` variables and different inputs or questions: select by price before practicing selection by shares, increase JPM before practicing a change to MSFT, and plot MSFT before practicing with NVDA. The lecture never depends on an exercise answer. Mathematical checks cannot judge a chart's readability or a written explanation; those require discussion or individual review.

For students who are struggling, use the supplied expression shape and ask them to change one condition or column at a time. For faster finishers, use the stretch prompt in each exercise; ask them to explain their answer before introducing another API. Do not turn early completion into an obligation to race through optional references.

### Module 1: questions and final exercises

| Question or exercise | Listen for | If it is missing |
|---|---|---|
| A: select positions with at least 10 shares | The comparison creates one Boolean per row; `.loc` uses that mask | Display the mask before selecting rows |
| B: double MSFT shares | All weights share a changed denominator; AAPL falls from 40% to one third | Recompute the $5,000 and $6,000 totals on paper |
| Sector aggregation | Four names can still mean 80% technology by value | Compare counting names with summing dollars |
| Desk update | A correct filtered table plus a quantified concentration statement | Ask for one value from the student's own table |

Core syntax: column selection, `.loc`, `.iloc`, comparisons, assignment, `sum`, `sort_values`, `merge`, and `groupby`. `pd.concat` appears in the duplicate-key demonstration and the optional new-holding task. Keep the expected merge error short; source selection matters more here than exception mechanics.

### Module 2: questions and final exercises

| Question or exercise | Listen for | If it is missing |
|---|---|---|
| Three-price example | +10% then −10% ends at −1%; the first return has no previous observation | Calculate 100 → 110 → 99 by hand |
| C: histogram | Fractions become percent only for display; the plot answers a distribution question | Ask students to read one axis aloud |
| D: locate bad prices | A close above the same day's high is inconsistent; a large return alone is not proof of an error | Compare a bounds mask with a return threshold |
| Vendor quotes | Matching labels and numeric prices are prerequisites for a valuation; a parsed number can still be invalid | Compare the two AAPL spellings and the negative quote |
| Text-cleaning exercise | JPM/NVDA identifiers match consistently; missing or malformed prices remain flagged | Ask which quotes the student would hold back from a valuation and why |
| Forward-fill example | Removing a null can create an invalid price relationship | Check the candidate against today's low/high |
| New-delivery task | An invalid volume is flagged; zero remains a placeholder | Ask what fact the repaired row still does not establish |

Do not teach the synthetic data generator live. Read the provided dataset, use short cells, and focus on interpreting intermediate values. Teach text cleaning through the vendor-to-portfolio matching question: show before/after identifiers and prices, rather than dissecting the regex character by character. The small two-day forward-fill counterexample is more useful here than cataloguing every missing-value option.

## Pacing and course connections

Keep the text example within the existing module by keeping chart formatting and the separate missing-volume example brief. If behind, shorten the duplicate-key demonstration. Preserve the vendor-text example, the return-order discussion, and the final validation. The end-of-notebook exercises can be completed after class. Include brief explanations between cells rather than uninterrupted Run All demonstrations.

Lecture 1 supplies lists, dictionaries, loops, and simple functions. This session extends them to tables. No knowledge of classes, decorators, context managers, or software architecture is expected. The LLM session is not a prerequisite for either data-analysis notebook. October 15 reuses selection, keys, and tiny-data checks; later modeling and backtesting lessons reuse date order, `shift`, and careful treatment of missing data.

## Reference library (optional, after class)

The expanded material is retained, but it is not extra required live content.

- [Series in detail](reference/2_pandas_series_basics.ipynb): constructors, slicing, NumPy operations, updates, compounding.
- [DataFrames in detail](reference/3_pandas_dataframes_basics.ipynb): alternative construction, assignment/renaming/dropping, concat and merge variants.
- [Visualization in detail](reference/4_visualization_seeing_your_data.ipynb): scatter/box/KDE, plotting styles, scatter matrices, lag/ACF, rolling statistics.
- [Cleaning methods](reference/5_data_cleaning_fundamentals.ipynb): drop/fill/interpolate, Z-score/IQR/percentile flags, clipping, duplicate policies, strings/regex.

Continue with the [October 15 lesson](../Lecture%204/readme.md). [Homework 2](hw2.md) remains due **October 29, 2026**, with its existing deliverables and grading requirements.
