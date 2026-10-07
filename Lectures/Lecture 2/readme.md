# October 8: Pandas, visualization, and cleaning

Yang · 9am-noon Pacific · first of two data-analysis sessions.
The course schedule calls this **Data Analysis I & II**. This folder keeps its existing Lecture 2 location; the October 12 LLM session is in Lecture 3.

## Live modules

Each module starts in a fresh kernel and is useful on its own afterwards.

| Time (PT) | Module | Main outcome |
|---|---|---|
| 9:00-10:25 | [Pandas to a portfolio](2_pandas_to_portfolio.ipynb) | Inspect, select, filter, assign, group, and join; compute portfolio exposure |
| 10:25-10:35 | **Break (10 minutes)** | |
| 10:35-noon | [Visualize and clean](3_visualize_and_clean.ipynb) | Choose charts, calculate returns and rolling statistics, diagnose faults, repair and validate |

**Total: 170 instructional minutes + 10-minute break = 180 minutes.** The notebook agendas include exercises and discussion, not just demonstration time.

## Before class (15-20 minutes)

From the repository root, run `uv sync --locked`. Select the project `.venv` interpreter as the notebook kernel (`.venv/bin/python` on macOS/Linux, `.venv\Scripts\python.exe` on Windows).
Run the [Jupyter tutorial](1_jupyter_tutorial.ipynb) through the first plot; practice editing a cell and Restart & Run All. API and magic-command sections are optional. Class begins with a 10-minute check of these skills, so students who missed prework can still start.

Open notebooks with the repository root, lecture folder, or reference folder as the working directory. They locate local data from the repository. Default runs require no network or credentials. The tutorial uses the bundled historical AAPL sample; live API access requires explicitly enabling `USE_LIVE_API` and setting a local `AV_API_KEY`. Never submit `.env` files.

## Teaching choices

Teach indexing and filtering once, with Series flowing directly into a DataFrame. Keep the label/position trap, alignment, and join cardinality demonstrations. Use the portfolio exercise to revisit these concepts rather than repeating every constructor and selector.

After the break, teach matplotlib's Figure/Axes interface through one dashboard. Connect plots immediately to a small cleaning exercise: normalize identifiers, remove exact duplicates, inspect missing/non-finite values, repair known faults, and recheck OHLC bounds. Preserve the raw data and mark estimates. End with students explaining one cleaning decision.

If questions consume time, use the worked answers during the final exercise debrief; keep the break and validation checks. The long data generator, alternative plotting styles, and exhaustive API variants belong to reference practice. Do not add them to the live agenda.

## Reference library (optional)

These retain the expanded examples from the original Lectures 2 and 3. They are not extra required modules or prerequisites for the second session.

- [Series in detail](reference/2_pandas_series_basics.ipynb): construction, slicing, NumPy operations, updates, compounding.
- [DataFrames in detail](reference/3_pandas_dataframes_basics.ipynb): alternate constructors, assignment, renaming/dropping, concat and merge variations.
- [Visualization in detail](reference/4_visualization_seeing_your_data.ipynb): KDE, procedural vs object-oriented plotting, scatter matrices, lag/ACF plots, rolling statistics, figure export.
- [Cleaning methods](reference/5_data_cleaning_fundamentals.ipynb): drop/fill/interpolate, Z-score/IQR/percentile flags, clipping, duplicate policies, text and regex extraction.

Generated outputs go into the ignored `outputs/` folder.

## Next session and homework

On [October 15](../Lecture%204/readme.md), continue with SQL, quality checks, a reusable report, and automation. [Homework 2](hw2.md) remains one assignment for the combined data-analysis module, due **October 29, 2026**. Its requirements are covered in the live route; the extended references provide additional practice.
