# HW 4

Everything is in [hw4.ipynb](hw4.ipynb): the code, a short explanation of each step, the accounting checks, the results and the chart.

To re-run it, put `equities_top2000.duckdb` in `Lectures/modeling/data/`, use the course environment (`uv sync --locked`), and run the notebook top to bottom (about 2 minutes).

Results for the final test (April 2016 - September 2026):

- **Version 2** (gradient boosting, buying the 20 top picks among the 100 largest stocks, 90-day holds) ended at $5.88M (18.4% CAGR), against $4.40M (15.2%) for the S&P 500 TR. It was chosen after seeing the first version's test result, and its result is sensitive to data cleaning and to the scikit-learn version ($5.15M with 1.9.1 instead of the locked 1.8.0), as the notebook explains.
- **Version 1** (Ridge, all stocks), chosen before the test, ended at $1.96M (6.6% CAGR).
