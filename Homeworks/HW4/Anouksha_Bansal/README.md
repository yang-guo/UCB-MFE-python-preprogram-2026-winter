# HW4 - Anouksha Bansal

**Everything is in [`hw4.ipynb`](hw4.ipynb)**: the code (features, model, my own backtester, accounting tests, development validation, final test, diagnostics) and the full write-up.

My strategy did not beat the benchmark. From April 1, 2016 to September 30, 2026, $1M grew to **$1.80M (5.8% CAGR)**. The same $1M in `^SP500TR` grew to **$4.40M (15.2% CAGR)**.

![Cumulative returns](cumulative_returns.png)

## How to run

From the repo root, run `uv sync --locked` and put `equities_top2000.duckdb` from bCourses in `Lectures/modeling/data/`. Then open `hw4.ipynb` with the repo's `.venv` kernel and run all cells. Generated files go to `outputs/`, which is git-ignored.

**Reused code (cited):** `load_data`, `apply_membership`, and `make_labels` are imported unchanged from `Lectures/modeling/course_data.py`; `financial_events` and `price_features` are adapted from the course versions.

AI was used to help build the logic and explanations of code in the notebook.