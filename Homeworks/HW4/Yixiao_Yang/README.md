# HW4 — Yixiao Yang

Open **[HW4.ipynb](HW4.ipynb)** first. It contains the feature definitions,
model and trading choices, validation, accounting checks, full-period results,
cumulative-return chart, limitations and source appendix. The saved outputs
can be read without the database.

## Run

Use Python 3.11+ and install the dependencies:

```bash
python -m pip install -r requirements.txt
python -m unittest test_hw4 -v
```

Run the notebook from top to bottom. It defaults to displaying the recorded
results. To regenerate everything, set `REGENERATE = True` and point
`DATABASE` to the official `equities_top2000.duckdb` from bCourses.
You can also use the command line, replacing `PATH` with the database path:

```bash
python hw4.py --database "PATH" --output outputs_regenerated
```

For a faster replay of the unchanged supplied forecasts and an independent
accounting audit:

```bash
python hw4.py --replay --database "PATH" --output outputs_replay
python audit_hw4.py --database "PATH" --output outputs_replay
```

The notebook and commands automatically verify and restore the supporting
ZIP. It contains detailed result tables, original forecasts, corporate-action
inputs, source observations and check records. Current run outputs are never
replaced by the bundled historical snapshot. The classroom database cannot
reproduce the final experiment; the full database is excluded from Git.

## Results

The final period is April 1, 2016–September 30, 2026; initial capital is $1M.

| Metric | Strategy | S&P 500 total return |
|---|---:|---:|
| Final value | $1,311,729.05 | $4,402,746.78 |
| CAGR | 2.62% | 15.17% |
| Maximum drawdown | -69.40% | -33.79% |
| Sharpe, annual risk-free 2% | 0.1903 | 0.7725 |

The Ridge strategy underperformed by $3,091,017.72. It also performed poorly
in development validation. The notebook explains extreme source labels,
missing observations, the verified accounting repairs and the next changes
I would investigate using a newly declared research design. Reorganizing the
files did not change the fitted strategy, its forecasts or these results.

## Files

| File | Role |
|---|---|
| `HW4.ipynb` | Main walkthrough, explanation, saved checks, results and sources |
| `data.py` | Data loading, dated features/labels, source diagnosis and accounting inputs |
| `hw4.py` | Model, validation, full run, frozen replay and report generation |
| `backtester.py` | Cash, orders, 90-day lots, positions and performance measures |
| `test_hw4.py` | 56 hand-calculated/accounting/information-timing checks |
| `audit_hw4.py` | Independent reconstruction and supporting-data verification |
| `requirements.txt` | Python dependencies |
| `supporting_evidence.zip` | Supporting data and recorded evidence; leave compressed for upload |
| `README.md` | These run instructions |

## Sources and assistance

The data-loading, price-feature and filing-availability conventions are
adapted from the course's
[course_data.py](https://github.com/yang-guo/UCB-MFE-python-preprogram-2026-winter/blob/5212932d7d2e9de49785ee2c7e0395579b42d694/Lectures/modeling/course_data.py)
and [data guide](https://github.com/yang-guo/UCB-MFE-python-preprogram-2026-winter/blob/5212932d7d2e9de49785ee2c7e0395579b42d694/Lectures/modeling/README.md).
The backtester is a separate implementation. Corporate-action/market-data
sources and the evidence limitations are given in the notebook's appendix.

ChatGPT assisted with implementation, source research, tests, accounting
reconciliation, documentation and file organization. Numerical results were
run and checked; synthetic examples are used only in tests. The initial
failed run and later accounting repairs are disclosed in the notebook.

Submit this whole folder at `Homeworks/HW4/Yixiao_Yang/` in your own branch.
Do not upload the outer download ZIP, DuckDB database or expanded run outputs.
