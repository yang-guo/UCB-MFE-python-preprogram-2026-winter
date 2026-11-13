# HW4 - Genrikh Kuznetsov


## How to run

1. run `uv sync --locked`
2. download `equities_top2000.duckdb` into `Lectures/modeling/data/`
3. Open the notebook with the `.venv` kernel and run all cells. For a quick trial, set `DATABASE = "equities_class.duckdb"`.

The notebook imports the data and feature helpers from `Lectures/modeling/course_data.py`. The backtester is written in the notebook. Large regenerable files go to `outputs/`, which Git ignores. The small result files go to `results/`.

## Approach

The model uses the Lecture 7 features (trailing price features, filing-date-aware financial ratios, and sector) in the lecture's preprocessing pipeline. It feeds a Ridge regression trained on each stock's same-day percentile rank of its 90-day return. The alpha and the target were chosen with expanding-window validation over 2012–2015, using only matured labels. Every month the strategy buys the 20 highest-scored stocks from that quarter's top-2,000 list, using one third of NAV split equally. Each purchase is held for 90 calendar days. The model is refitted every quarter on newly matured labels.

## Result (Apr 1, 2016 – Sep 30, 2026)

| | Strategy | S&P 500 TR |
|---|---:|---:|
| Final value | $2,106,125 | $4,402,747 |
| CAGR | 7.35% | 15.16% |
| Max drawdown | −28.2% | −33.8% |
| Sharpe (rf 2%) | 0.44 | 0.77 |

The strategy did not beat the benchmark.

## Files

| File | Contents |
|---|---|
| `hw4_Genrikh_Kuznetsov.ipynb` | The full analysis, executed |
| `results/performance.csv` | Metrics for both portfolios |
| `results/yearly_returns.csv` | Calendar-year returns |
| `results/validation_rank_ic.csv` | Development validation by target type, alpha, and year |
| `results/cumulative_returns.png` | Cumulative-return chart |

AI assistance: I used Claude Code to help read the course materials, draft certain parts of the code and text, and debug. I reviewed and ran all of it.
