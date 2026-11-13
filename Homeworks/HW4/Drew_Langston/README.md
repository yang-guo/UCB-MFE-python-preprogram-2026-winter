# HW4 — Stock Strategy
Drew Langston

## Files
- `hw4_stock_strategy_clean.ipynb`: complete analysis, backtest, accounting checks, charts, and discussion.
- `README.md`: setup and run instructions.

## Setup
Use Python 3.12 and install the required packages:

```bash
python -m pip install jupyterlab ipykernel duckdb pandas numpy scipy scikit-learn matplotlib yfinance curl_cffi
```

Use a recent scikit-learn version supporting `sparse_output` and `keep_empty_features`.

Download the course-provided database and place it at this path relative to the notebook:

`data/equities_top2000.duckdb`

The database is not included in this submission.

## Running the notebook
1. Start JupyterLab in the folder containing the notebook.
2. Open `hw4_stock_strategy_clean.ipynb` using the Python environment containing the required packages.
3. Select **Kernel → Restart Kernel and Run All Cells**.
4. Wait until execution finishes and the kernel is idle.

Internet access is required for supplemental Yahoo Finance and Investing.com price observations. Provider availability and revisions may affect reproducibility.

Saved outputs are included for inspection. Run all cells in order because later cells incorporate data repairs and replace preliminary results.

## Method
The strategy uses Ridge regression with momentum, volatility, financial ratios, and sector inputs. The Ridge penalty is selected using expanding chronological validation within the development period.

The portfolio starts with $1,000,000, makes monthly purchases, and holds separate purchase lots for 90 calendar days. It is long only, uses no borrowing, and includes transaction fees, cash limits, and position caps.

The notebook contains:
1. Data exploration and feature/target examples.
2. Model development and chronological validation.
3. Backtesting, corporate-action handling, and accounting checks.
4. Performance evaluation and limitations.

The final test covers April 1, 2016 through September 30, 2026. Annual model fits use only completed outcomes available before fitting.

## Results
The final strategy value was $3,933,687.36, compared with $4,402,746.78 for the S&P 500 total-return benchmark. CAGR was 13.93% versus 15.16%, and the Sharpe ratio was 0.596 versus 0.773.

The strategy did not outperform the benchmark on compound return or Sharpe ratio. Detailed results, charts, proposed next experiments, and data limitations appear in Part 4.

## Sources and adaptations
Trailing-price feature calculations draw on Lecture 7.2. Missing-price conventions and performance calculations draw on `Lectures/modeling/backtest.py`. Adaptations are identified in notebook comments.

The primary dataset is the course-provided DuckDB database. Supplemental price histories use Yahoo Finance through yfinance and Investing.com. Corporate-action sources and verification checks are recorded in the notebook.

## AI assistance
AI assistance was used primarily to help debug Python code, especially issues involving missing prices, ticker changes, and pricing inconsistencies encountered during backtesting. It also helped draft portions of the code and explanations, investigate data gaps and corporate actions, develop accounting checks, and organize explanations. The notebook documents the implemented methods, assumptions, checks, and limitations.
