# HW4: stock model and trading strategy

This project predicts stock returns and tests a trading strategy against the S&P 500 total-return index. It uses the full homework database. The code is split into small files so you can add features and try other settings.

## What is kept where?

| File or folder | What it does |
|---|---|
| `config.json` | Features, stock count, costs, position limits and other settings |
| `run.py` | Runs the research and saves the results |
| `strategy/data.py` | Reads the database and builds monthly feature rows |
| `strategy/model.py` | Trains Ridge models and chooses one using development data |
| `strategy/backtest.py` | Tracks cash, purchases, sales, open lots and daily value |
| `tests/test_backtest.py` | Checks accounting and timing using small made-up examples |
| `audit.py` | Checks saved trades independently against the original database |
| `results/FINDINGS.md` | Actual results, assumptions and limitations |
| `results/validation.csv` | Development scores for all model choices |
| `results/fit_audit.csv` | Shows which dates each fit could use |
| `results/performance.csv` | Final value, CAGR, drawdown and Sharpe for both portfolios |
| `results/cumulative_returns.png` | Strategy and index on one chart |
| `results/locked_config.json` | Settings saved before the final test predictions |
| `results/run_metadata.json` | Database checksum and package versions |
| `results/ACCOUNTING_CHECKS.md` | Full-run cash, fill, membership and maturity checks |
| `results/TEST_RESULTS.txt` | Saved output from the small accounting tests |
| `outputs/results/` | Full daily values, trades, predictions, open lots and data warnings; kept locally and ignored by Git |

The database stays in `Lectures/modeling/data/equities_top2000.duckdb`. Do not copy it into this submission. The sample notebook is a reference only and is not needed to run this project.

## How to run

Open a terminal in the **main course repository folder**. Use the course environment:

```bash
uv sync --locked
uv run python Homeworks/HW4/Prabhutosh_Kumar/HW4/run.py
```

If the course `.venv` already exists, this also works on macOS/Linux:

```bash
.venv/bin/python Homeworks/HW4/Prabhutosh_Kumar/HW4/run.py
```

On Windows use `.venv\Scripts\python.exe` instead. Paths in the config are relative to the main course repository, so you can launch the script from any folder. The full run may take several minutes. It prints progress for each batch of 100 stocks. Small reports go in `results/`; large files go in `outputs/results/`. Another run with the same output path overwrites its reports.

Run the checks from this submission folder:

```bash
cd Homeworks/HW4/Prabhutosh_Kumar/HW4
../../../../.venv/bin/python -m unittest discover -s tests -v
../../../../.venv/bin/python audit.py
```

## What the strategy does

At the close before each month's first market session, score the stocks in that entry quarter's universe. Buy the best 30 at the next session's close. Spend at most one quarter of the portfolio value, limited by cash already available. Plan no more than 5% of the portfolio in one stock. Prices can change before the fill, so the actual weight can be higher. Hold each purchase for 90 calendar days. Sell at the close on maturity, or the next quoted market session. Leaving the universe does not force a sale.

The initial cash is $1 million. There is no borrowing or short selling. Buys and sells each cost 0.1%. Fractional adjusted-price units are allowed. Cash earns no interest, and dividends are not added again. The benchmark buys `^SP500TR` at the April 1, 2016 close and holds to September 30, 2026. The strategy uses the March 31 signal for that first close. Open positions count in the final value.

Ridge is a linear model with regularization to reduce unstable coefficients. Features measure momentum, short-term reversal, volatility, drawdown and company financial ratios. The course helper only uses financial reports from the day after filing; old statements are blanked after 550 days. Missing values are filled using training medians. Extreme feature values are clipped and features scaled using training data only. Training target extremes are clipped to training-set percentiles; actual trading prices are never changed.

Train on completed outcomes from 2010 onward. Compare three regularization settings using yearly validation in 2013–2015. Targets that finish after 2015 are excluded from validation. Choose the highest average rank correlation, save the choice, and fit once on April 1, 2016 with completed development outcomes. No test-period retraining or test-based strategy search is used. Monthly stock count and allocation settings are fixed design choices rather than optimized parameters. The buffer allows late-2015 labels to finish.

## How to add features or run experiments

1. Copy `config.json` to a new JSON file so the submitted settings stay available.
2. Change `features`, `top_n`, `fee`, `tranche_fraction`, `max_stock_weight`, or `ridge_alphas`.
3. To add a new price feature, calculate it in `extra_features()` in `strategy/data.py`, include it in that file's selected extra columns, and register its name in `run.py`'s `allowed` set. Add the name to your config's `features` list. Use only current and earlier prices. For other data sources, join by the date the information became available.
4. Extend the causality test to cover the new feature.
5. Save each experiment separately:

```bash
uv run python Homeworks/HW4/Prabhutosh_Kumar/HW4/run.py --config Homeworks/HW4/Prabhutosh_Kumar/HW4/my_config.json --output Homeworks/HW4/Prabhutosh_Kumar/HW4/results_experiment
```

Use development validation to make decisions. Once you inspect final-test results, later changes are exploratory and cannot be called a fresh out-of-sample result. The assignment fixes the 90-day hold and final test end date; the runner checks these.

## Missing prices and trust in the results

Never invent a fill. A missing entry is cancelled. A missing exit stays open until there is a real quote. Last observed prices are used for daily provisional marks, with incidents recorded. A mark older than five sessions is counted as unresolved exposure. Set `strict_prices` to `true` to stop when a held price is missing for more than five sessions. The default allows the research run to finish and makes the unresolved exposure explicit; its NAV is not a verified liquidation value.

The database has incomplete corporate-action and delisting proceeds, some suspicious adjusted prices and approximate capitalization inputs. We flag large moves and ranking inputs without guessing split corrections. Financial values can be restated. These limitations affect whether an apparent win or loss is economically reliable. Read `results/FINDINGS.md` before interpreting the chart.

For model training only, the reused label helper leaves an outcome missing when the exit quote is delayed by more than five market sessions. Such rows are excluded from fitting and validation, which can bias the learned sample. Prediction and stock selection never require a known future outcome. Live simulated holdings remain open through longer gaps and are reported as unresolved.

## Reuse and assistance

The causal data loader, base features, membership joins, financial joins, labels and training preprocessing reuse `Lectures/modeling/course_data.py`, as allowed by the assignment. The portfolio engine and runner are independently implemented here. The attached sample was read for context, not copied. OpenAI Codex assisted with implementation, checks and writing. Review and understand the code before submitting.
