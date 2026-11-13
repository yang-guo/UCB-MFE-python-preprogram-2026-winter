# HW4 - Wen Sun

This folder contains a stock-return model, a monthly long-only strategy, and a daily backtester.

## Run

Open and run:

```text
hw4_workflow.ipynb
```

Recommended:

```text
Restart and Run All
```

The notebook:

1. Loads benchmark, universe, membership, and fundamentals.
2. Builds price, volume, turnover, and sector-relative features.
3. Screens factors by out-of-fold absolute IC.
4. Tunes and compares OLS, Ridge, Lasso, and boosted trees.
5. Generates quarterly walk-forward test scores.
6. Runs the backtester and accounting checks.
7. Writes outputs and the cumulative-return chart.

## Outputs

Results are written to:

```text
outputs/
```

including:

- `nav_history.csv`
- `fills.csv`
- `open_lots.csv`
- `diagnostics.csv`
- `pnl.csv`
- `predictions.csv`
- `fit_audit.csv`
- `validation.csv`
- `accounting_checks.json`
- `config.json`
- `cumulative_return.png`