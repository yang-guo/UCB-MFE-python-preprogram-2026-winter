# HW4 Report

## Objective

Build a model for 90-calendar-day stock returns, select a monthly long-only portfolio, and backtest it against `^SP500TR` from April 1, 2016 through September 30, 2026.

## Workflow

1. Load prices, benchmark, fundamentals, universe, and quarterly membership.
2. Build causal price, volume, turnover, and sector-relative features.
3. Create monthly signal dates and 90-day holding-period labels.
4. Screen numeric factors using training-fold absolute IC.
5. Tune and compare OLS, Ridge, Lasso, and boosted trees on purged 2013/2014 folds.
6. Score the test period with quarterly expanding-window refits.
7. Run the daily backtester and accounting checks.

## Data and universe

Final data: `equities_top2000.duckdb`.

The investable universe is `universe_membership`, the up-to-2,000 largest stocks by capitalization proxy in each quarter. Membership is matched to the next-session entry date.

## Features and preprocessing

Features include:

- price momentum at multiple horizons;
- volatility;
- drawdown;
- relative volume;
- Amihud illiquidity;
- return-volume correlation;
- turnover;
- sector-relative versions of numeric factors;
- sector profitability rank;
- financial ratios and filing age.

Numeric features are quantile-clipped, median-imputed with missing indicators, and standardized. Sector is one-hot encoded. For model fitting, the 90-day target is winsorized to `[-0.9, 5.0]`.

Factor screening uses training folds only and keeps factors whose absolute mean daily IC meets the threshold selected in the notebook.

## Model and validation

Purged chronological folds use 2013 and 2014 as validation years. Each training fold uses only completed labels before the validation period.

The notebook tunes and compares:

- OLS
- Ridge
- Lasso
- boosted trees

Each model is tuned on the same folds. The final model is selected by validation mean daily rmse.

## Trading strategy

- Initial capital: $1,000,000.
- Long only, no leverage.
- Monthly entry at the first session close.
- 90-calendar-day holding period.
- Up to 20 names per monthly cohort.
- Cohort budget: 1/3 of NAV.
- Name cap: 5% of NAV.
- Sector cap: 25% of NAV.
- Cost assumption: 1 bp per side.
- Cash interest: 0%.

Missing prices: carry the last observed close. If a holding has no quote for more than 126 market sessions and a reappearing quote is at least 20x higher or 20x lower than the last observed price, freeze the mark and ignore that quote.

## Backtester and accounting evidence

The daily backtester:

- decides using the previous close;
- fills at the current close;
- tracks cash, lots, fees, and PnL;
- sells matured lots when a valid quote exists;
- includes open positions in final NAV.

Accounting checks in `outputs/accounting_checks.json` passed, including:

- final NAV equals cash plus open-position marks;
- no negative cash;
- cash reconstructed from fill cash changes matches NAV history;
- first buy/sell PnL and timing checks;
- sales occur on or after due date.

## Results

| Metric | Strategy | S&P 500 TR |
|---|---:|---:|
| Final value | $2,293,889 | $4,402,747 |
| Total return | 129.39% | 340.27% |
| CAGR | 8.23% | 15.16% |
| Maximum drawdown | -40.71% | -33.79% |
| Annualized volatility | 39.63% | 17.82% |
| Sharpe ratio | 0.342 | 0.773 |

The strategy did not beat the benchmark: final value was $2,108,858 lower, CAGR was 6.93 percentage points lower, and the strategy had higher volatility and drawdown.

## Limitations

- Missing/stale prices materially affect final NAV.
- Delisting records are incomplete; no reliable recovery assumption is available.
- Financial data are latest vendor values, not true point-in-time vintages.
- Sector labels are current snapshot labels.
- Transaction costs are simplified and do not include market impact.

## Findings and next steps

The backtester and evaluation pipeline work, but the model did not add consistent stock-selection alpha. Next steps would be to use more stable factors, tune directly for IC, and separately handle illiquid or delisted names.

## Reuse

This work reuses course feature, preprocessing, and accounting ideas from `course_data.py` and `backtest.py`. AI assistance was used primarily for debugging, and refining the code.
