# HW4: 90-day stock-return strategy

## Status and an important data gap

The strict backtest **does not produce a reliable full-period result**: PCGU has no supplied quote after August 15, 2023. Three lots purchased before the gap are still open and their scheduled maturities fall after the final quote. The data does not establish their settlement proceeds, so the default run stops on August 23 after five stale market sessions. We cannot infer a zero payout, a sale at the last price, or a replacement security from the available data.

For diagnostic purposes only, `--allow-stale` continues to value unresolved lots at the last observed adjusted close and never books an imaginary sale. This is a valuation assumption, **not** a validated 90-day backtest. The three PCGU lots remain unresolved at the end. Their carried value is approximately $109,400, and the 784-session stale interval makes the reported portfolio values and return statistics unreliable. The chart and CSV are labeled provisional. In particular, these figures cannot establish whether the strategy beat the benchmark.

## Approach

Use the first trading session of each month for entries, with features and membership known at the previous close. Screen candidates to the top 200 in the entry quarter's dated `universe_membership` list. This is a historical rank screen, not a screen for current survival. Never close an existing position merely because it leaves the quarterly universe.

The four predictors are adjusted-close momentum over 21 and 63 sessions, momentum from 252 to 21 sessions ago, and annualized 63-session daily-return volatility. Prices are aligned to the benchmark calendar without filling gaps; 63 daily returns must be present. There are no financial-statement features. A standardized ridge regression predicts the adjusted-close return from the next session's entry to the first observed close on or after 90 calendar days. Targets without an entry quote or an exit within five sessions of maturity are censored, not filled in. Each model fits the scaler on its training data only.

The specification is fixed: ridge penalty 100, top 200 membership screen, 15 highest predicted returns, equal allocation, 10 basis points on each side. Validation trains on outcomes completed before 2014 and evaluates completed labels for entries from January 2014 through September 2015, with labels ending by December 2015. It yields 8,880 training and 4,116 validation rows, model RMSE **13.64** and constant-training-mean RMSE **18.18** (returns in decimal units). These extreme errors indicate substantial data/return outliers, so they should not be interpreted as evidence of useful forecasting. The final model uses 13,585 completed development labels through December 2015; it is frozen before the final period, and the January–March 2016 buffer is not used for training.

Each month's orders are chosen and budgeted at the preceding signal close, before the entry close is known. The strategy allocates at most one-third of current portfolio value and no more than the cash available at the signal, divided equally over up to 15 names. A missing entry price leaves the money in cash. Adjusted-price units are fractional; cash earns zero and dividends are not added again. Sales use observed closes at maturity or, for a short quote gap, the next observed close; an unresolved gap stops the strict run. Transaction costs are charged on purchases and sales. Uninvested cash and open lots are included in daily value; there is no borrowing or shorting.

## Provisional diagnostic comparison

The sensitivity run starts April 1, 2016 and ends September 30, 2026. The benchmark is $1 million invested in `^SP500TR` at the first test close. CAGR uses actual days / 365.25; maximum drawdown includes initial capital; Sharpe uses daily returns, 252 sessions and a 2% annual risk-free rate converted to a daily rate.

| Metric | Strategy, stale-mark scenario | S&P 500 TR |
| --- | ---: | ---: |
| Final value | $3,338,754 | $4,402,747 |
| CAGR | 12.17% | 15.17% |
| Maximum drawdown | -36.86% | -33.79% |
| Daily Sharpe | 0.523 | 0.773 |

The sensitivity scenario finishes $1,063,992 below the benchmark. This is only a scenario, not a defensible estimate of the true shortfall: in particular the unresolved securities cannot be reinvested and their end value is unknown. There were 3,627 recorded fills and 33 open lots at the end, including the three stale PCGU lots. See [the diagnostic chart](provisional_returns.png) and [metrics](provisional_metrics.csv).

## Limitations and next steps

- The membership ranking can contain implausible inputs. For example, 2014-Q4 ranks INPX second using a selection close of $6.88 billion and 536.88 shares; the implied basis is not a credible ordinary share price. The 2023-Q3 PCGU record ranks it 25th using $149.51 and 2.132 billion reported shares. These are flags for investigation, not verified contemporaneous market caps. This makes the top-200 screen particularly vulnerable to source errors.
- The source lacks complete dated security identities, original filing vintages and delisting proceeds. Dated quarterly membership reduces avoidable survivorship bias but does not eliminate it. Adjusted-price and corporate-action errors can materially distort both labels and rankings; the validation RMSE illustrates that risk.
- The model and strategy were specified before this diagnostic test. The stale-mark scenario was added **after** the test exposed the gap, strictly to illustrate its impact, not to select or optimize a strategy. A defensible final evaluation needs independent corporate-action/settlement records and an audit of extreme price and return observations. Consider robust target handling and a different point-in-time liquidity screen, then re-evaluate without tuning against the final period.
- The strategy trades a concentrated group of names with overlapping lots. Its performance is exposed to equity market risk, model errors, transaction-cost assumptions and stale or missing data. Actual costs and market impact may exceed the fixed fee.

## Run and tests

From the repository root, install the course dependencies (`uv sync --locked`) and place `equities_top2000.duckdb` under `Lectures/modeling/data/`. The database is read only and must not be committed.

```bash
.venv/bin/python -m unittest discover -s Homeworks/HW4/Vishal_Soni -p 'test_*.py' -v
.venv/bin/python Homeworks/HW4/Vishal_Soni/hw4.py
# Diagnostic only; writes the provisional chart and CSV:
.venv/bin/python Homeworks/HW4/Vishal_Soni/hw4.py --allow-stale
```

The tests check a $300 hand-worked purchase and 90-day sale, an unfilled missing entry, entry-fee accounting and a stale-price stop. The feature and target timing conventions follow the descriptions in `Lectures/modeling/README.md` and `Lectures/modeling/course_data.py`; the backtester and code here were written separately. OpenAI Codex assisted with the implementation and analysis; the results and limitations above should be reviewed before submission.
