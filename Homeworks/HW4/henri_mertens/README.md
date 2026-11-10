# HW4 — stock prediction and fixed-horizon trading

Henri Mertens

The model predicts the adjusted-price return from the next market session's
close to a sale 90 calendar days later. The portfolio buys the highest-ranked
stocks using a fixed monthly allocation rule. Prediction accuracy and actual
portfolio performance are evaluated separately.

## Research design

All fitting uses observations dated 2010–2015 with outcomes completed by
December 31, 2015. The 2016 buffer is not used for training. The model,
preprocessing and trading rules remain fixed throughout April 1, 2016 to
September 30, 2026. There are no test-period coefficient updates, including
for the Kalman model.

The expanding windows use 2012 and 2013 for hyperparameter tuning. Each
training set contains only earlier observations whose actual 90-day outcomes
ended before the validation boundary, with five extra market sessions of
separation. All stocks on a date stay in the same partition. Training never
uses a block after validation, so an embargo after the validation block
would not remove any additional training observations.

Hyperparameters are tuned with 12 seeded Optuna TPE trials per model, four
of which initialize the search. The objective is average monthly Spearman
rank correlation between predictions and realized returns. Rank correlation
is relevant because the strategy selects stocks by their relative scores.
The search is deliberately limited; it does not establish a global optimum.

The six tuned models are compared on 2014 using the same ranking metric.
RMSE and then model name break ties. Only outcomes completed by the end of
2014 enter this comparison. The winner is then checked on 2015 without
changing the winner, search spaces or trading rules in response to that
check. The final fit uses completed pre-2016 outcomes. The configuration,
source fingerprints and fitted model are saved before the final test starts.

The earlier 30-stock Ridge pilot also examined 2015. Therefore the later
development check is not a pristine, previously unseen holdout; the genuinely
reserved performance evaluation is the 2016–2026 final test.

This separates tuning, model-family selection and a later development
check. It reduces selection bias but does not eliminate overfitting: six
model families and 72 parameter configurations are examined, there are few
validation years, and the 90-day monthly labels overlap. Neither Boruta
decisions nor individual stock-month observations are independent evidence
of statistical significance. No claim of a selection-adjusted p-value is
made.

## Features and models

The candidate inputs are the lecture's five price features (21- and 63-session
momentum, longer momentum excluding the last month, 63-session volatility,
and 252-session drawdown), six financial ratios (net margin, operating ROA,
debt/assets, cash/assets, free cash flow/assets, and accruals/assets), and
filing age. Static sector classifications and current active status are
excluded. Financial inputs become usable after the recorded filing date
plus one calendar day; stale ratios follow the lecture's 550-day rule.

Each fit uses at most 6,000 date-balanced stock-month training rows,
sampled with a fixed seed independently of target values. Every model uses
the same sample within a fold. This makes the RBF support-vector experiment
practical on a laptop, but sacrifices training information. Validation and
prediction retain all historically eligible stocks. Monthly signal dates
also reduce the number of heavily overlapping daily outcomes.

Within each training window only, fit 1st/99th-percentile feature clipping,
median imputation, scaling and Boruta. Boruta compares forest importance
with shuffled shadow inputs, using 128 shallow trees and up to 30 iterations.
Only confirmed features are retained; tentative inputs are recorded but
excluded. If none are confirmed, the fold predicts the training mean.
The fitted selector is shared across trials because its settings and
training data do not change; it never sees validation labels.

Training targets are clipped at their training sample's 1st and 99th
percentiles and standardized for fitting. Reported validation errors use
original, unclipped outcomes. These robust training choices limit the
influence of extreme returns without removing future investment candidates.

Models and search ranges:

| Model | Parameters searched |
|---|---|
| Elastic Net | alpha 0.0001–1; L1 ratio 0.05–0.95 |
| XGBoost | 80–240 trees; depth 2–4; learning rate 0.02–0.15; child weight 10–100; row/column fractions 0.7–1; L2 1–100 |
| CatBoost | 80–240 iterations; depth 3–5; learning rate 0.02–0.15; L2 3–100 |
| MLP | layers (16), (32), or (32,16); L2 0.001–1; learning rate 0.0003–0.003 |
| Kalman regression | coefficient process variance 0.0000001–0.01; observation variance 0.1–10 |
| RBF SVR | C 0.01–10; epsilon 0.05–0.5; gamma 0.001–1 |

The MLP has no random validation split or early stopping; its iteration cap
is fixed at 150 and convergence warnings are counted. Kalman regression
models coefficients as a random walk during training, with a joint update
for each monthly cross-section. Its final coefficients are then frozen.

## Portfolio rules

- Start with $1,000,000, long-only, without borrowing.
- At the close before the first market session of each month, rank eligible
  stocks by predicted 90-day return and select the top 20. Ticker breaks ties.
- Commit at most 25% of the preceding close's NAV, limited by available cash.
  Divide that budget equally among selected stocks, subject to a planned
  5% NAV cap per name including existing lots. Unused budgets remain in cash.
- Buy at the following session's close. Dollar budgets are fixed before
  that fill price is known. A missing entry quote cancels the order without
  selecting a replacement using future information.
- Hold each lot for 90 calendar days and sell at the maturity close or next
  market session. Leaving the eligible universe does not force a sale.
- Charge 10 basis points on each purchase and sale. Cash earns zero.
  Fractional adjusted-price units represent total-return exposure; dividends
  are not added again. This abstracts from actual-share corporate actions,
  integer lots, market impact and liquidity constraints.
- Carry a missing held-stock quote for at most five sessions and record
  stale marks or delayed exits. A longer gap stops the evaluation as
  unresolved; no liquidation or delisting recovery is invented.

The equal-dollar rule avoids estimating unstable Kelly weights from noisy
return forecasts. It is fixed in advance rather than selected by trying
many portfolio variants. No stop-loss or triple-barrier exit is added,
because this assignment prescribes a 90-day holding period. The 2015
development portfolio starts in February, matching the first available
confirmation prediction; this timing is not the final-test schedule.

Compare with $1,000,000 invested in the S&P 500 total-return index at the
first final-test close. The benchmark is held through the same end date,
without trading fees. Both portfolios report final value, actual-time CAGR,
maximum drawdown including initial capital, and daily-return Sharpe using
a 2% annual risk-free rate. Open positions are marked at the test end.

## Recorded results

XGBoost won the 2014 model-family comparison with monthly rank IC 0.0404,
followed by RBF SVR (0.0365) and CatBoost (0.0296). Its 2015 confirmation IC
was 0.0191. The fixed 2015 development portfolio lost 13.20%, while its
benchmark gained 3.18% over the same February–December dates.

The final interval contains 2,640 market sessions and 3,834 elapsed calendar
days. The corrected accounting run produced:

| Metric | Strategy, net of stated costs | S&P 500 total return |
|---|---:|---:|
| Final value | $1,312,631.29 | $4,402,746.78 |
| CAGR | 2.63% | 15.17% |
| Maximum drawdown | -53.72% | -33.79% |
| Sharpe ratio | 0.146 | 0.773 |

The strategy gained 31.26% cumulatively but trailed the benchmark by
309.01 percentage points and $3,090,115.49 of final wealth. It did not
deliver a convincing forecasting or portfolio advantage. The weak ranking
scores, changing selected features and large drawdown limit confidence.
Cash holdings also reduce market exposure; no attribution experiment was
run to isolate their effect. These results were retained without changing
the model or portfolio parameters after observing final-test performance.

The original run stopped at MYCC because merger proceeds were absent.
Subsequent accounting-only revisions added verified cash acquisitions,
stock conversions and ticker continuations. All original failures and
forecast fingerprints remain recorded. Cash entitlements were restricted
until original lot maturity, rather than assuming immediate reinvestment.
Stock conversions retain original lot dates. Later vendor rows for closed
predecessors cannot create new trades. Verified CervoMed and Onterris quotes
supplement holdings only; the current Yahoo SATS series was rejected because
it identified a different issuer. No held DISH conversion required those
unavailable quotes in the completed path.

The completed run has one day with a stale held-stock valuation, within
the five-session limit. An independent cash-flow reconstruction reconciled
daily cash to within $0.000001. All buys matched historical quarterly
membership, orders preceded entries by one session, all realized lots
preserved the 90-day maturity, and final open lots matched actual supplied
September 30 quotes. The original fitted model and forecasts were unchanged
across the corporate-action revisions. See `results/accounting_audit.json`.
Sixteen focused synthetic tests cover timing, feature causality, purging,
cash, costs, maturity, gaps and corporate actions.

The reserved period has now been evaluated. Further model or trading-rule
changes would need a new independent evaluation period; the existing
2016–2026 result should not be represented as unseen again. Ideas for a
separate future study include a larger training sample, more stable feature
selection and a simpler momentum benchmark. They were not tried on this
final period.

## Files and running

To read the submitted report, open `03_purged_model_research.ipynb` with its
saved outputs. It also runs from the included compact `results/` files,
without the database or a new model search. `results/.../portfolio_values.csv`
contains the small daily series used to redraw the final comparison.

The commands below rebuild the full experiment in a fresh checkout. The
existing local experiment is preserved and refuses to overwrite its frozen
model or evaluation records.

Place the bCourses databases in `Lectures/modeling/data/` as specified by
the assignment. From this homework directory, with Python 3.12:

```powershell
uv venv --python 3.12.9 .venv
uv pip install --python .venv/Scripts/python.exe -r requirements.txt
.venv/Scripts/python.exe -m unittest test_research -v
.venv/Scripts/python.exe research.py develop
.venv/Scripts/python.exe evaluate.py development
.venv/Scripts/python.exe evaluate.py final
.venv/Scripts/python.exe evaluate_corporate_actions.py --revision 1
.venv/Scripts/python.exe audit_results.py
```

`evaluate.py final` is a one-time evaluation. It requires unchanged source
and model fingerprints and refuses to overwrite an existing final-test
start record. Keep the original record and results if a data problem is
discovered; do not retune in response to final-test performance.

The corporate-action command is a separate accounting correction using
the original frozen forecasts. In the recorded work, revision 3 completed;
earlier stops remain preserved. A fresh run with the complete action table
can complete in revision 1. The small supplemental quote CSV and provenance
file are included, so reproducing the correction does not require an API.

- `01_data_overview.ipynb`: database inventory, quality and filing-date audit.
- `02_model_development.ipynb`: preliminary 30-stock Ridge exercise.
- `03_purged_model_research.ipynb`: full-universe experiment and results.
- `research.py` and `research_config.json`: features, splits, Boruta,
  model comparison, search spaces and frozen experiment.
- `backtester.py`: independently implemented cash/lot accounting.
- `evaluate.py`: development portfolio and one-time final evaluation.
- `test_research.py`: leakage and hand-checkable accounting cases.
- `audit_results.py`: independent ledger, maturity, membership and model
  fingerprint reconciliation of the completed run.
- `evaluate_corporate_actions.py`, `corporate_actions.json`: verified event
  accounting and source references, separate from model research.
- `supplemental_quotes.csv`, `supplemental_quote_source.json`: CervoMed and
  Onterris holding quotes, with issuer/overlap checks and provenance.
- `results/`: compact saved evidence and performance comparisons.
- `outputs/`: local panels, trial records, feature selections, model,
  fingerprints, prediction files, fills, NAV history and diagnostics.

## Assignment checklist

- [x] Code and run instructions: the Python files, notebooks, requirements
  and commands above.
- [x] Feature, model, chronological-validation and trading-rule explanations.
- [x] Full homework universe with dated quarterly membership, including
  inactive names; membership changes do not liquidate existing lots.
- [x] A 90-calendar-day return target aligned with next-session-close entries.
- [x] Training-only preprocessing and feature selection; completed-label
  purging and a frozen pre-2016 fit for the specified final test.
- [x] An independently implemented $1 million, long-only backtester with
  cash, costs, purchases, sales, open lots and daily NAV.
- [x] Timing, missing-price, corporate-action, cash-interest and adjusted-price
  assumptions; no invented fills or double-counted dividends.
- [x] Both required performance tables and one combined cumulative-return
  chart, including final open positions and the 2% Sharpe hurdle.
- [x] Hand-checkable tests and independent accounting/timing evidence in
  `test_research.py`, `audit_results.py` and `results/accounting_audit.json`.
- [x] Findings, underperformance, limitations and ideas for a separate future
  study, without retuning on the observed final test.
- [x] Reused-code references and disclosure of Codex assistance.
- [x] Databases, environment and large intermediate outputs excluded from Git.
- [x] Commit and push the reviewed HW4 submission.
- [ ] Open the submission pull request and assign `kevinramlal` and
  `raghuramkowdeed` (due November 12, 2026; no cutoff time is specified).

## Data limitations and references

The source is availability-aware research data, not a certified historical
point-in-time feed. Recorded filing dates do not undo later restatements.
Quarterly capitalization rankings can contain inconsistent split/share
units; delisting proceeds and issuer mappings are incomplete. Current
active status is not used to select past stocks. Missing returns are
excluded only where a completed label is needed for fitting/scoring,
never as a filter on final-test investment candidates. Results remain
conditional on the supplied coverage and adjusted-price conventions.

Loading, price features, filing joins, membership and clipping reuse
[`course_data.py`](../../../Lectures/modeling/course_data.py). The backtester
is original to this submission. Codex assisted with implementation, tests
and explanations.

Method references:

- [BorutaPy: all-relevant feature selection](https://github.com/scikit-learn-contrib/boruta_py)
- [Optuna TPE sampler](https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.TPESampler.html)
- [Scikit-learn: selection bias and nested validation](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html)
- [Bailey, Borwein, López de Prado and Zhu: The Probability of Backtest Overfitting](https://escholarship.org/uc/item/4w1110bb)

The separate tuning/selection/check periods and accounting safeguards draw
on these general principles. This is not an implementation of a specific
strategy from *Advances in Financial Machine Learning*.
