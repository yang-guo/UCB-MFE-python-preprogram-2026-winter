# HW4 — Xiaofei Yan

Open hw4_analysis.ipynb for research design, validation, accounting evidence and
current evaluation status. Active market inputs come only from the supplied
course database. There are no external quotes or assumed merger payouts.

## Current status: incomplete portfolio evaluation

Quarterly Ridge forecasts cover the requested April 2016–September 2026 period.
The portfolio simulation stops on 2019-10-10 because held BID lacks quotes beyond
the five-session stale-mark limit. Only completed sessions through 2019-10-09
are reported. The saved metrics and charts compare strategy and benchmark over
that same shorter period; they are not full-period results to September 2026.
The BID position is retained as unresolved, not deleted or assigned an invented
sale/recovery. The earlier retrospectively excluded experiment is archived under
ignored outputs/ and is not the main submitted result.

## Run instructions

Download equities_top2000.duckdb from bCourses to Lectures/modeling/data/.
From the repository root:

```sh
cd Homeworks/HW4/Xiaofei_Yan
uv venv .venv --python 3.12.9
source .venv/bin/activate
uv pip install -r requirements.txt
python accounting_checks.py
python model_selection.py
python run_strategy.py
```

The final command saves run_status.json and a partial account history if a held
price cannot be valued reliably. It does not restart after removing securities.
After a complete set of matching quarterly scores exists, use
`python run_strategy.py --reuse-predictions` to repeat only portfolio accounting.
Feature caches and large files stay under ignored outputs/. Move an old feature
cache aside after changing feature/price-quality rules. Full quarterly fitting
can take time. The notebook can run from the personal directory or repository
root, using submitted results/ without outputs/ or the database. Regenerating
models requires the database and shared Lectures/modeling/course_data.py.

## Research design

Development: 2010–2015. Validation years: 2012, 2013, 2014. Buffer: January–March
2016. Requested final evaluation: April 1, 2016–September 30, 2026. Earlier history
supplies lookbacks. Alpha 1000 was selected from [1,100,1000] using mean fold RMSE
0.19939 and 2,912,177 eligible mature development samples. Quarterly expanding
fits use only label_end < fit_date. Training-fitted clipping, median imputation,
missing indicators and scaling are refitted each time. The 20 numeric features
(10 fundamental, 10 price/volume) and formulas are in the notebook. Financials
become available from filing date plus one day; stale/unavailable data stay missing.

Monthly Top 10 selections use preceding-close information and next-session
closing fills. Equal budgets total at most one third of preceding NAV, limited
by preceding cash and inclusive of fees. Missing entry quotes reject budgets
without redistribution. Each purchase is a separate 90-calendar-day lot. Mature
lots wait for a real quote if needed. No borrowing; initial cash $1,000,000;
5 bps per side; cash interest zero. Use fractional adjusted-price units without
adding dividends again. Leaving quarterly membership does not liquidate a lot.

## Files and evidence

- feature_panel.py, price_quality.py, common_stock_screen.py: data preparation.
- model_selection.py: chronological parameter selection.
- portfolio_backtest.py: independent accounting and execution.
- accounting_checks.py: hand-verifiable synthetic accounting/timing cases.
- run_strategy.py: quarterly forecasts and the strict, single-pass simulation.
- results/: small saved validation, partial performance, charts, status and evidence.

All seven Python files are dependencies. Keep them with requirements.txt and the
notebook. Submit neither the database, virtual environment, outputs/, nor models.
Cumulative return, cash fraction and drawdown charts identify their partial range.
PnL reconciles cash/holdings and realized/unrealized components. Sharpe uses daily
NAV returns and a 2% annual hurdle; CAGR uses the evaluated first-to-last test-close
span. First-session fees and initial capital are included in returns/drawdown.

## Findings, limitations and future work

A full strategy-versus-benchmark conclusion is unavailable with the unresolved
BID holding. Database inspection finds its last quote before the gap on 2019-10-02
and no dividend event near the gap supporting a cash payout. Later source records
cannot establish a contemporaneous exit or verified issuer continuity. Do not
remove this stock because its future history becomes missing. No future coverage
blacklist or stop-and-rerank loop is used in the main experiment.

Random Forest was attempted but abandoned for runtime. The earlier seven-symbol
exclusion experiment was investigated and then withdrawn because it used future
coverage information. Repairing source coverage/identifiers and obtaining reliable
corporate-action records is future work; the full-period comparison must be rerun
after defensible repair. Historical security types and original filing vintages
are not certified. Closing-price execution and liquidity assumptions also limit
realism. Beating the benchmark is not established by this incomplete evaluation.

## Attribution

Reuse/adapt Lectures/modeling/course_data.py for features, availability joins,
membership, labels, folds and preprocessing; follow Lecture 8/2_research_workflow
for quarterly fitting. Accounting concepts reference 1_backtesting.ipynb; the
independent simulator does not import the course OMS/Backtester. Codex assisted
with planning, implementation, diagnostics and documentation.

## Price quality policy


These conservative rules were chosen after investigating development-period
price records, before evaluating the final test. They are review thresholds,
not evidence that a large genuine return is impossible.

- Preserve the source database. Keep a CSV audit of each flagged quote under
  local `outputs/data_quality/`, with its symbol, date, price, volume and reason.
- Quarantine nonpositive/nonfinite adjusted prices, following the course policy.
- Quarantine a zero-volume quote when its adjusted price is at least 100 times
  or at most 1/100 of the latest preceding positive-volume adjusted price.
  This flags the inconsistent zero-volume price regime seen in AAM, without
  assuming which corporate action or vendor issue caused it.
- After that step, flag observed-price changes of at least 10 times or at most
  1/10 as unresolved scale transitions. These are not automatically rescaled
  or declared erroneous; real events can also trigger this conservative flag.
- Mark new-entry/training signals ineligible when the trailing 253 market
  sessions include a quality flag. That prevents the 252-return features from
  crossing the unresolved event, using only information known by the signal.
- Censor a training target if its entry-to-exit interval includes a flagged
  quote. Preserve the original target as `target_before_quality`. Future label
  availability/quality is never used to decide whether to buy at a past signal.
- Do not interpolate, fabricate split factors, remove an entire stock using
  its future history, or clip validation targets to make reported errors smaller.

Lecture 8's training-fitted 1st/99th-percentile feature clipping remains in the
Ridge pipeline. It is distinct from source-price review and label censoring.
Financials still follow the lecture's filing-date-plus-one-day availability
rule and stale-data treatment. Restated vendor snapshots and unverified price
adjustments remain limitations. Flagged quotes in an existing holding require
the backtest's missing/unresolved-price policy; an entry screen alone does not
make held-position valuation reliable.


## Security and coverage assumptions

The source lacks dated issue types. Instrument names, code patterns and a
conservative review list exclude 200 non-common/uncertain issues. The remaining
3,649 symbols are assumed common-stock candidates, not individually certified
historical shares. REIT common shares and ordinary A/B classes are retained.
Snapshot classifications and ticker reuse remain limitations. Quarterly membership
and causal quality/history masks apply; future coverage is never an entry screen.
