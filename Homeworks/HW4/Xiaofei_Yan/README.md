# HW4 — Xiaofei Yan

Open hw4_analysis.ipynb for research design, validation, accounting evidence and
current evaluation status. Features and model fitting use the supplied course
database. Execution additionally uses the sourced BID/DISH events and a small
archived BODI quote supplement described below; the source database is unchanged.

## Current status: full-period scenario with assumed stale-price exits

Quarterly Ridge forecasts cover the requested April 2016–September 2026 period.
The main saved scenario reaches 2026-09-30. During missing-quote intervals,
carry the latest accepted price for valuation. On the first market session on
or after a lot's 90-calendar-day maturity, use a genuine current quote when
available; otherwise book a hypothetical exit at the last accepted quote dated
no later than maturity. Record this as ASSUMED_STALE_EXIT, never as an observed
SELL. Cash becomes available at that assumed exit date, not the old quote date;
the usual 5 bps exit cost applies. No future quote is used to choose the price.

This is an explicit simplification requested after encountering test-period
data gaps. It does not establish that these exits were executable, and does
not satisfy an observed-fills-only interpretation of the assignment. The strict
reference stops on 2024-10-22 (last completed session 2024-10-21); its WOLF lot,
status, performance and accounting evidence remain in results/strict_reference/.
The earlier retrospective stock-exclusion experiment stays archived under
ignored outputs/ and is not used. No future-coverage exclusions or reranking.

Three assumed exits occur (prices per adjusted-price unit):

| Symbol | Maturity/assumed exit | Last quote used | Price | Quote age in sessions |
| --- | --- | --- | ---: | ---: |
| PEGY | 2024-12-02 | 2024-11-19 | 3.400 | 8 |
| WOLF | 2024-12-02 | 2024-10-14 | 11.235 | 34 |
| ASGN | 2026-07-30 | 2026-05-22 | 19.500 | 46 |

See results/assumed_exits.csv for amounts, fees, price dates and source labels.

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
python run_strategy.py --missing-price-policy last_before_maturity
```

The final command saves the assumption scenario to results/. It does not restart
after removing securities. Run `python run_strategy.py --missing-price-policy strict`
for the observed-quote reference in results/strict_reference/ (it stops on WOLF).
The strict reference retains the five-session stale limit; the assumption scenario
carries marks until maturity. Unresolved price-scale transitions still stop either
mode; the scenario only relaxes the missing-quote treatment.
After a complete set of matching quarterly scores exists, use
`python run_strategy.py --reuse-predictions --missing-price-policy last_before_maturity`
to repeat only scenario portfolio accounting (use `strict` for the reference).
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
lots wait for a real quote in strict mode; the main scenario instead uses the
disclosed last-price maturity assumption when needed. No borrowing; initial cash $1,000,000;
5 bps per side; cash interest zero. Use fractional adjusted-price units without
adding dividends again. Leaving quarterly membership does not liquidate a lot.

## Files and evidence

- feature_panel.py, price_quality.py, common_stock_screen.py: data preparation.
- model_selection.py: chronological parameter selection.
- portfolio_backtest.py: independent accounting and execution.
- accounting_checks.py: hand-verifiable synthetic accounting/timing cases.
- run_strategy.py: quarterly forecasts, strict reference and assumption scenario.
- corporate_actions.csv, stock_mergers.csv: sourced compulsory event ledgers.
- supporting_data/: small BODI execution quote snapshot and provenance.
- results/: full-period scenario, charts, assumed-exit ledger and evidence.
- results/strict_reference/: partial strict status, performance and unresolved lots.

All seven Python files, the two event CSVs and supporting_data/ are dependencies.
Keep them with requirements.txt and the notebook. Submit neither the database,
virtual environment, outputs/, nor models.
Cumulative return, cash fraction and drawdown charts identify the assumed-exit scenario.
PnL reconciles cash/holdings and realized/unrealized components. Sharpe uses daily
NAV returns and a 2% annual hurdle; CAGR uses the evaluated first-to-last test-close
span. First-session fees and initial capital are included in returns/drawdown.

## Findings, limitations and future work

Under the declared full-period scenario, the strategy finishes at about $638,352
(-36.16%), versus about $4,402,747 (+340.27%) for the benchmark: approximately
376.44 percentage points of underperformance. This is conditional on stale-price
valuations and three hypothetical exits, not an observed-fills-only result.
The strict reference remains incomplete at WOLF. Its last course quote before
the gap is 2024-10-14; the lot entered on 2024-09-03 and matures on 2024-12-02.
The scenario uses 11.235 on maturity, not the post-reorganization 2025 quote.
No future coverage blacklist or stop-and-rerank loop is used.

Stale marks suppress measured price variation during gaps. Assuming cash recovery
at the old price can overstate recoverability and changes future purchase budgets.
Consequently effects extend beyond the three exits; these results are not a
guaranteed lower or upper bound on real performance. Obtain genuine historical
quotes and verified issue mappings before interpreting them as executable returns.

Random Forest was attempted but abandoned for runtime. The earlier seven-symbol
exclusion experiment was investigated and then withdrawn because it used future
coverage information. Repairing source coverage/identifiers and obtaining reliable
corporate-action records is future work; a strict full-period comparison still
requires defensible repair. Historical security types and original filing vintages
are not certified. Closing-price execution and liquidity assumptions also limit
realism. The strategy underperforms in both the shorter strict reference and
the full-period assumed-exit scenario; neither justifies parameter retuning.

## Sourced operational repairs

These repairs were investigated after test-period operational failures, not
preregistered. Model parameters, scores, training labels and rankings are not
retuned. Remaining source coverage/identity problems can still stop simulation.

- **BID / Sotheby's:** [NYSE's SEC notice](https://www.sec.gov/Archives/edgar/data/823094/000087666119001019/ruleprovisionnotice.htm)
  confirms compulsory $57 cash consideration effective before the 2019-10-03
  open. The final course quote on 2019-10-02 has raw and adjusted close 56.99,
  so the adjusted-unit conversion factor is 1. Record CASH_MERGER, not SELL.
  Same-day cash availability and 5 bps exit cost are simplifying assumptions;
  taxes and settlement delays are omitted. The modern BID snapshot name is a
  different issuer; do not bridge its multi-year gap to value the old holding.
- **DISH / EchoStar:** [EchoStar's completion notice](https://ir.echostar.com/news-releases/news-release-details/echostar-corporation-completes-merger-dish-network-corporation)
  states that the merger completed on 2023-12-31, with 0.350877 EchoStar shares
  per DISH share. Apply the conversion at the first subsequent market session,
  2024-01-02; retain cost basis, entry date and maturity. No exchange trade or
  conversion fee is invented. Both course price bases have raw/adjusted ratio 1
  around the event. The database calls EchoStar ECHO, whereas the official
  ticker is SATS: using that source series is an explicit issuer-name mapping
  assumption, not a verified ticker-master repair. Course DISH rows after the
  effective merger are not used for the converted holding.
- **BODY / BODI:** [the issuer's rename announcement](https://investors.thebeachbodycompany.com/news/news-details/2024/The-Beachbody-Company-Inc.-to-Change-Ticker-Symbol-to-BODI-on-March-4/default.aspx)
  confirms the 2024-03-04 ticker change with unchanged CUSIP. The archived Yahoo
  daily snapshot in supporting_data/ adds 85 absent quotes, only for execution
  and valuation. Preserve all course quotes and do not rescale prices. Across
  20 overlapping observations, the largest relative difference is 0.2457%
  (2024-03-04: course 8.14 versus supplement 8.16); the other 19 agree to rounding.
  This validates approximate scale, not all vendor values. Source URLs and the
  comparison are recorded in the provenance and supplemental_quote_audit.json.

[Wolfspeed's 2024-10-15 announcement](https://investor.wolfspeed.com/news/news-details/2024/Wolfspeed-Announces-750M-in-Proposed-Funding-from-U.S.-CHIPS-Act-and-Additional-750M-From-Investment-Group-Led-By-Apollo-Galvanizing-Global-Leadership-in-Delivering-Next-Generation-Silicon-Carbide-Technology/default.aspx)
does not resolve its daily quote gap. Obtaining consistent historical prices and
rechecking held-unit adjustments is future work. The unresolved strict WOLF
lot is saved in results/strict_reference/open_positions.csv; the main scenario's
remaining end-date lots are in results/open_positions.csv.

## Next improvements to missing-price exits

The three stale-price exits are scenario assumptions, not evidence that their
actual maturity-date closes were unavailable in the market. The course database
has no quote on any of those exit dates. Before interpreting the assumed cash
recoveries as executable, obtain and verify the old WOLF series through its
2024-12-02 maturity, including its adjustment basis before the 2025 reorganization.

The other two gaps may be ticker-mapping problems. A [Pineapple/SUNation SEC filing](https://www.sec.gov/Archives/edgar/data/22701/000089710124000584/pegy241204_8k.htm)
states that PEGY changed to SUNE effective 2024-11-19. An [ASGN/Everforth
filing](https://www.sec.gov/Archives/edgar/data/890564/000089056426000020/earningsrelease-q126.htm)
states that ASGN changed to EFOR effective 2026-04-24. The course database has
no SUNE or EFOR price rows. Retrieve their historical quotes from a documented
provider, verify the issue identity and adjusted-price convention against
overlapping old-ticker observations, and check for corporate actions (including
[PEGY's earlier reverse split](https://www.sec.gov/Archives/edgar/data/22701/000089710124000537/pegy241094_8k.htm))
before substituting genuine maturity-date prices.
No such outside quotes have yet been obtained or validated for these three exits.
The full-period analysis should then be rerun and compared with both the strict
reference and current assumption scenario; other gaps may still be revealed.

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
