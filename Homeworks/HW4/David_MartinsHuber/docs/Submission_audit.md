# HW4 submission readiness audit

Audit date: November 12, 2026. Branch: `hw_4_David_Martinshuber`. Engine commit: `8b03da6`, with additional uncommitted work. `backtesting/portfolio.py` and package exports in `backtesting/__init__.py` changed concurrently during this audit; the tested snapshot and follow-up checks are distinguished below.

## November 13 structural correction follow-up

The original November 12 findings below are retained as the audit record. The
four requested structural corrections have since been implemented:

- **Reproducibility (finding 2):** `Backtesting.ipynb` explicitly builds data,
  FF5 factors, pair research, all three component strategies, and the optimizer
  in a fresh namespace. Old mismatched outputs were removed. It is a bounded
  Q1 2013 development smoke test on an explicitly illustrative 24-stock subset,
  not final evaluation or evidence of an unbiased universe-selection procedure.
- **Currency (finding 4):** history is preserved. New-entry eligibility and
  financial features use currency knowledge available at the signal date;
  later reports do not change earlier views. Missing currency remains unknown,
  and cannot erase a previously known non-USD currency. Existing lots retain
  their maturity schedule. The source-vintage limitations remain.
- **Missing prices (finding 3):** `delisting="last_price"` and the OMS early-sale
  bypass are rejected. Bounded stale marks are for valuation only; excessive
  held-price gaps stop the run. Sales still require an observed same-session
  quote and 90-day maturity. Dataset gaps may therefore block an experiment
  until investigated; the engine does not invent proceeds.
- **Research targets/calendar (findings 5–6):** forward returns use next-session
  entry and the first session on/after entry + 90 calendar days. Unfinished
  targets are missing. Research SQL filters invalid/off-calendar observations
  before computing lags and rejects cross-gap moves as daily returns. Old
  research CSVs are explicitly marked outdated and notebook outputs cleared.

Regression checks include future-filing historical-prefix invariance, exact
currency availability boundaries, preservation of existing holdings, atomic
rejection of early/stale liquidation, holiday-adjusted maturity, tail censoring,
and invalid/off-calendar/gap SQL fixtures. The read-only full-database calendar
check returns exactly **6,726 benchmark sessions** for both research proxies,
without refitting models or running the final portfolio experiment. The lecture
loader functions remain unchanged. The development notebook also passed every
code cell in a fresh process/namespace (145 seconds locally, including factor
and pair construction); accounting and maturity assertions passed. This bounded
run is not a full-universe efficiency benchmark.

Verification record: the initial complete discovery run executed **528 tests**
in 463.5 seconds: 524 passed and four failed on a residual OMS cash reservation.
The bookkeeping now counts pending purchases and clears the reserve exactly
when the final purchase is released, without discarding legitimate tiny pending
orders; execution rollback restores that count. The original four assertions
were preserved. A subsequent affected-module run passed **143 tests**, including
all four previous failures. The expanded structural regression file separately
passed **8 tests**. A further portfolio/optimizer/factor/market-data/research
integration run passed **78 tests** in 52.0 seconds. This is a full-run result
followed by targeted verification,
not a claim that a second complete discovery run was performed.

**Still unresolved after this run:** whether the stale carried marks support a reliable final NAV; development-period model/strategy selection provenance; and efficiency certification for the intended strategy set. The final period was executed, but the performance evidence remains exploratory until the missing-price valuation problem is resolved.

## November 13 completed run and submission notebook

The user then authorized the full period run using `Backtesting.ipynb`'s
two-component ERC setup and the indexed pending-sale lookup. The result completed
all **2,640 sessions** from April 1, 2016 through September 30, 2026 in about
29.6 minutes. Its outputs and inline charts have been promoted to the main
`Backtesting.ipynb`; the alternate notebook was removed. The synthetic
`Testing.ipynb` was moved to `audits/Testing.ipynb` and the audit runner updated.

The run produced these **exploratory, unvalidated** portfolio/benchmark values:

| Metric | Portfolio | ^SP500TR |
|---|---:|---:|
| Final value | $2,432,336 | $4,402,747 |
| CAGR | 8.83% | 15.16% |
| Maximum drawdown | -38.63% | -33.79% |
| Sharpe (2% annual risk-free rate) | 0.450 | 0.773 |

The output also reports **234,681 stale marks**, **16,328 open lots at the end**,
and `max_stale_sessions=10,000`. Positions stayed open and were marked using
carried prices, so the engine did not fabricate exit fills, but terminal NAV
relies heavily on stale observations. These metrics must not be presented as a
reliable final evaluation until missing held quotes and terminal marks are
resolved in a rule-compliant way. The component shadow summaries include
pre-test warm-up history and are not directly comparable to the main-period
portfolio. The generated comparison/chart show benchmark underperformance; the
homework does not require beating the benchmark for a pass.

The output accounting/timing assertions passed, including test-period coverage,
cash reconciliation, observed fills, and maturity dates. Those checks do not
validate stale marks as sound valuations. Final-period execution is complete;
reliable final-period evaluation remains unresolved. Model selection provenance
and full-strategy efficiency certification remain separate work.

## Original November 12 assessment

**Not ready to claim completion of HW4 yet.** The submission has substantial engine, research, strategy, and testing infrastructure. The current visible backtest is a development demonstration, not the required final-period evaluation. Reproducibility and two trading/data policies also need attention before final results can be trusted.

This assessment is against the [actual HW4 assignment](../../../../Lectures/Lecture%208/hw4.md) and [modeling data guide](../../../../Lectures/modeling/README.md), not an invented numerical rubric. The stated pass/no-pass criteria are a working independently written backtester, correct evaluation, readable code, and clear explanation. Beating the benchmark is bonus recognition, not a condition for passing. Multiple models, an optimizer, alpha/beta, VaR, and risk policies are optional extensions.

This audit changes documentation only. It does not repair the identified implementation issues, execute a full final-period experiment, or publish/commit additional work.

## Original November 12 blockers and findings

### 1. The required final-period result is absent from the inspected artifacts

The required final test is **April 1, 2016–September 30, 2026**, using the full homework database. `Backtesting.ipynb` currently loads through `2014-12-31` and runs the main portfolio for `2013-01-01`–`2013-03-31` with shadow history beginning in 2012. Its saved main report contains 60 daily observations. Those are development results, not evidence of the assigned final test.

The full database is present and its benchmark reaches `2026-09-30`; data availability is not the same as completing the evaluation. No saved final-period portfolio NAV/fill/result artifact was found in the inspected files. Research state CSVs are development-only through 2015 after the cutoff correction; they are not portfolio backtest results.

Needed: freeze a chosen strategy/configuration, run the assigned period, verify requested/effective coverage, and save the required comparison table and cumulative-return chart with an interpretation.

### 2. The current backtesting notebook is not a reproducible fresh-kernel run

The portfolio-construction cell uses `pair_result` and `pair_factors` without defining them in `Backtesting.ipynb`. They are built in a separate, unexecuted section of `Research.ipynb`; another notebook's variables are not a reproducible dependency.

Current source requests factor tilt, momentum, and pairs. Saved allocation/component outputs show only factor tilt and momentum. This is evidence that the saved results do not correspond to the current three-component source configuration. Some cells have outputs while execution counts are unset.

Needed: explicitly construct or load the selected components inside a documented runner, then restart the kernel and run all cells. Keep outputs only for that exact configuration. Using one working strategy is sufficient; there is no requirement to submit every experimental component.

### 3. `delisting="last_price"` can violate the 90-day and observed-fill rules

Sources: `backtesting/engine.py` held-price-gap handling and `backtesting/oms.py::forced_sale`. The normal `delisting="error"` path stops for investigation, but the optional last-price path sells a held security at its carried mark after the stale-price limit, even before maturity and without an observed quote on that execution date.

Independent reproduction:

- Eight benchmark sessions starting January 2, 2020; A has quotes only on the first two sessions, at $10.
- Buy A on January 3; lot maturity is April 2.
- Run with `max_stale_sessions=1`, `delisting="last_price"`.
- Actual result: sale on **January 7 at $10**, before maturity, with no A quote on January 7.

The assignment requires 90-calendar-day holdings and says not to invent fills. The data guide explicitly warns that a final quote is not reliable delisting proceeds. Labeling the fill `delisting` does not supply execution evidence or establish an exception to those rules. Factor/pairs usage examples and the pairs notebook currently select this option.

Needed: do not present this mode as compliant final execution. Use observed-price rules and investigate unresolved held-stock gaps; document any dataset limitation that prevents a reliable result. The development demo's `max_stale_sessions=1000` also needs a justified final policy rather than silently carrying marks for years.

### 4. Currency filtering uses later filings to remove earlier stock history

Source: `backtesting/market_data.py::MarketData.from_frames`, particularly the whole-symbol exclusion derived from all supplied fundamental rows. `MarketData.load` enables this behavior.

Adding a future non-USD filing can remove a stock from all earlier prices, membership, and financial rows. A synthetic A report dated May 15, 2020 changed whether A appeared in the January 2020 view. This violates the intended historical-prefix independence even though the later financial as-of merge itself respects publication dates.

The real database has two non-USD reporter symbols under the current rule: SMMT's first such filing is October 31, 2011, and GFL's is November 6, 2024. The issue is not the number of affected names; it is using future information to select an earlier universe. Treating missing currencies as USD is a separate documented assumption, not proof that currency comparability has been established.

Needed: make exclusions availability-aware at each decision date, or use a genuinely ex-ante universe restriction with a documented historical basis. A final-window data load must not retrospectively change earlier eligible universes merely because later reports were added.

### 5. Research forward-return diagnostics do not match the required trade target

Source: `research/pipeline.py::forward_return`.

- It measures from the signal-date price, rather than the next-session entry close.
- It chooses the last available close on or before signal date + 90 days, rather than the first permissible close on/after entry + 90 days.
- When the target lies beyond the available dataset, it can use the final observed price and report a shorter return instead of censoring the unfinished outcome.

Independent reproduction: prices `[100, 110, 121]` on September 28–30, 2026 produce a reported `forward_90d` of **21%** for September 28, although the 90-day horizon ends in December and cannot be observed. The calculation uses two days of data.

The function is documented as evaluation-only, so this is not proof that the current HMM fit consumes future targets. It does mean these tables cannot be treated as correctly aligned 90-day trade-return evaluation. Fix/censor the diagnostic or clearly separate it from the homework target. Never train or choose a strategy using these incomplete outcomes as if they were mature.

### 6. The cap-weighted research proxy contains off-calendar observations

`research/market.py` builds its proxy with direct SQL rather than the loader's complete price quarantine/calendar policy. Saved benchmark state files contain **6,726** dates, while cap-weighted state files contain **6,740**. There are 14 extra dates, including September 11, 2001 and several Sundays in 2024, absent from the supplied benchmark calendar.

Needed: audit these rows and align the research input policy with the supplied market calendar before treating its regimes as exchange-session signals. This is a data-quality issue, not evidence of an executable return on those dates. The verbatim loader already quarantines off-calendar prices; the separate SQL research path bypasses that safeguard.

### 7. Buffer trading and final-test capital need an explicit, fair boundary

`docs/FactorStrategy.md`, `docs/PairsTilt.md`, and the research pairs example start funded simulations on January 4, 2016. The factor documentation proposes building sleeves during the buffer and reporting from April.

Calling `performance()` on such a result includes the buffer, and slicing dates alone does not establish the required comparison of $1,000,000 against $1,000,000 invested in `^SP500TR` at the first final-test close. Initial holdings and equity on April 1 must not be silently substituted for the specified starting condition.

For straightforward compliance, start the funded main account on April 1 with $1,000,000 cash; use earlier history for model lookbacks and shadow-account warm-up. If an alternative initial-position convention is proposed, it needs explicit justification and an equivalent capital/benchmark comparison rather than being assumed compliant. This audit does not approve that alternative.

## Criterion-by-criterion status

“Implemented” below means code/evidence exists; it does not certify a final-period run.

| Assignment requirement | Current evidence | Status / remaining work |
|---|---|---|
| Personal submission folder | `Homeworks/HW4/David_MartinsHuber/` exists | Correct location. |
| Independently written backtester | Modular engine, OMS, strategies, analytics; design and AI attribution documented | Implemented; execution caveat in finding 3. |
| Full bCourses dataset for final experiment | Local full DuckDB; `^SP500TR` covers 2000-01-03–2026-09-30 | The final-period run completed; stale marks and open lots materially limit confidence in its reported performance. |
| Dated quarterly universe, including inactive names | Engine membership gate and effective membership inputs; no active-snapshot filter found in inspected strategy/engine code | Implemented, but currency exclusion creates a separate future-selection problem. |
| Keep holdings after universe removal | Lots are independent of new-entry eligibility | Implemented; do not replace with forced stale exits. |
| Features available at prediction time | Filing + one-day availability and backward as-of panels | Implemented partly; future currency exclusion and source restatements remain. |
| Predict the 90-calendar-day trade return | FF5 strategy has next-session/maturity-based supervised returns | Candidate implementation exists; final predictor choice and research diagnostic mismatch need explanation. |
| Train only on completed outcomes | Engine training-panel cutoff and factor model maturity checks; HMM walk-forward fitting | Infrastructure/tests exist; self-fitting models still require model-specific evidence. |
| Learned preprocessing fitted on permitted data | Walk-forward research code and tests | Partially evidenced; freeze and document the final training/selection procedure. |
| Development in 2010–2015 with chronological validation | Development demo and research models exist | Missing one clearly reported chronological validation protocol and selection result for the final configuration. |
| Settle model/strategy before final evaluation | Several candidate strategies and configurable optimizers/risk policies | No final frozen experiment manifest found. |
| Final period April 1, 2016–September 30, 2026 | Full 2,640-session run and metrics in `Backtesting.ipynb` | Executed; performance remains exploratory because of the stale marks/open lots below. |
| $1m, long only, no leverage/double spending | OMS reservations, finite cash checks, long-only proposals, initial-capital settings | Implemented; show invariants for the actual final result. |
| Decide before close fill is known | Prior-close contexts and next-session execution barrier | Implemented/tested; maintain it in custom models and research inputs. |
| Hold each purchase 90 calendar days | Maturity-based lots and standard exits | Standard path implemented; optional last-price forced sale conflicts. |
| Missing data handled honestly | Quarantine warnings, rejected entries, stale diagnostics, default failure mode | The run used `max_stale_sessions=10,000`, with 234,681 stale marks and 16,328 terminal open lots; valuations require resolution before results are reliable. |
| Value open positions at terminal date without later prices | Daily NAV and terminal lots retained | Implemented; verify actual final coverage/marks on September 30, 2026. |
| State transaction-cost, cash-interest, adjusted-price assumptions | Docs specify costs, zero cash interest, adjusted units/no extra dividends | Present; final configuration must repeat the actual settings. |
| Compare with $1m in `^SP500TR` at first test close | Engine benchmark path and analytics | Implemented; required-period comparison missing and buffer convention unresolved. |
| Final value, CAGR, maximum drawdown, Sharpe with 2% RF for both | Required-period table saved in the executed notebook | Reported, but strategy value is materially exposed to carried stale marks. |
| One chart with both cumulative-return curves | Final-period chart saved in `Backtesting.ipynb` | Present; interpretation is limited by stale valuations. |
| Explain whether benchmark was beaten and by how much | Comparison API and development output | Final conclusion absent; outperformance itself is not mandatory. |
| Accounting and timing evidence | Test suites, synthetic notebook, earlier audit artifacts | Substantial; also attach checks for final cash/holdings/fills and actual missing-data incidents. |
| Explain features/model/validation/strategy and limitations | Detailed model/design docs, research notebook | Strong technical detail; needs one concise final narrative, selected configuration, unsuccessful ideas and interpretation. |
| Cite reused/AI-assisted work | Loader attribution, AGENTS rule, documentation | Present; keep citations accurate for additional reused preprocessing/research methods. |
| Keep databases/large generated files out of Git | DuckDB and `.venv` ignored | Confirmed for local database/environment. Select representative results deliberately. |
| Submit complete work through PR and assign reviewers | Engine-only commit exists; research/notebooks/docs/strategy files remain untracked | Submission incomplete in Git. PR/push/reviewer status was not checked. |

## Evidence and test execution

- Reviewed assignment, source modules, README/model docs, both root notebooks, generated research filenames/configuration, and Git status.
- The two loader functions still match the lecture source verbatim.
- All three notebooks pass format validation and Python-cell syntax parsing. This is **not** clean-kernel execution of the final experiment.
- Confirmed local benchmark identity `^SP500TR` and coverage through September 30, 2026 with read-only SQL.
- Confirmed the full database and local virtual environment are Git-ignored.
- Independently reproduced findings 3–5 and inspected stored CSV dates for finding 6. These findings do not require a full expensive portfolio run to demonstrate.
- Full-suite command: `../.venv/bin/python -m unittest discover -s tests -q` from the submission folder.
- Full-suite result: **512 tests passed**, exit code 0, in **529.127 seconds** (about 8 minutes 49 seconds). The run emitted dependency deprecation, convergence, intentional partial-coverage, and numerical-boundary warnings; no test failure was reported.
- Source hashes detected concurrent changes to `backtesting/portfolio.py` and `backtesting/__init__.py` after the initial snapshot/full-suite start. The later diff adds Ledoit-Wolf covariance support and changes the minimum-variance default, so the 512-test run is evidence for the earlier imported snapshot, not a certification of every later edit. Follow-up optimizer checks are recorded below. This audit itself edited only this report and its README link and did not overwrite that work.
- The broad dependency ranges in `requirements.txt` are not a frozen environment, so record tested package versions when finalizing reproducibility.
- Follow-up optimizer command: `../.venv/bin/python -m unittest -v tests.test_portfolio tests.test_erc`. **33 tests passed**, exit code 0, in 3.704 seconds after inspecting the concurrent diff. This is a targeted regression check, not a repeat of all 512 tests or a complete independent audit of the new covariance estimator.

Passing component tests cannot substitute for a reproducible final evaluation. Some tests explicitly validate the optional last-price exit/global currency-removal behavior, so a green suite would not make those choices compliant with the homework directives. Earlier independent audits cover narrower snapshots; their “passed” statements are not blanket approval of later research additions.

No full 2000–2026 research recomputation or funded 2016–2026 experiment was executed during this audit. Source-vintage truth, hidden future knowledge in arbitrary external models, and statistical research quality are not established by these checks.

## Minimum route to a reviewable submission

1. Choose the smallest final strategy/portfolio that is ready to explain. Extra models are optional.
2. Resolve the currency-selection and missing-price execution issues; correct or clearly separate the research forward-return diagnostic and off-calendar proxy data.
3. Record development folds, training/feature availability, model/optimizer/risk settings, refit rules, seeds, costs, and the final configuration. The research notebook already exposes full-period forward state statistics; disclose whether these informed selection. Walk-forward fitting alone does not establish untouched final-period model selection.
4. Make the final runner self-contained. Remove undeclared notebook dependencies and regenerate outputs from a clean kernel for the exact selected components.
5. Run the main final interval with the required initial capital and benchmark boundary. Keep coverage exact and `allow_partial=False`. Use lookback/shadow history separately from funded test performance.
6. Save a compact final report: required metrics for both portfolios, comparison in dollars/percentage points, cumulative-return chart, and final NAV/cash/open-lot reconciliation. Check signals precede buys, exits meet maturity and use observed quotes, cash never goes negative, and no post-cutoff price is consumed. Explain unresolved gaps instead of fabricating exits.
7. Add a short explanation of what worked, failed, and would be tried next; disclose limitations and reused/AI-assisted work. Do not claim exact point-in-time source vintages or fully survivorship-free coverage.
8. Review and commit the intended remaining submission files, including the final results and run instructions. Keep raw databases/environment/irrelevant bulk outputs out of Git. Submit a PR and assign `kevinramlal` and `raghuramkowdeed` as required. This audit neither commits nor pushes those files.

The goal is correct, understandable evidence for the assigned experiment. A portfolio that fails to beat the benchmark can still satisfy the stated pass/no-pass requirements.
