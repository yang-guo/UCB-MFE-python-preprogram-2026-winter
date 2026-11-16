# HW4 final submission audit

Audited November 15, 2026, against `Lectures/Lecture 8/hw4.md`,
`Lectures/modeling/README.md`, the Lecture 7 modeling examples, and the Lecture 8
backtesting and research-workflow examples. The audited scope is the explicit
`submission_files.txt` list, not every file in the research working directory.

**Recommendation:** the technical submission is ready for a pull request with
the methodological and source-data limitations already disclosed. No unresolved
implementation or calculation defect was found. The requirement mapping below
and the report define the scope of this assessment.

## Requirements and course conventions

| Requirement or convention | Evidence reviewed | Assessment |
|---|---|---|
| Full homework database; historical quarterly membership, including inactive stocks | Read-only full database; dated membership checked for all 2,325 purchases | Satisfied |
| Historical inputs and filing availability | Market-calendar features; same-date peer ranks; filing+1; invalid filing events excluded; 550-day staleness rule | Satisfied within vendor-vintage limitations |
| No future-survival or complete-outcome entry filter | Prediction eligibility does not use targets; 4,858 unknown targets retained among 224,163 forecasts | Satisfied |
| Predict the actual holding return | Next-session entry and 90-calendar-day labels; maturity tests | Satisfied |
| Chronological fitting and training-only preprocessing | Pre-2013 fit with 51,633 completed examples; last training label December 3, 2012; saved imputer, scaler and coefficients | Satisfied |
| Development-period model and strategy comparisons | Selected continuous 2013–2015 comparisons; separate treatment of annual reset diagnostics | Documented |
| Optional retraining | Fixed-model policy explicitly selected; monthly feature refresh without retraining | Permitted by HW4 |
| Settle model and strategy before running the final test | Fixed rules for the reported run; evaluation scope explained in report 23 | Qualification; sequence not certified |
| Own backtester; $1m, long only, no leverage | Project lot/event ledger; 170 hand-case and regression tests | Satisfied |
| Decisions before fill price is known | Prior-close budgets, next-session close fills, dated entry and cash checks | Satisfied |
| 90-day sales and terminal open positions | 2,292 maturity sales checked; 32 terminal lots valued September 30, 2026 | Satisfied for scheduled trades |
| Mandatory corporate actions | Seven rename lot transitions and one cash-merger conversion; cited terms, original successor maturities and explicit cash-receivable policy | Explicit economic-accounting assumption; no invented discretionary fill |
| Costs, cash interest and missing prices | 5 bps per side; zero cash interest; five-session mark allowance; no unresolved held-price gap in completed account | Satisfied within declared assumptions |
| Benchmark timing and required metrics | $1m in supplied ^SP500TR at first test close; same endpoint; independent final value, CAGR, drawdown and daily 2% Sharpe | Satisfied |
| Cumulative-return chart and explanation | Executed report 23 with both curves, drawdown panel, findings and limitations | Satisfied |
| Unsuccessful ideas and future research | Selected feature, allocation and fitting-policy comparisons; learning guide | Satisfied |
| Code, Markdown, notebooks, dependencies and instructions | All submitted Python/JSON parses; Ruff reports zero findings; Markdown structure and notebook metadata checked; pinned packages match tested environment | Satisfied for documented Python 3.12/Windows workflow |
| Reproducibility | Clean file-only copy with empty generated caches rebuilt forecasts, full account and report; numeric outputs reproduced exactly | Satisfied in tested environment |
| Citations and AI assistance | Course adaptations identified in modules/report; external evidence cited; AI contribution disclosed | Satisfied |
| Submission hygiene | Explicit small file list; no database, virtual environment, cache, credential file or unrelated homework included | Satisfied by manifest; staging still pending |

## Verification scope

The clean-copy replay reproduced the forecast checksum, every daily NAV row,
entry-cap audit and performance table exactly. Independent reconstruction checks
2,641 daily values, 4,617 fills, eight corporate-action lot transitions, cash,
holdings, receivables, fees and original maturities. The maximum NAV difference
is approximately $0.0000000051, consistent with floating-point arithmetic.

The full test suite passes 170 tests. One clean-copy run emitted a Windows
physical-core-detection warning and used logical cores instead; no test failed.
The notebook contains four successfully executed code cells and an embedded
chart. The plot was visually inspected. The clean-copy audit reused the tested
Python environment; it did not claim a new dependency download or test another
operating system.

The documented price-unit corrections are applied
consistently to features and valuation. A wording inconsistency in inherited
repair metadata was corrected during this audit, and the selected entry scripts
were formatted consistently. Parsed-code comparisons and the replay establish
that formatting did not change the calculations. The saved model was not refit.

The price review independently compared 92 of 117 flagged large held-price
moves, with no discrepancy above 0.5 percentage point. Independent quotes were
unavailable for the other 25. Those observations remain an explicit limitation;
they are not automatically errors, and successful accounting does not certify
their economic accuracy. Static sectors, historical issuer aliases and restated
financial snapshots are also disclosed limitations.

## Result and final actions

The strategy finishes at $2,376,734.37 versus $4,402,746.78 for the benchmark.
Its CAGR is 8.60%, maximum drawdown is -38.50%, and Sharpe is 0.425; the benchmark
values are 15.17%, -33.79%, and 0.773. Beating the benchmark is bonus recognition,
not a pass requirement in the assignment.

Use the explicit file list when staging. Confirm the PR changes are confined to
`Homeworks/HW4/Sterling_Couts/`, with base `main` and comparison branch
`hw_4_Sterling_Couts`. Assign `kevinramlal` and `raghuramkowdeed` as directed by
HW4. Nothing was staged, committed, pushed or submitted by this audit.

The published due date is November 12, 2026. The course permits at most three
late days, subject to the student's remaining allowance. Verify the applicable
submission cutoff and remaining allowance; this audit cannot confirm the
student's late-day balance or an unstated due-time/timezone.
