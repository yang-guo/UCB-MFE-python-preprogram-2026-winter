# Independent audit — correction and verification round

Updated November 12, 2026. The first audit report and evidence are preserved in [round1/README.md](round1/README.md). Production code was corrected following that review, then five section reviewers independently retested the expected behavior and searched for additional counterexamples. Reviewers received the homework rules, user architecture decisions, source code, documented contracts, and earlier findings. They made no production or shared-test edits.

The risk-management extension was added after this audit round. Its separate regression/notebook validation and limits are described in [Risk.md](../docs/Risk.md); the five-agent findings below do not independently certify that extension. A later review found and corrected risk defects: risk review trusted the strategy's mutable planning snapshot, exact drawdown thresholds missed due to floating-point rounding, and crafted symbol or budget types could evade caps. The full suite now contains 201 tests, including adversarial snapshot, boundary, symbol-identity, numeric-type, policy-configuration, cash-convergence, and coordinator/engine phase regressions.

## Findings and corrections

| Area | Corrected behavior |
|---|---|
| Requested coverage | Requests outside supplied benchmark coverage fail by default. Explicit partial runs warn and expose requested/effective dates and clipping status. Weekends inside the supplied range are supported. The engine cannot infer exchange holidays beyond the supplied calendar. |
| Setup documentation | README and notebook now agree on the explicit local full-database path. The smaller class dataset requires a deliberate path change. Loading cutoffs and result coverage are visible. |
| Incremental lifecycle | Calls before `begin()` raise deliberate lifecycle errors. Promoting a partial-run warning to an exception leaves initialization uncommitted and permits a later valid run. |
| OMS fee/cash rules | Fees are restricted to 0–10,000 bps. Derived buy/sell values validate before any commit; unsupported numeric results leave account/order state unchanged. Large valid full-budget purchases use conservative representable sizing rather than relaxing the budget/borrowing invariant. |
| Benchmark normalization | Ratio-first arithmetic avoids overflow for large constant benchmark quotes. Unrepresentable growth fails with `FAILED`, not `COMPLETE`. |
| Portfolio arithmetic | Deployment fractions are calculated before scaling by main NAV. Derived allocations and totals must be finite. Invalid shadow returns are rejected before publication and optimizer invocation. |
| Analysis arithmetic | Derived returns, cumulative returns, required metrics, and complete benchmark comparison outputs validate for nonfinite arithmetic. This includes percentage-point scaling. Intentional undefined sample statistics remain `NaN`. |

During the second review, agents found additional counterexamples in the initial fixes: large-capital full-budget rounding, benchmark multiplication order, warnings-as-errors initialization, allocation multiplication order, shadow return overflow, and comparison percentage-point overflow. These were reproduced, corrected, and retested against the agents' fixed expectations. The expected cash rules were not weakened to make the tests pass.

No further defect was confirmed in the final tested paths. This is a bounded result, not a universal correctness certificate.

## Final evidence

- **33 regression tests passed at the time of this round** (529 today, including the later risk, market-data, ERC, and optimizer tests), including 13 cases added from audit findings. Original normal-range behavior remains covered.
- **Engine:** 15 independent 231-session ledger paths, strict label maturity, future-filing exclusion, detached histories, decision/execution separation, coverage/lifecycle safeguards, ratio-first normalization, and failure-state assertions pass.
- **OMS:** 200 independent high-precision Decimal-ledger scenarios pass. An additional **12,000 full-budget purchases** across capital from $1m to `1e300`, prices around $0.001–$1m, and fees 0–10,000 bps obey the strict budget cap, finite-value checks, nonnegative cash, and single-fill ledger reconciliation. Invalid buys, overflowing sales, and overflowing valuations reject atomically.
- **Portfolio:** 200 analytical two-asset capped/shrunk minimum-variance cases pass, maximum weight error about `2.22e-16`. Separately calculated shadow/main ledgers over 119 shadow and 100 main sessions match. Price-sensitive strategies and return-sensitive allocation pass future-data perturbations. Distinct 180-/365-day training windows obey separately calculated maturity/availability expectations. Strict/partial coordinated coverage and numeric-publication safeguards pass.
- **Analytics:** 300 cases pass independent least squares, manual quantiles, running-peak drawdown, scalar Sharpe, and log-based CAGR; maximum annualized-alpha discrepancy about `1.44e-14`. Additional overflow, short-sample, malformed-input, comparison, and plot-array checks pass.
- **Loader/demo/notebooks:** verbatim lecture-function parity remains exact. A real AAPL/BHRB slice matches the lecture output: 725 retained quotes and 275 invalid source observations unchanged. Independent ranking/tie/universe/cash/schedule/warm-up checks and documented setup agree. Every testing-notebook code cell executes headlessly, including the complete 33-test suite.
- **Mutation sensitivity:** at that time a passing clean temporary copy ran all 33 tests; the same six deliberately broken copies were detected. Import or discovery failures do not count. This remains six selected regressions, not a comprehensive mutation score.

## Reproduce from the submission folder

```bash
../.venv/bin/python -m unittest discover -s tests -v
../.venv/bin/python audits/engine_checks.py
../.venv/bin/python audits/oms_checks.py
../.venv/bin/python audits/portfolio_checks.py
../.venv/bin/python audits/analytics_checks.py
../.venv/bin/python audits/loader_notebook_checks.py
../.venv/bin/python audits/mutation_checks.py
```

The loader audit requires the local full database at `data/equities_top2000.duckdb`; other numerical checks use synthetic data. The preserved round-two scripts assert the expected corrections, rather than only printing known-bad behavior. Location/import setup was made portable without changing the independent expectations. `independent_results.json` records their rerun outputs and `mutation_results.json` records the mutation checks. The `round1/` archive describes the earlier source version; it is historical evidence, not a runner for that version against today's code.

## Objectivity and limits

Expected results come from homework/design rules, independently calculated cash ledgers, analytical constrained solutions, or independent regression/quantile calculations. Comparisons with the same engine are not treated as sufficient proof. The reviewers were explicitly allowed to fail the implementation and found additional cases before the final pass. Earlier evidence is retained for comparison.

No full real-data final-period strategy experiment, research-model performance assessment, original-financial-vintage audit, or large-component stress test was performed. The code cannot prevent arbitrary Python from loading future files or detect future knowledge hidden inside a supplied pretrained model. Historical feature provenance and final-period research discipline remain separate obligations. Passing these tests does not guarantee unseen floating-point inputs or arbitrary external mutation of public objects are safe.
