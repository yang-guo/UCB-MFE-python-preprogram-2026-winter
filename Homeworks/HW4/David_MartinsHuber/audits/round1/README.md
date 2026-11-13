# Independent audit — November 12, 2026

Five separate reviewers covered engine/contracts, OMS, portfolio coordination/optimization, analytics, and loader/demo/notebook integration. Reviewers were asked to derive expected results from requirements and independent mathematics, seek counterexamples, distinguish documented constraints from bugs, and leave production code and the existing tests unchanged. The primary reviewer reproduced material findings and checked the existing test suite's sensitivity to deliberate regressions.

This is a bounded code audit, not proof that every strategy is causal or that the final homework experiment is valid. Production fixes have not been applied as part of this audit.

## Findings to address

| Priority | Finding | Evidence and implication |
|---|---|---|
| Medium | Requested data coverage is not enforced or disclosed | `backtesting/engine.py:154` intersects requested dates with available benchmark dates. Asking for the final HW4 end date with a shorter database can return a successful result ending years earlier. Confirm the required terminal market session explicitly or record/validate requested versus effective coverage; weekend end dates need careful handling. |
| Medium | README and notebook database paths disagree | README tells a fresh user to download the class database under `Lectures/modeling/data`, but `Backtesting.ipynb` loads the full database under the submission's `data/`. Current local setup works; following only the documented setup can fail. |
| Low | Incremental API calls before `begin()` expose internal errors | `prepare`, `advance`, and `finish` access uninitialized attributes and raise `AttributeError`, rather than a deliberate lifecycle error. Normal `run()` usage is unaffected. |
| Low | Fees above 100% break ledger reconciliation | OMS accepts such fees, then `oms.py:125` clips negative cash after a sale to zero. With $100 initial cash, 20,000 bps fees, and a flat-price round trip, cash is zero but initial cash plus fill flows is approximately -$33.33. Default five-basis-point fees are unaffected. |
| Low | Finite positive prices can produce nonfinite fills | A subnormal price `np.nextafter(0, 1)` passes input validation, but computed units/fees become infinite and the order is marked filled. Validate derived arithmetic before changing account state. This is outside realistic course equity prices. |
| Low | Finite NAV inputs can overflow derived metrics | NAV `[1e-300, 1e300, 1e300]` passes analytics input validation but produces infinite returns/CAGR and negative-infinite VaR. Derived arithmetic needs validation; ordinary tested returns produced no confirmed formula error. |

The coverage item is a requirement/reporting safeguard, not proof that range intersection is intrinsically an invalid generic backtesting convention. The notebook mismatch is setup drift, not a defect in the required verbatim loader.

## Evidence

- Existing automated suite: **20/20 passed**.
- Engine: **12 independently calculated stochastic ledger paths**, 189 sessions each, including fees, multiple lots, entry/mark gaps, maturity, and daily NAV; strict label-boundary and future-filing checks passed. Expected NAV did not come from the implementation's fills or OMS.
- OMS: **200 randomized normal-range sequences**, including reservations, shuffled fills, rejections, and sales, reconciled against an independent cash ledger. Four malformed-quote cases preserved state; provenance checks passed.
- Portfolio: **200 analytical two-asset optimization cases**, including covariance shrinkage and weight caps, passed against a closed-form scalar solution. Maximum observed weight error was about `2.22e-16`. A hand-calculated constrained allocation of $4/$6 matched exactly.
- Causality: a price-sensitive strategy and return-sensitive optimizer passed future price/calendar/membership perturbations. These complement the existing price-insensitive strategy fixture. Changed unfinished labels and truncated-run prefixes were also checked in the existing suite.
- Analytics: **250 random cases** checked against independent least squares, manually sorted quantiles, peak-by-peak drawdown, scalar Sharpe, and log-based CAGR. Maximum annualized-alpha difference was about `1.69e-14`. Short samples, alternative risk-free rates/frequencies, malformed inputs, comparison values, and plotted curves were inspected.
- Loader: both loading functions match the lecture verbatim. An `AAPL`/`BHRB` real-database slice through 2001 matched lecture outputs exactly: **725 valid quotes** retained; **275 invalid source observations** unchanged. Demo strategy cases and notebook syntax/format checks passed.
- Targeted mutation check: a passing clean copy first ran all 20 tests; six deliberately broken copies were then detected by the suite. The mutations bypassed reservations, exposed future prices, admitted immature boundary labels, replaced optimization with equal weights, reversed alpha's benchmark term, and disabled held-price-gap stops. Dependency/import/discovery failures were excluded as invalid evidence. This is six selected checks, not a comprehensive mutation score.

## Reproduce

From `Homeworks/HW4/David_MartinsHuber`, use the existing HW4 venv:

```bash
../.venv/bin/python -m unittest discover -s tests -v
../.venv/bin/python audits/engine_checks.py
../.venv/bin/python audits/oms_checks.py
../.venv/bin/python audits/portfolio_checks.py
../.venv/bin/python audits/analytics_checks.py
../.venv/bin/python audits/loader_notebook_checks.py
../.venv/bin/python audits/mutation_checks.py
```

The loader audit requires the local full database at `data/equities_top2000.duckdb`; the others use synthetic data. These files preserve the independent reviewers' checks; only import/file-location setup was made portable. `independent_results.json` records their rerun outputs; `mutation_results.json` records the selected mutation outcomes. Some known-bad boundary probes print observed behavior instead of asserting success; a script's zero exit code therefore does not certify that the printed defects are fixed.

## Test-quality gaps and limits

The original future-price test used proposals that ignored prices; the sensitive independent scenario is stronger. Comparing a one-shadow run with the same standalone engine is a regression check, not an independent accounting oracle. The randomized cash/NAV and analytical optimization checks provide different evidence. Preserve these independently derived expectations when converting findings into regression tests; do not weaken expectations to accommodate the code.

No full final-period strategy experiment, large-component stress test, source-vintage audit, or research-model quality assessment was performed. Interfaces cannot stop arbitrary Python from loading future files or a supplied pretrained model from already containing future knowledge. Point-in-time feature provenance remains a research obligation. Settings and strategy selection can also overfit the final period even when execution code obeys its cutoff.
