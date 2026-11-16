# HW4 requirement checklist — selected strategy

Authority: `Lectures/Lecture 8/hw4.md` and `Lectures/modeling/README.md`.
Verified describes the evidence below, not certification of every source row.

| Requirement | Evidence | Status |
|---|---|---|
| Full homework database and historical membership | Source hash and dated entry checks; 2325 purchases checked | Verified |
| 90-day target, next-session entry and completed training labels | Frozen fit chronology; forecast/lot tests and independent maturity checks | Verified |
| Training-only preprocessing and declared update policy | Saved pre-2013 pipeline, 64,459 development forecasts reproduced; no evaluation fit | Verified |
| Settle model and strategy before running the final test | Fixed rules for the reported run; evaluation scope explained in report 23 | Qualification noted |
| Explain features, model, validation and strategy | Report 23 and learning guide | Verified |
| Own long-only, unlevered $1m backtester | Lot/event engine; cash, holdings and receivables independently reconstructed | Verified |
| Costs, missing-price policy, cash interest and events | Explicit assumptions and cited registries; no unresolved held-price gap | Verified within stated assumptions |
| Final period April 2016–September 2026, including terminal holdings | Complete 2641-row account; 32 open lots marked at end | Verified |
| Same-period S&P total-return benchmark | First test close entry, same endpoint, independent NAV comparison | Verified |
| Final value, CAGR, maximum drawdown, daily 2% Sharpe | `performance.csv`, independent metrics and report table | Verified |
| Plot both cumulative returns; explain result and risks | Report 23 and `comparison.png` | Verified; visual review recorded separately |
| Test accounting and timing | 170 tests; 4617 fills independently reconciled; error < $0.000001 | Verified |
| Inspect flagged large price moves | 92/117 independently compared; 25 unavailable | Reviewed; residual limitation documented |
| Discuss unsuccessful ideas and next research | Selected development comparisons and learning guide | Verified |
| Readable instructions, citations and AI assistance | README, report 23, annotated modules and source registries | Verified |
| Keep databases and large generated files out of Git | Explicit manifest excludes data/caches; clean-copy replay verified in FINAL_AUDIT.md | Verified by manifest; staging pending |
| Commit, push, PR and reviewer assignment | Separate approval required; reviewers specified in assignment | Not started |
