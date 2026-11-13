# Indexed pending-sale lookup

`Backtesting.ipynb` uses the coordinator in `backtesting.indexed_sales`. It
reuses the original coordinator and engines, replacing only their fresh OMS
instances. The alternative implementation never patches existing modules or
changes an already-running account.

The OMS retains every order for accounting, but checks for duplicate pending
sales using a lot-ID dictionary. Successful fills, rejected sales, and discarded
sales remove the corresponding index entry. Failed fills preserve it. Maturity,
quote validation, fees, reservations, refits, and execution barriers are inherited
unchanged. Direct external mutation of order statuses is unsupported; use the
OMS methods so its index stays synchronized.

## Verification

Six dedicated checks passed: full single-account and shadow-portfolio ledger
equivalence, duplicate/maturity/quote guards, index cleanup, absence of historical
iteration, and session rollback after an injected successful-sale-then-failure.
Seven existing accounting/backtesting checks also passed against the alternative
classes in an isolated test namespace. The original engine, OMS, and portfolio
module matched their pre-change SHA-256 hashes after verification. The submitted
notebook now imports this coordinator and contains the completed final-period run.

```bash
../.venv/bin/python -m unittest tests.test_indexed_sales.IndexedSaleChecks -v
../.venv/bin/python audits/indexed_sales_checks.py
```

The synthetic microbenchmark in `audits/indexed_sales_results.json` measured
100 submit-and-discard operations per book size. At 50,000 historical completed
orders, the original check averaged about 5.3 ms versus 0.005 ms for the index.
This isolates a bookkeeping operation; it is **not an end-to-end runtime claim**.
A full final-period run subsequently completed in the submitted notebook; its
results and stale-valuation caveat are documented there.

The pending-sale optimization reduces historical-order lookup cost. Strategy
model refits remain a separate source of runtime.

## Date progress

New runs print the first completed main-account session, every 63 completed
sessions, and the final session. Records include the simulated date, completed
and total sessions, percentage of sessions, and elapsed seconds. Session
percentage is not a remaining-runtime estimate. Use `progress_every=21` for
roughly monthly updates; model fitting between completed sessions can take time
without advancing the date counter. Eight indexed-sale and progress checks pass,
including exact ledger equivalence with the original implementation.
