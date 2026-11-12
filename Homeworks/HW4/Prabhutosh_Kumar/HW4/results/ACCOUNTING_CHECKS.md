# Full-run accounting audit

PASS: 3,747 buys and 3,620 sales checked.

- Every fill matches the raw database adjusted close.
- Every purchase uses the preceding market session's signal.
- Every purchase has entry-quarter membership available at the signal close.
- Every lot matures exactly 90 calendar days after entry.
- Every sale is the first valid quoted market close on or after maturity.
- Daily cash reconciles with the complete transaction ledger; cash never goes negative.
- Fees and cash flows match units times fill price independently.
- Daily and final NAV reconcile to cash plus holdings.
- First buy is April 1, 2016; no trade uses a post-test date.

These checks verify accounting and timing, not the economic accuracy of source prices or stale marks.
