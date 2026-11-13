# HW3

```bash
python3 test_hw3.py
```

**Memoization:** `memorize_3s` keeps a separate cache for each decorated function, storing each argument's result and completion time using `time.monotonic()`. Results are reused for less than three seconds after successful completion and recomputed at three seconds or later. Cache hits do not extend expiry, falsey results are cached, and exceptions propagate without caching the failed call.

**Tree traversals:** Recursive versions combine the left and right subtree results with the current value in the required order. Iterative versions track pending nodes with a list used as a stack:

- **Pre-order:** Push right before left so the left child is processed first.
- **In-order:** Save ancestors while walking left, then pop and visit a node before exploring its right subtree.
- **Post-order:** Push left before right to build Root–Right–Left order, then reverse the result.

All traversals return a new list, return `[]` for an empty tree, and leave the tree unchanged. Iterative versions use no recursion.

All 16 supplied tests pass.
