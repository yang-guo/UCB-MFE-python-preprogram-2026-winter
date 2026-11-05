# HW3: Jing Jian

## Run the supplied checks

From this folder, using Python 3.12 or later:

```bash
uv run --no-project --python 3.12 python test_hw3.py
```

The supplied suite checks caching behavior and all six tree traversals.

## Approach

### Memoization

`memorize_3s` creates a dictionary inside the decorator, which its wrapper retains
as a closure. Each argument maps to a result and a completion timestamp measured
with `time.monotonic()`. A result is reused only when its age is less than three
seconds; at exactly three seconds it is recomputed. Cache hits do not change the
timestamp. Membership checks allow falsey results to be cached. The cache is
updated only after the wrapped function successfully returns, so exceptions
propagate without caching a failed call. Each decorated function has its own cache.

### Tree traversals

Each recursive traversal uses a local result list and a recursive helper. The
base case is a missing node (`None`). Appending the current value before, between,
or after the child calls produces pre-order, in-order, or post-order respectively.

The iterative traversals track pending work explicitly:

- **Pre-order:** A stack holds nodes still to visit. Push the right child before
  the left child so the left subtree is processed first.
- **In-order:** A stack holds ancestors while descending left. Pop and visit the
  next node, then explore its right subtree.
- **Post-order:** Stack entries contain a node and a flag. The first encounter
  schedules the node for later and schedules its children; the flag marks when
  both child subtrees have been processed and the node can be visited.

All traversals return a new list without modifying the input tree. Each takes
O(n) time and O(h) auxiliary space, excluding the O(n) output list, where n is the
number of nodes and h is the tree height. The iterative versions use no recursive
calls and can handle trees beyond Python's normal recursion depth.
