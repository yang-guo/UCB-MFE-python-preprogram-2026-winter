# HW3 - Dev Rao

## How to run

From this folder:

```
python test_hw3.py
```

(or `py -3 test_hw3.py` on Windows). All 16 checks pass and it ends with `OK`.

## Part 1: memorize_3s

Each decorated function gets its own dict, created inside `memorize_3s`, mapping `arg -> (result, finished_at)`.

When an entry expires: `finished_at` is read from `time.monotonic()` right **after** the function returns, so a slow call starts its 3 seconds when it finishes, not when it started. On the next call with the same argument, if `now - finished_at < 3` the cached result is returned. At exactly 3 seconds or later it recomputes and stores a fresh timestamp. A cache hit never touches the timestamp, so hits don't extend the expiry.

Some details:
- I check `arg in cache` instead of the cached value, so falsey results like `0`, `None` or `[]` still count as cached.
- If the function raises, the exception passes straight through before anything is written, so failed calls are never cached.
- Each argument has its own timestamp, so caching `f(2)` doesn't change when `f(1)` expires.

## Part 2: traversals

The recursive versions are the definitions written directly: for example pre-order is `[root] + pre_order(left) + pre_order(right)`, with `None` returning `[]`. Each call builds a new list, so nothing is kept between calls.

The iterative versions use a Python list as an explicit stack instead of the call stack. This is how each one keeps track of pending work:

- **Pre-order:** the stack holds nodes still to visit. Pop a node, record it, then push its right child and then its left child. The left child is on top, so it gets visited first.
- **In-order:** the stack holds nodes whose left subtree is being processed. Walk left from the current node, pushing every node on the way down. When you can't go further left, pop a node, record it, and continue from its right child.
- **Post-order:** each stack entry is `(node, children_done)`. The first time a node comes off the stack (`False`), I push it back as `(node, True)` and then push its right and left children. When it comes off with `True`, both children are already finished, so I record it. That gives left, right, root.

None of them recurse or change `sys.getrecursionlimit()`, so they handle trees deeper than the recursion limit (the test uses about 1,100 levels). The tree itself is only read, never modified.