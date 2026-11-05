# HW3 — Suge Li

## Run

From this folder:

    uv run --no-project python test_hw3.py

(Or `python test_hw3.py` / `python3 test_hw3.py` where applicable.)

## Memoization (`memorize_3s`)

`memorize_3s` wraps a function that takes one positional, hashable argument.
Each decorated function gets its own `cache` dict, created inside the
decorator, so different functions never share entries. Each argument maps to
`(result, expiry_time)`.

**When entries expire.** After `func(arg)` returns successfully, I record
`expiry_time = time.monotonic() + 3`, where the clock reading is taken *after*
the call completes — the computation time itself is not part of the 3-second
window. On a later call, the entry is a hit only if
`time.monotonic() < expiry_time`, i.e. strictly less than 3 seconds after the
completion time. At exactly 3 seconds or later (`now == expiry_time` or
beyond), the entry is stale and the function is recomputed, overwriting the
old entry.

A cache hit returns the stored result and **does not** modify `expiry_time`,
so hits never extend an entry's lifetime (no sliding expiration). Hits are
detected with `arg in cache`, not by the truthiness of the value, so falsey
results (`0`, `False`, `None`, `[]`) are cached and returned correctly.

If `func(arg)` raises, the exception propagates unchanged and nothing is
written to the cache — the write happens only after a successful return.
`time.monotonic()` is looked up through the `time` module at call time (kept
`import time`), so tests can control the clock.

## Traversals

All six functions return a new list of values (`[]` for an empty tree), never
modify the tree, and keep no state between calls.

### Recursive versions

Each recursively visits left/right subtrees and places `[root.val]` according
to the order:

- **Pre-order**: root, left, right → `[root.val] + left + right`
- **In-order**: left, root, right → `left + [root.val] + right`
- **Post-order**: left, right, root → `left + right + [root.val]`

### Iterative versions (loop + explicit stack)

These avoid recursion entirely and use a Python list as an explicit stack, so
they handle trees deeper than Python's recursion limit without raising it.

**How pending work is tracked.** The stack holds nodes that have been
discovered but not yet visited; each loop iteration pops one and decides what
to push next.

- **Pre-order**: start with `[root]`. Pop a node, visit it, then push its
  **right child first, then left** — because the stack is LIFO, the left child
  is popped (and visited) next. This reproduces "root, left, right".

- **In-order**: use a `node` pointer to walk left, pushing each node onto the
  stack until reaching `None` (the leftmost node is now on top). Pop and visit
  it, then move the pointer to its right subtree and repeat. The stack holds
  the ancestors whose left subtrees are being / have been processed. The outer
  loop runs while `stack or node`.

- **Post-order**: generate "root, right, left" by popping a node, visiting it,
  and pushing its **left child first, then right** (mirror of pre-order). Then
  reverse the resulting list. Reversing "root, right, left" yields
  "left, right, root", which is exactly post-order. This needs no recursion
  and no visited-flags.