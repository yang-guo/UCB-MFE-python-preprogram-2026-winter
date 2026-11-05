# HW3: Wen Sun

## Run command

From this directory, with the homework environment activated:

    python test_hw3.py

All 16 supplied checks pass.

## When cache entries expire

`memorize_3s` stores, per decorated function, a dict mapping each hashable
argument to `(result, finished_at)`. `finished_at` is read with
`time.monotonic()` immediately after the wrapped function successfully returns,
so the 3-second window starts when the computation finishes.

Every call reads the clock again via `time.monotonic()` and treats an entry as
valid only while `now - finished_at < 3`. At exactly 3 seconds or later the
entry is recomputed and replaced. Cache hits do not update `finished_at`, so
hits never extend expiry. Exceptions propagate before anything is stored, so
failed calls are not cached. Falsey results are returned as stored.

## How the iterative traversals track pending work

All iterative versions use an explicit Python list as a stack and a `while`
loop, with no recursive helpers and no calls to the recursive versions.

- Pre-order: pop a node, record it, push right before left so the left subtree
  is processed first.
- In-order: descend left pushing nodes until none remain, pop and record the
  leftmost pending node, then move to its right subtree; loop while the current
  node is non-`None` or the stack is non-empty.
- Post-order: do a Root-Right-Left traversal (push left before right), then
  reverse the collected list to get Left-Right-Root.

Because pending work lives on an explicit stack, the iterative versions handle
trees deeper than Python's default recursion limit without raising it.