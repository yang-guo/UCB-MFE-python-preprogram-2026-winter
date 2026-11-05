# HW3 — Antonio Rivera

## Run

From this folder (`Homeworks/HW3/Antonio_Rivera/`), with Python 3.12 or later:

```bash
uv run --no-project python test_hw3.py
```

(`python test_hw3.py` also works if `python` is 3.12+.) All 16 tests pass.

## Part 1: memorize_3s

**What the cache stores.** Each decorated function gets its own `cache` dict, created when
`memorize_3s` runs. It maps `arg → (result, timestamp)`, where `timestamp` is the
`time.monotonic()` reading taken just after `func(arg)` returns.

**When an entry expires.** On each call, if `arg` is cached, the wrapper checks
`time.monotonic() - timestamp < 3`. If that is true, it returns the cached result. Once
3 seconds or more have passed, the check fails. At exactly 3.0 s, `3 < 3` is false, so the
entry has expired. The wrapper then calls `func(arg)` again and replaces the entry with the
new result and a new timestamp. The timestamp is taken after the computation finishes, so a
slow function's 3 seconds start when it finishes, not when it was called. The wrapper checks
for `arg in cache` instead of whether the result is truthy, so falsey results like `0`,
`None` and `[]` are cached too.

**Why hits do not extend expiry.** A cache hit only reads the entry. The wrapper writes to
the cache only after it calls `func`. The timestamp therefore always marks when the stored
result was computed, so a value that is requested often still gets recomputed every 3 seconds.

**Why exceptions are not cached.** If `func(arg)` raises, the exception propagates out of
the wrapper before the line `cache[arg] = ...` runs. Nothing is stored, so the next call with
that argument tries `func` again rather than returning a failure. Any older entry for that
argument was already expired, so it is still ignored.

## Part 2: Traversals

### Recursive

All three use the same base case: `root is None` returns `[]`. Each one builds a new list
by joining the results for the left and right subtrees, and the only difference is where
`[root.val]` goes:

- pre-order: `[root.val] + left + right`
- in-order: `left + [root.val] + right`
- post-order: `left + right + [root.val]`

### Iterative

None of these call a function recursively. Python lists serve as explicit stacks, so the
depth of the tree is limited by memory rather than by Python's recursion limit.

**Pre-order.** The stack starts as `[root]` and holds nodes that have not been visited yet.
Each loop pops a node and records it immediately. Then it pushes `node.right` followed by
`node.left`. The stack is last-in, first-out, so the left child is popped next, which gives
Root, Left, Right.

**In-order.** `current` points to the subtree that still has to be explored. The stack
holds ancestors that were passed on the way down to the left but have not been recorded
yet. Each loop:

1. Moves `current` down the left side of its subtree, pushing every node it passes.
2. Pops the deepest one. That node has no unvisited left subtree, so it is recorded.
3. Sets `current = node.right`, so its right subtree is explored next.

The loop condition is `while stack or current` because either one can be empty while work
remains. After a node is popped, the stack can be empty while `current` is a right subtree
that has not been explored yet (for example, just after the root is recorded). And
`current` can be `None` (a node with no right child) while the stack still holds ancestors
waiting to be recorded. The traversal is finished only when both are empty.

**Post-order.** This uses the pre-order loop with the push order swapped. Each loop pops a
node and records it, then pushes `node.left` followed by `node.right`. The right child is
popped first, so the list is built in Root, Right, Left order. Reversing that list at the
end gives Left, Right, Root, which is post-order.
