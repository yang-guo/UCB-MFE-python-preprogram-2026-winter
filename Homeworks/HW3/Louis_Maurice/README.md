# HW 3: Advanced Python — Louis Maurice

## Running the checks

From this folder:

```bash
python test_hw3.py        # or: python3 test_hw3.py
```

All 16 supplied checks pass (`OK`).

## Part 1: `memorize_3s`

The decorator follows the wrapper pattern from Lecture 5 (`timer` / `timed` with
`@wraps`) and the dictionary cache from Lecture 6 (`fib_cache`). Each decorated
function has its own `cache` dictionary: key is the argument, value is
`(result, time the result was computed)`.

**When entries expire:** the time is read with `time.monotonic()` right after
the function returns. On the next call with the same argument, if 3 seconds or
more have passed, the function is called again and the entry is replaced.
Otherwise the cached result is returned. Cache hits don't change the stored
time, so they don't extend the expiry. If the function raises an exception,
nothing is stored.

## Part 2: Tree traversals

The recursive versions are the `preorder_traversal`, `inorder_traversal` and
`postorder_traversal` functions from the Lecture 6 notebook.

The iterative versions use a list as a stack, as in
`generate_parentheses_iterative` from Lecture 6
(`stack.pop()` in a `while stack:` loop).

**How they track pending work:** the stack holds the work still to do. Because
a stack is last-in, first-out, items are pushed in the reverse of the order they
should be handled.

- **Pre-order:** pop a node, record it, then push right and left (left is
  handled first).
- **In-order / post-order:** the stack holds `(node, visited)` pairs. When a
  node is first popped (`visited=False`), it is pushed back with
  `visited=True` together with its children, in reverse order: in-order pushes
  right, node, left, and post-order pushes node, right, left. When a node is
  popped with `visited=True`, its value is recorded.

`None` children are pushed and skipped when popped. Because no recursion is
used, deep trees don't hit Python's recursion limit.
