# HW3: Advanced Python — Prabhutosh Kumar

## Files and execution

- `hw3.py`: the cache decorator, unchanged `Node` class, and six traversals.
- `test_hw3.py`: an unchanged copy of the supplied standard-library checks.

Use Python 3.12 or later. From this directory:

```bash
python3 test_hw3.py
```

Alternatively, use the course's uv command without syncing course dependencies:

```bash
uv run --no-project python test_hw3.py
```

From the repository root, the existing course environment can run the checks:

```bash
.venv/bin/python Homeworks/HW3/Prabhutosh_Kumar/HW3/test_hw3.py
```

There are no third-party dependencies, so no requirements file is necessary.
The implementation is AI-assisted; the supplied course tests are unchanged.

## Three-second memoization

Each decorated function closes over its own dictionary. A hashable argument
maps to a `(result, completed_at)` pair. Membership is checked independently of
result truthiness so `0`, `False`, `None`, and empty containers remain cacheable.

An entry is valid only when `time.monotonic() - completed_at < 3`.
Exactly three seconds means expired. A hit returns the stored result without
updating its timestamp. On a miss or expiry, the underlying function runs;
only after it returns successfully do we read the clock and store its result.
Thus slow computations receive a full three seconds after completion, and
exceptions propagate without storing a failed result. An older expired entry
may remain after a failure, but it cannot be used as a valid cache hit.

This supports one positional hashable argument, as required. There is no
background cleanup, thread synchronization, or multiple-argument support.
Cache lookup is expected O(1), excluding the wrapped computation, with O(k)
storage for k distinct arguments.

## Binary-tree traversals

Each call creates a fresh result list. Nodes and their links are never changed;
values are neither sorted nor deduplicated. Empty trees return an empty list.

The recursive versions use a local recursive visitor. Only the position of the
append changes: before both children for pre-order, between them for in-order,
and after both for post-order. Appending to one list avoids repeatedly copying
subtree lists.

The iterative versions use loops and explicit stacks, without recursive helpers:

- **Pre-order:** visit a popped node immediately; push right before left so the
  left child is processed first by the last-in-first-out stack.
- **In-order:** push the entire left path, then pop and visit its ancestor;
  continue from that ancestor's right child. The stack tracks pending ancestors.
- **Post-order:** stack entries hold a node and an expansion flag. An unexpanded
  node schedules its own later visit, then right and left child tasks. The left
  subtree executes first, the right second, and the expanded parent last.

All traversals take O(n) time for n nodes. The returned list uses O(n) space;
recursive call stacks or explicit stacks use O(h) auxiliary space for height h.
Iterative traversals do not depend on Python's recursion limit. Recursive
versions are intended for the small trees specified in the assignment.

## Validation

Run `test_hw3.py` to check cache expiry boundaries, independent entries/functions,
falsey results, exceptions, slow calls, all traversal orders, repeated values,
fresh results, unchanged inputs, and deep iterative trees. The supplied suite
contains 16 tests and should finish with `OK`.

Only this personal submission folder is needed for submission. Shared starters,
virtual environments, databases, and credentials are not included.
