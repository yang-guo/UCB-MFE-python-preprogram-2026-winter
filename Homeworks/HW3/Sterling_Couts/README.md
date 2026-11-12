# HW3: Advanced Python — Sterling Couts

## Files and running the tests

- `hw3.py`: memoization decorator, the supplied `Node` class, and six tree traversals.
- `test_hw3.py`: supplied tests, unchanged.

Use Python 3.12 or later. No additional packages are required. From this submission folder, run:

```powershell
python -m unittest -v test_hw3
```

The full supplied suite passed all 16 tests during review. Running `hw3.py` alone defines the functions but does not run the tests.

## Memoization with expiration

`memorize_3s` returns a wrapper that retains a dictionary in a closure. Each decorated function has its own dictionary. Each argument is a key whose value contains the successful result and its completion timestamp.

A cached result is reused only when its age is strictly less than three seconds. At exactly three seconds or later, the wrapper recomputes the result. The timestamp comes from `time.monotonic()` after the underlying function returns successfully, so computation time does not consume the cache's validity period. Cache hits do not reset the timestamp.

Expired entries are checked when requested; no background timer removes them. Falsey results such as `0`, `False`, and `None` are valid cached results. Exceptions propagate without caching the failed computation. The wrapper supports the assignment's single positional, hashable argument.

## Recursive traversals

A missing node returns an empty list. For an existing node, recursive calls collect the left and right subtree results. The current value and subtree lists are combined in the requested order:

- Pre-order: Root, Left, Right.
- In-order: Left, Root, Right.
- Post-order: Left, Right, Root.

These functions return new lists and do not modify the tree. In-order traversal follows tree structure rather than sorting values. Repeated list concatenation can take quadratic time on a chain-shaped tree; these recursive implementations are intended for the assignment's small recursive test cases.

## Iterative traversals and pending work

Each iterative function uses a Python list as a last-in, first-out stack and does not call a recursive traversal.

- **Pre-order:** the stack holds nodes awaiting processing. The current node is recorded immediately. Its right child is pushed before its left child, so the left subtree is processed first.
- **In-order:** the stack holds ancestors awaiting recording while `current` descends left. After reaching a missing left child, the function pops and records a node, then explores its right subtree. Processing continues while either a current node or a pending stack entry remains.
- **Post-order:** the stack holds `(node, ready_to_record)` pairs. On the first encounter, the node is scheduled for later recording, followed by its right and left children. Stack order causes both subtrees to finish before the node is recorded. The marker is stored in the stack, not on the tree.

The iterative versions handle deep trees without increasing Python's recursion limit. Each takes linear time in the number of nodes, with auxiliary stack space proportional to tree height, in addition to the returned list.

## Validation

The supplied tests cover cache reuse, independent arguments and functions, exact expiration boundaries, completion-based timestamps, falsey results, and exceptions. Tree tests cover the example tree, empty and single-node trees, chains, repeated and unsorted values, repeated calls, unchanged inputs, and deep iterative traversal.

Passing the tests establishes the checked behavior; the implementation was also inspected for the required recursive and iterative approaches.
