# HW 3: Nirmit Ghughu

## How to run

From this folder:

```bash
python test_hw3.py          # or: python3 test_hw3.py
```

The suite (16 tests) ends with `OK`. `test_hw3.py` is the supplied file, unchanged.

**AI use:** the implementation in `hw3.py` was written with an AI assistant (Claude), following the instructor's guidance for this assignment. I reviewed and tested it.

## Part 1: `memorize_3s`: when cache entries expire

Each decorated function gets its own dictionary, created when `memorize_3s` is called, that maps an argument to `(result, finished_at)`.

- `finished_at` is read from `time.monotonic()` **after** the underlying function returns. A slow call's three seconds therefore start when it finishes, not when it starts.
- On a later call, an entry is used only if `time.monotonic() - finished_at < 3`. At exactly 3 seconds or later, the function is called again and the entry is overwritten with the new result and a new finish time.
- A cache hit does not change `finished_at`, so repeated hits never extend the expiry.
- Each argument has its own entry and its own timestamp, so caching `f(2)` does not affect `f(1)`.
- Hits are detected with `arg in cache`, not by checking the cached value. Falsey results (`0`, `False`, `None`, `""`, `[]`) are still cache hits.
- The cache is written only after the function returns. If it raises, the exception propagates and nothing is stored, so the next call retries.

## Part 2: Traversals: how the iterative versions track pending work

The recursive versions follow each order's definition directly. Each returns `[]` for `None` and otherwise concatenates the root's value with the results for the left and right subtrees, in the order the traversal requires.

The iterative versions use a Python list as an explicit stack instead of the call stack. That way they also work on trees deeper than the recursion limit.

- **Pre-order (Root, Left, Right):** the stack holds nodes that still need to be visited. Pop a node, record it, then push its **right** child before its **left** child. The stack is last-in-first-out, so the left subtree is processed first.
- **In-order (Left, Root, Right):** the stack holds ancestors whose own value and right subtree are still pending. Walk left from `current`, pushing every node. When `current` becomes `None`, pop the leftmost pending node, record it, and continue from its right child. The loop ends when there is no current node and the stack is empty.
- **Post-order (Left, Right, Root):** the stack holds `(node, expanded)` pairs. The first time a node is popped (`expanded=False`), it is pushed back as `(node, True)`, followed by its right child and then its left child. Those pop first, so both subtrees finish before the node comes up again with `expanded=True`, and only then is its value recorded.

None of the functions modify the tree. Each call builds and returns a new list.
