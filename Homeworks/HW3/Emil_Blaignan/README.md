# HW 3: Advanced Python (Emil Blaignan)

## Run the checks

From this folder:

```bash
python3 test_hw3.py
```

All 16 checks pass (the output ends with `OK`).

## Part 1: `memorize_3s`

The decorator keeps one dictionary per decorated function, mapping each argument to `(result, expiry_time)`.

- After the underlying function successfully returns, I read `time.monotonic()` and store `expiry_time = finish_time + 3`.
- On a later call with the same argument, I read the clock again. If `now < expiry_time`, the cached result is returned. At exactly 3 seconds or later, the function is called again and the new result and expiry replace the old entry.
- Entries are only checked when that argument is requested again, so there is no background cleanup.
- Cache hits never change the stored expiry, so they do not extend it.
- Membership is tested with `arg in cache`, so falsey results such as `0`, `None`, `False` and `[]` are cached correctly.
- The result is stored only after `func(arg)` returns, so if it raises, the exception propagates and nothing is cached.
- Because the expiry is measured from when the function finishes, slow computations still get the full three seconds.

## Part 2: Tree traversals

The recursive versions use a nested `visit(node)` helper that returns when the node is `None` and appends to a shared `result` list. The only difference between the three orders is where `node.val` is appended relative to the two recursive calls (before both for pre-order, between them for in-order, after both for post-order).

The iterative versions use a list as an explicit stack, so they do not depend on Python's recursion limit:

- **Pre-order:** the stack holds nodes still to be visited. I pop a node, record its value, then push its right child followed by its left child, so the left child is popped first.
- **In-order:** a `current` pointer and a stack. I keep pushing `current` and moving left, which stacks up the ancestors whose own value and right subtree are still pending. When there is no left child left, I pop a node, record it, and move to its right child.
- **Post-order:** I run a mirrored pre-order (root, right, left) by pushing the left child before the right child, then reverse the result at the end to get left, right, root.

Each iterative function returns `[]` for an empty tree and builds a new list on every call without changing the tree.
