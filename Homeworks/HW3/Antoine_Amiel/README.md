# HW 3: Advanced Python — Antoine Amiel

Run the supplied checks from this folder:

```bash
python3 test_hw3.py
```

## Part 1: `memorize_3s`

Each decorated function gets its own dictionary mapping an argument to `(result, finished_at)`, where `finished_at` is read with `time.monotonic()` right after the underlying function returns.

- **Expiry:** an entry is valid while `time.monotonic() - finished_at < 3`. At exactly 3 seconds or later it is recomputed and replaced.
- **Cache hits** return the stored result without touching `finished_at`, so they never extend the expiry.
- **Falsey results** (`0`, `None`, `[]`, ...) are cached correctly because a hit is detected with `arg in cache`, not by testing the stored value.
- **Exceptions** propagate naturally: the result is stored only after the function returns, so a failed call leaves the cache unchanged.

## Part 2: Traversals

The recursive versions follow the definitions directly: an empty subtree returns `[]`, and each node combines its own value with the lists from its left and right subtrees in the required order.

The iterative versions use a Python list as an explicit stack, so their depth is not limited by the recursion limit:

- **Pre-order:** pop a node, record it, then push its right child before its left child so the left child is processed first.
- **In-order:** follow left children down from the current node, pushing each one. When there is no further left child, pop a node: its left subtree is finished, so record it and move to its right subtree.
- **Post-order:** run a pre-order variant that pushes left before right, giving Root, Right, Left. Reversing that list gives Left, Right, Root.
