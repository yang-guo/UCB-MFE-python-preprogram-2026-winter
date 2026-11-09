# HW 3: Qianhao Meng

Memoization with a 3 second expiry, and three binary tree traversals written both recursively and iteratively.

## Run the checks

Requires Python 3.12 or later. From this folder, with the course virtual environment:

```bash
../../../.venv/bin/python test_hw3.py
```

The run ends with `Ran 16 tests` and `OK`.

## Part 1: memorize_3s

- `memorize_3s` creates a `cache` dict for each decorated function. The key is the argument and the value is a `(result, finished_at)` pair.
- `finished_at` is read with `time.monotonic()` after the wrapped function returns, so the 3 seconds start when the computation finishes.
- On each call the wrapper checks `x in cache` and `time.monotonic() - finished_at < 3`. If both hold, it returns the saved result. At 3 seconds or later it calls the function again and replaces the entry with the new result and completion time.
- A hit only reads the entry, so it does not extend the expiry.
- `x in cache` decides a hit, so falsey results such as `0`, `False` and `None` are cached like any other value.
- The entry is written after `func(x)` returns, so a call that raises is not cached.

## Part 2: tree traversals

The recursive versions follow the lecture pattern. An inner helper stops at `None`, recurses into the left and right children, and appends to a `result` list created once per call to the outer traversal function and shared by its recursive helper calls. The three functions differ only in where the append happens: before the two recursive calls for pre-order, between them for in-order, and after them for post-order.

The iterative versions use lists as explicit stacks to track pending work:

- Pre-order: the stack holds nodes not yet visited. Pop a node, record it, then push its right child before its left child so the left child is popped first.
- In-order: the stack holds ancestors whose left subtree is not finished. Walk left from the current node, pushing each node on the way. When there is no left child, pop a node, record it, and continue from its right child.
- Post-order: two stacks. The first stack runs a root, right, left visit by pushing the left child before the right child. Each popped node goes onto the second stack. Popping the second stack at the end reverses that order into left, right, root.
