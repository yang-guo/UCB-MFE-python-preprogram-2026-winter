# HW3 — Advanced Python

## Run the tests

From this folder, run:

```bash
uv run --no-project --python 3.12.9 python test_hw3.py
```

## Memoization

Each wrapped function has its own dictionary. For each input, it stores the result and the time the original function finished. A result expires after three seconds. Reusing a saved result leaves its timestamp as it was. If the original function raises an exception, the exception passes through and the failed call adds nothing to the cache.

## Tree traversal

The recursive versions call themselves on the left and right children, then combine the results in the requested order. Each call returns an empty list when it reaches a missing node.

The iterative versions use a stack to track nodes still to visit. Pre-order records the current node first and pushes the right child before the left so the left comes off first. In-order saves nodes while moving left, then pops a node, records it, and moves to its right branch. Post-order collects values in root, right, left order and reverses the list to get left, right, root.

