# HW3: Advanced Python

Run the supplied checks from this folder with Python 3.12 or later:

```bash
python3 test_hw3.py
```

`memorize_3s` stores a result and its completion time separately for each argument. A result expires three seconds after the wrapped function returns; cache hits do not change that time. Exceptions are raised without caching the failed call.

Each traversal has a recursive and an iterative version. Pre-order uses a stack of nodes, adding the right child before the left. In-order keeps the path to the next leftmost node on a stack. Post-order stores each node with a flag, so the node is visited after its children. Every call returns a new list.
