# HW3: Memoization and Binary Tree Traversal

## Running the Tests

Using the course's `uv` setup, run:

```bash
uv run --no-project python test_hw3.py
```

Alternatively, run:

```bash
python test_hw3.py
```

## Part 1: Memoization with Expiration

The `memorize_3s` decorator uses a dictionary to cache each input's result and completion timestamp.

A cached result remains valid for **less than 3 seconds** after the original function successfully finishes. At exactly 3 seconds or later, the function is called again and the cache entry is updated. Cache hits do not reset the expiration time.

Each decorated function has its own cache. If the original function raises an exception, the failed call is not cached.

## Part 2: Binary Tree Traversal

The recursive implementations visit nodes by recursively processing their left and right subtrees.

The iterative implementations use explicit stacks to track pending work:

- **Pre-order (Root, Left, Right):** Uses one stack, pushing the right child before the left child so the left child is processed first.
- **In-order (Left, Root, Right):** Uses a stack to remember parent nodes while moving left. When the leftmost node is reached, nodes are popped, recorded, and their right subtrees are explored.
- **Post-order (Left, Right, Root):** Uses two stacks. The first processes nodes in Root–Right–Left order, while the second reverses that order to produce Left–Right–Root.

All traversal functions return a new list of node values and return an empty list for an empty tree.