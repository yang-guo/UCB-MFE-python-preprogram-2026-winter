# HW3 Submission

## Memoization

The `memorize_3s` decorator caches successful results by argument. Each cache entry records the result and the time the wrapped function finishes. An entry is reused for less than three seconds; cache hits do not extend its expiry. Failed calls are not cached.

## Tree traversals

The recursive traversals visit nodes in preorder (root, left, right), inorder (left, root, right), and postorder (left, right, root).

The iterative traversals use explicit stacks to keep track of nodes that still need processing. Preorder pushes the right child before the left child so the left child is visited first. Inorder stores ancestors while moving down the left side. Postorder reverses an intermediate traversal order to visit children before their parent.

## Run the checks

From this folder, run:

```bash
python test_hw3.py
```
