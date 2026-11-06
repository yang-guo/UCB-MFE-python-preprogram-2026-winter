# HW3: Advanced Python

Run the supplied checks from this folder with:

```bash
python test_hw3.py
```

## 1. Cache Expiration

The decorator stores each function's result along with an expiration time. A cached result is valid for three seconds after the function finishes executing. If the same argument is passed again before expiration, the cached result is returned. Otherwise, the function runs again and the cache is updated. Cache hits do not reset the timer.

## 2. Iterative Tree Traversals

The iterative traversals use a stack to keep track of nodes that still need to be visited. Since a stack follows the Last-In, First-Out (LIFO) principle, the order in which nodes are pushed determines the traversal order.

- **Pre-order:** Push the right child before the left child so the left child is visited first.
- **In-order:** Keep moving left, storing nodes on the stack. Process each node when returning from its left subtree, then move right.
- **Post-order:** Traverse in Root, Right, Left order and reverse the result to get Left, Right, Root.
