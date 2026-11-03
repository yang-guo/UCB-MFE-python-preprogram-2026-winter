# HW 3: Advanced Python

## Run Command

Run the supplied test suite using:

```bash
python3 test_hw3.py
```

## Approach

### 1. Memoization with Expiry (`memorize_3s`)

Implemented a decorator that caches successful function results separately for each argument.

- Each argument has its own cached result and expiry timestamp.
- Uses `time.monotonic()` to measure elapsed time.
- A cached result is valid for **less than 3 seconds** after the underlying function successfully completes.
- At exactly 3 seconds or later, the cached entry expires and the function is executed again.
- Cache hits do not reset or extend the expiry time.
- Falsey return values (`0`, `False`, `None`, and empty lists) are also cached correctly.
- Exceptions are propagated and failed function calls are not cached.

### 2. Binary Tree Traversals

Implemented all three traversal algorithms recursively and iteratively. Each function returns a new list of node values and handles an empty tree by returning an empty list.

**Preorder (Root → Left → Right)**

- Recursive: Visit the root, then recursively traverse the left and right subtrees.
- Iterative: Uses a stack to track pending nodes. Pushes the right child before the left child so that the left child is processed first.

**Inorder (Left → Root → Right)**

- Recursive: Traverse the left subtree, visit the root, and then traverse the right subtree.
- Iterative: Uses a stack to track pending ancestors. Continues traversing left while pushing nodes onto the stack. Pops and visits a node when no further left child exists, then explores its right subtree.

**Postorder (Left → Right → Root)**

- Recursive: Traverse the left and right subtrees before visiting the root.
- Iterative: Uses two stacks. The first stack processes nodes in Root → Right → Left order, while the second stack reverses this order to produce Left → Right → Root.

### 3. Complexity Analysis

Let `n` represent the number of nodes and `h` represent the height of the tree.

| Traversal | Time Complexity | Auxiliary Space |
|---|---|---|
| Preorder Recursive | O(n) | O(h) |
| Preorder Iterative | O(n) | O(h) |
| Inorder Recursive | O(n) | O(h) |
| Inorder Iterative | O(n) | O(h) |
| Postorder Recursive | O(n) | O(h) |
| Postorder Iterative | O(n) | O(n) |

All iterative implementations use explicit stacks and do not rely on recursive function calls.