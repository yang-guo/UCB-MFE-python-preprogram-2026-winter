# HW 3: Advanced Python — Min Jin

## Files

- `hw3.py`: my implementation of `memorize_3s` and the six tree traversals
- `test_hw3.py`: the supplied checks, unchanged

## How to run

From this folder (`Homeworks/HW3/Min_Jin/`), using Python 3.12 or later:

```bash
uv run --no-project --python 3.12 python test_hw3.py
```

All 16 tests pass (`OK`).

## Part 1: `memorize_3s`

The cache lives in the main `memorize_3s` function, and it runs once per decorated function, so each decorated function gets its own separate cache. Each cache entry stores `(result, time.monotonic())`. The first is the result and the second is the timestamp of the `func` finishing computing the result, and cache hits don't change it. The entry is valid when the difference between its stored timestamp and the current access timestamp is less than 3 seconds.

## Part 2: Tree traversals

### Recursive versions

In the recursive version, the base case is always when it hits the empty node and then return an empty list. All pre-, in-, and post-order recursively traverse through the left subtree first and then the right subtree, but the difference is the timing of recording the current node's value:
- pre-order: before both subtrees
- in-order: between them
- post-order: after both   

### Iterative versions

The iterative traversal would require the use of stack so that we can track the most recent element using `.pop()`. In pre-order traversal, a node's right is pushed and then the left is pushed so that when we pop the stack, the left subtree is always added to the result list first, resulting in the root->left->right order. In in-order traversal all nodes along the path going left from the current are pushed onto the stack, pops the last entry when the traversal hits an empty node, records its value, and then current moves to that popped node's right node so that the traversal follows left->root->right. In post-order, since the order is the reversal of root->right->left, we first traverse like pre-order but pushes left side onto the stack first so that we would then always pop the right side first. After we recorded the values in the root->right->left order, we can just reverse the list to obtain the correct left->right->root order. The iterative versions work on very deep trees because the stack is an ordinary Python list, so there are no recursive calls and very deep trees can't hit Python's recursion limit. 