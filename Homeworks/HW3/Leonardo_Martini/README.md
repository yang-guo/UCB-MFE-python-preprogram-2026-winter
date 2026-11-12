# HW3 - Leonardo Martini

## How to run

From this folder:

```bash
uv run --no-project python test_hw3.py
```

All 16 tests pass.

## Part 1: memorize_3s

The decorator keeps a dictionary `cache` for each decorated function. For every argument it saves the time when the function finished (`time.monotonic()`) and the result.

When the function is called again with the same argument, I check if it is in the cache and if less than 3 seconds have passed since that time. If yes, I return the saved result. If not (new argument, or 3 seconds or more have passed), I call the function again and save the new time and result.

A cache hit does not change the saved time, so an entry always expires 3 seconds after it was computed. If the function raises an error, nothing is saved, because the line that saves to the cache is never reached.

## Part 2: tree traversals

The recursive versions return `[]` for an empty node and join three lists in the right order (for example, for pre-order: the node value, then the left subtree, then the right subtree).

The iterative versions use a list as a stack to keep track of the nodes still to process:

- Pre-order: I pop a node, add its value, then push the right child and then the left child, so the left one comes out first.
- In-order: I go down to the left as far as possible, pushing every node on the way. Then I pop one, add its value and move to its right child.
- Post-order: I do the same as pre-order but push the left child before the right one, which gives root, right, left. At the end I reverse the list to get left, right, root.

Since they do not use recursion, the iterative versions also work on very deep trees.
