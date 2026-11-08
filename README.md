# HW3: Memoization and Binary Tree Traversal

`hw3.py` implements the three-second memoization decorator and recursive and
iterative versions of pre-order, in-order, and post-order tree traversal.

## Run

Use Python 3.12 or later, then run:

```bash
python test_hw3.py
```

## Implementation Notes

Part1
Each decorated function has its own cache, and each argument has a separate
result and timestamp. A result is cached after the function successfully
finishes and remains valid while its age is less than three seconds. At exactly
three seconds it is recomputed. Cache hits do not reset the timestamp, and
failed calls are not cached.

Part2
The iterative traversals use explicit stacks to track pending work. Pre-order
pushes the right child before the left so the left is visited first. In-order
stores waiting ancestors while moving down the left side, then visits each
ancestor before its right subtree. Post-order stores `(node, ready)` pairs so a
node is visited only after both children. Every traversal returns a new list,
and an empty tree returns `[]`.
