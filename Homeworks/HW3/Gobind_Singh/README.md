# HW3: Gobind Singh

Run the checks from this folder:

```bash
python3 test_hw3.py
```

## Memoization

`memorize_3s` keeps a private dictionary for each decorated function. A key is the single argument; the value is the successful return value and the `time.monotonic()` reading taken after that call returns. A later call reuses the stored result only when the elapsed time is strictly less than 3 seconds. At exactly 3 seconds the wrapper calls the function again and replaces the entry. A cache hit returns the stored value and leaves the original timestamp in place, so repeated hits do not extend the lifetime. If the function raises, the exception propagates and nothing is stored for that argument.

## Iterative traversals

Each iterative traversal walks the tree with an explicit list used as a stack and never calls itself.

- Pre-order pushes the right child, then the left child, so the left child is popped next. The node value is recorded when the node is popped.
- In-order walks left children onto the stack until it hits `None`, records the popped node, then steps to its right child.
- Post-order stores `(node, seen_children)` on the stack. The first time a node is popped, its right child and then its left child are pushed, and the node goes back on the stack marked seen. The second time it is popped, both children are already done, so its value is recorded.
