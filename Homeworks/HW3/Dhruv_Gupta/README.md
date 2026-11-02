# HW 3: Dhruv Gupta

## How to run

From this folder, with Python 3.12 or later:

```bash
python3 test_hw3.py
```

No extra packages are required. On Windows, `py -3 test_hw3.py` is the same check. The suite should end with `OK`.

## When cache entries expire

`memorize_3s` gives each decorated function its own dictionary. The key is the single argument. The stored value is the successful result and the `time.monotonic()` reading taken after that call returns. The timer starts at completion, so time spent inside the function does not count toward the three seconds.

A later call with the same argument reuses the stored result only while the elapsed time is strictly less than 3 seconds. At exactly 3 seconds, and at any later time, the wrapped function runs again. The new result and its finish time replace the old entry. A cache hit returns the stored result and does not update the timestamp, so repeated hits do not extend the lifetime. A call for one argument does not change another argument's expiry.

If the wrapped function raises, the exception propagates and that attempt is not stored. The next call tries again. Results such as `0`, `False`, `None`, and `[]` are stored the same way as any other result.

## How the iterative traversals track pending work

Each iterative function uses a Python list as a stack. `pop()` takes the next node, and `append()` saves a child that still has to be visited. None of them calls a recursive helper. Each call builds a new list and leaves `val`, `left`, and `right` unchanged. An empty tree returns `[]`.

Pre-order starts with the root on the stack. The loop pops a node, records it, then pushes the right child and the left child. The left child sits on top, so it is visited next. The stack holds the nodes whose subtrees are still pending, with the next one on top.

In-order uses the stack together with a current-node pointer. The inner loop walks left, pushing every node it passes. Those stacked nodes are the ancestors whose left side is finished and whose own value is not yet recorded. When the pointer is `None`, the loop pops one ancestor, records it, and moves the pointer to that node's right child.

Post-order starts with the root on the stack. The loop pops a node, records it, then pushes the left child and the right child. That visit order is root, right, left. Reversing the list once the stack is empty turns it into left, right, root. Until that reverse, the result list is the pending output and the stack is the pending nodes.
