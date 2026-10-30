# HW 3: Farshid Gilak

## Running the checks

From this folder:

```bash
python test_hw3.py
```

## Part 1: `memorize_3s`

Each decorated function gets its own `cache` dictionary, created when the decorator is applied. The dictionary maps an argument to a `(result, finished_at)` pair, where `finished_at` is the `time.monotonic()` reading taken right after the underlying function returns.

On each call, if the argument is in the cache and `time.monotonic() - finished_at < 3`, the stored result is returned. Otherwise the function is called again and the entry is replaced with the new result and a new timestamp. So an entry expires exactly 3 seconds after its computation finished. Cache hits do not update the timestamp, so they do not extend the expiry. Expiry is only checked when the same argument is requested again.

Membership is checked with `arg in cache` rather than by testing the stored result, so falsey results such as `0`, `None`, or `[]` are still cache hits. If the function raises, the exception propagates before anything is written to the cache, so failed calls are not cached.

## Part 2: Binary tree traversals

The recursive versions follow the definitions directly: an empty tree returns `[]`, and otherwise the node's value is combined with the left and right subtree results in the required order.

The iterative versions use a Python list as an explicit stack of pending nodes:

- **Pre-order:** Start with the root on the stack. Pop a node, record its value, then push its right child and then its left child, so the left subtree is popped and visited first.
- **In-order:** Walk left from the current node, pushing each node onto the stack. When there is no further left child, pop a node, record its value, and move to its right child. The stack holds the ancestors whose values and right subtrees are still pending.
- **Post-order:** Same as pre-order, but push the left child before the right child. This visits nodes in Root, Right, Left order, which is the reverse of post-order, so the result list is reversed at the end.

Because the iterative versions store pending work in a list instead of the call stack, they handle trees deeper than Python's recursion limit.
