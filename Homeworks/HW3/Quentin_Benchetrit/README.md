# HW 3

## Running the tests

From the folder that contains `hw3.py` and `test_hw3.py`:

    python test_hw3.py

## When a cache entry expires (`memorize_3s`)

For each argument, the cache stores the result and the time the computation
finished (`time.monotonic()`). The cached result is reused while less than
3 seconds have passed since that finish time. At exactly 3 seconds, the
function is called again and the entry is replaced with the new result and
its new finish time. A call that hits the cache does not extend the expiry.
If the function raises an exception, nothing is cached and the exception
propagates to the caller.

## How the iterative traversals track pending nodes

All three use an explicit list as a stack instead of recursion, so they work
on trees deeper than Python's recursion limit.

- **Pre-order:** pop a node and record it, then push its right child and then
  its left child. The left child is on top, so it is processed first.
- **In-order:** walk left from the current node, pushing every node on the way.
  When there is no left child, pop a node, record it, and move to its right
  child.
- **Post-order:** do a mirrored pre-order that records Root, Right, Left
  (push the left child, then the right child, so the right is popped first).
  Reversing that list gives Left, Right, Root, which is post-order.
