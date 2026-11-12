## Approach

Memoization: each decorated function has its own dictionary mapping `arg -> (result, finish_time)`. The time is recorded with `time.monotonic()` after the function returns. An entry is used while less than 3 seconds have passed since then. At 3 seconds or more, the function is called again. Cache hits don't reset the timer, and exceptions are not cached.

Iterative traversals: an external stack holds the nodes being processed.
- Pre-order: pop a node, record it, push right then left.
- In-order: push nodes while going left, then pop one, record it and move to its right child.
- Post-order: collect Root, Right, Left with a stack, then reverse the result.

Recursive traversals: simple base case and follows logic of each traversal by adding outputted lists.