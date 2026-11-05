# HW3: First Last

## How to run
From this folder:

    python test_hw3.py

All tests should finish with `OK`.

## Part 1: Memoization (`memorize_3s`)
The decorator keeps a dictionary called `cache` for each function it wraps, so
different functions never share results. The dictionary maps each argument to a
pair: `(result, time the function finished)`.

When the wrapped function is called with an argument:
- If the argument is in the cache and `time.monotonic() - saved_time` is less
  than 3 seconds, the saved result is returned and the saved time is not changed.
  A cache hit does not extend the expiry.
- Otherwise (never seen, or 3 seconds or more have passed), the real function is
  called and the new result is stored with a new timestamp.

Entries expire exactly at 3 seconds: an age of 2.999 is still valid, an age of
3.0 is not. The timestamp is taken after the function returns, so a slow
function's 3 seconds start when it finishes. If the function raises an
exception, it passes up to the caller and nothing is stored, so failed calls are
not cached. Expiry is only checked when the same argument is requested again.
I check `arg in cache` instead of the result's value, so results like `0`,
`None`, or `[]` are still cached.

## Part 2: Tree traversals
The recursive versions call themselves on `root.left` and `root.right`, and an
empty tree (`None`) returns `[]`.

The iterative versions use a loop and a list as a stack to track the nodes that
still need to be handled, so they do not hit Python's recursion limit.

- **Pre-order:** The stack starts with the root. Pop a node, add its value to the
  result, then push its right child and then its left child. The left child is
  pushed last, so it is popped first.
- **In-order:** The stack holds nodes we have walked past but not yet recorded.
  Keep going left, pushing each node. When there is no left child, pop a node,
  record it, and move to its right child.
- **Post-order:** Two stacks are used. The first stack walks the tree (pop a
  node, push its left child, then its right child) and each popped node is
  pushed onto the second stack. This puts the nodes in Root, Right, Left order
  on the second stack. Popping the second stack gives Left, Right, Root.
