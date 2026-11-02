# HW 3

Run: `python test_hw3.py`

**Cache expiry:** each decorated function has its own dict mapping each argument to its result and the `time.monotonic()` time at which the function returned. A result is reused while less than 3 seconds have passed and recomputed at 3 seconds or later. Expiry is only checked when that argument is requested again. Cache hits don't reset the time, and failed calls aren't cached because the exception is raised before anything is saved. I used `@wraps` so the wrapper keeps the original function's name and docstring.

**Recursive traversals:** an empty node returns `[]`. Otherwise the function joins the node's value with the recursive results for its left and right parts, in the traversal's order: Root, Left, Right for pre-order; Left, Root, Right for in-order; and Left, Right, Root for post-order.

**Iterative traversals:** a list is used as a stack of nodes still to visit, so there is no recursion and deep trees work. Pre-order pops a node, records it, then pushes right then left so that left is visited first. In-order pushes the current node and its left ancestors, then pops one, records it and moves to its right child. Post-order is pre-order with left pushed before right, which gives Root, Right, Left, and reversing that gives Left, Right, Root.
