# HW 3

## How to Run
Put `hw3.py` (the solution) and `test_hw3.py` in the same folder, then run:

    python test_hw3.py

## Approach

**memorize_3s**: A decorator storing each argument's (result, stored_at)
in a dict. A call returns the cached result if its age, measured with
time.monotonic(), is under 3 seconds; otherwise it calls the function
and stores a fresh timestamp. Hits never refresh it.

Explain when cache entries expire:
- If func() was ever called for x, then it has an entry in cache.
- When wrapper(x) is called, the cache for x is checked for expiry, i.e. whether func(x) was called within 3 seconds or not.
- An expired entry triggers a fresh func(x) call, and the new (result, now) overwrites the old one.
- The wrapper and its cache live for the life of the program.
- Each time the outer function runs, the def executes again, producing a new wrapper with a new, empty cache. When the outer scope ends and nothing references that wrapper, the wrapper and its cache are garbage-collected, disappearing with the enclosing scope.

**Tree traversals**: Each order is implemented both recursively and
iteratively.
- Recursive: concatenate left, right, and root values in the required
  order (pre-order, in-order, post-order).
- Iterative: uses an explicit stack, handle trees deeper than Python's normal recursion limit.
  - pre-order is achieved by recording the current node val, then track pending work by pushing the children nodes in a way that ensures left is popped before right
  - post-order is achieved similarly to pre-order except for a different popping order, and an additional trick of reversing the result list to get the right traversal
  - in-order uses 2-level nested-while loop and a position tracker (cur) to traverse the tree

Explain how iterative traversals track pending work: 

  - Pre-order: simply using a stack, always push right before left.
  - In-order: uses a position tracker, a stack, and nested while loop to reach as far as possible on the left before everything on the right.
  - Post-order: using a stack similar to pre-order, always push left before right.

