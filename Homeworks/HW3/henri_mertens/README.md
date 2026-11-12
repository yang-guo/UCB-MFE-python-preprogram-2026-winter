# HW3: Henri Mertens

## Run the tests

Use Python 3.12 or later. No additional packages are required.
From this folder, run:

```bash
python test_hw3.py
```

## Memoization

`memorize_3s` gives each decorated function its own dictionary. Each argument
maps to a result and the time the calculation finished, measured with
`time.monotonic()`. A cached result is reused only while its age is less than
three seconds. Cache hits do not restart the timer. Expired entries are replaced
when requested again; failed calls are not cached.

## Tree traversals

The recursive versions record each node before, between or after visiting its
children to produce pre-order, in-order or post-order respectively.

The iterative versions use explicit stacks:

- **Pre-order:** push the right child before the left so the left is visited first.
- **In-order:** save nodes while descending left, then record a popped node and
  move to its right child.
- **Post-order:** store a node with a flag that schedules it to be recorded after
  both subtrees have been processed.

Each traversal returns a fresh list, leaves the tree unchanged and takes O(n)
time for n nodes. Iterative traversal does not use recursion.
