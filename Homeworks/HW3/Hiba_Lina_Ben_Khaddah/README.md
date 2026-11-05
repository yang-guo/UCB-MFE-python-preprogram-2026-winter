# Homework 3 — Advanced Python

Hiba Lina Ben Khaddah

## Running the tests

From `Homeworks/HW3/Hiba_Lina_Ben_Khaddah/`, run:

```bash
python3 test_hw3.py
```

The implementation uses only the Python standard library.

## Memoization approach

`memorize_3s` keeps a separate cache for each decorated function. Each hashable
argument maps to its successful result and the time when the computation
finished. A cached result is reused while its age is strictly less than three
seconds; at exactly three seconds or later, the function is evaluated again.
Cache hits do not reset the timestamp, and failed calls are not cached.

## Tree traversal approach

The recursive functions follow the required visit order directly and return an
empty list when the tree is empty. The iterative functions use lists as stacks
to track pending work:

- Pre-order pushes the right child before the left child so the left child is
  processed first.
- In-order stores ancestors while moving left, then visits each ancestor before
  exploring its right subtree.
- Post-order uses two stacks. The first collects nodes in root-right-left order,
  and popping the second reverses that order to left-right-root.
