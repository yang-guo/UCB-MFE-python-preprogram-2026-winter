# HW3 – Ananya Das

This assignment covers memoization and binary tree traversals.

## Implementation

Implemented all seven functions in `hw3.py`:

- **Memoization:** A decorator that caches function results for three seconds, with independent expiry for each argument.
- **Recursive traversals:**
  - Preorder: Root → Left → Right
  - Inorder: Left → Root → Right
  - Postorder: Left → Right → Root
- **Iterative traversals:** Implemented the same three traversal orders using explicit stacks, with support for deep trees.

## Running the tests

Ran: python3 test_hw3.py

All 16 supplied tests pass.