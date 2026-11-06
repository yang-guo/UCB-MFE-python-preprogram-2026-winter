\# Homework 3 — Shravani Koli

\## Overview
I implemented a 3-second caching decorator and six binary-tree traversal functions using recursion and iteration.



\## Three-Second Cache (`memorize\_3s`)


The `memorize\_3s` decorator saves a function's result so it doesn't have to calculate it again for the same input.

Each cached result is only valid for 3 seconds, starting from when the function finishes running. If the same input is used within that time, the saved result is returned. After 3 seconds, the function runs again and updates the cache. Using a cached result does not restart the timer, and errors are not saved.


\**How the Iterative Traversals Work

Instead of recursion, the iterative functions use stacks to remember which nodes still need to be visited.

- **Preorder (Root -> Left -> Right):** right child is added to the stack before the left child so the left side gets visited first.

- **Inorder (Left -> Root ->  Right):** Stack was used in order to remember parent nodes while moving left, and then return to them before exploring the right side.

- **Postorder (Left -> Right -> Root):** Nodes were processed in the order of Root -> Right -> Left order and then the result was reversed at the end.

\** Running the Tests

From my HW3 folder, please run: `uv run --no-project python test_hw3.py`

All 16 provided tests passed.