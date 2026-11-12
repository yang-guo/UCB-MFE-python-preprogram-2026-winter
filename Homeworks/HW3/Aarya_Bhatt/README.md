Homework 3: Advanced Python

Aarya Aamish Bhatt

->How to Run Tests
From inside `Homeworks/HW3/Aarya_Bhatt/`, run the standard library unit test suite:

```bash
python3 test_hw3.py
```

---

->Approach & Implementation Details

-->Part 1: Memoization with Expiry (`memorize_3s`)
- Independent caches per function and argument using closures and timestamp tracking.
- Uses `time.monotonic()` for elapsed time tracking.
- Expiry is strictly less than 3 seconds after computation finishes.

-->Part 2: Binary Tree Traversals
- Implemented recursive and iterative traversals (Pre-order, In-order, Post-order) using explicit stacks.
