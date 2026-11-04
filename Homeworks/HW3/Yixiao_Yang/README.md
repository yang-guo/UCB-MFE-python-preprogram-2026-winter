# HW3 — Yixiao Yang

## Run the tests

Use Python 3.12 or later. From this folder, run:

```bash
python3 test_hw3.py
```

The supplied `test_hw3.py` has not been edited. No extra packages are required.

## Part 1: Expiring memoization

`memorize_3s` uses a closure containing a dictionary. Each input argument maps to a pair of its cached return value and the time the function **finished** computing that value. The decorator uses `time.monotonic()` for elapsed time.

On a call, an entry is reused only when `now - completed_at < 3`. At exactly three seconds, or afterward, the underlying function runs again and the cache entry is replaced. Cache hits do not change the timestamp. Each application of the decorator makes its own dictionary, so functions do not share cached results. The argument's presence is checked rather than the truthiness of its result, so values such as `0` and `None` are cached. When the function raises an exception, no new entry is saved.

## Part 2: Binary tree traversal

All traversal functions return a fresh list and leave the nodes untouched.

| Traversal | Recursive approach | Iterative stack approach |
| --- | --- | --- |
| Pre-order | Append root, recurse left, then right | Visit the popped node, push right then left so left is visited first |
| In-order | Recurse left, append root, recurse right | Push the whole left path; pop and visit a node, then move to its right subtree |
| Post-order | Recurse left, then right, then append root | Produce Root–Right–Left using a stack and reverse the list to obtain Left–Right–Root |

The iterative functions use loops and explicit stacks instead of the Python call stack. They therefore also work on trees that exceed Python's usual recursion depth.
