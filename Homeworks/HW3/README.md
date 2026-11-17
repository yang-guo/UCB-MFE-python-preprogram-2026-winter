# HW3 — Shubham Gupta

Use Python 3.12 or later. From this folder, run the supplied checks with:

```sh
python test_hw3.py
```

The course copy of `test_hw3.py` belongs alongside `hw3.py`. You can also run
the independent checks in this folder with `python test_hw3_extra.py`.

## Memoization

Each decorated function has its own dictionary. The dictionary maps an
argument to its result and the monotonic clock reading taken immediately
after the call succeeds. An entry is reused while its age is less than
three seconds. At three seconds or more the next call recomputes the result.
A hit leaves the timestamp alone, and a call that raises never writes a
result. Since the lookup tests for the key, falsey results are cached too.

## Tree traversals

The recursive versions append node values before, between, or after
visiting the two children, depending on the requested order. Each call
creates its own result list.

- Pre-order pushes the right child before the left child so the left is
  processed first.
- In-order uses the stack to remember ancestors while walking down a left
  branch. After visiting the top ancestor, it moves into that node's right
  subtree.
- Post-order pushes the left child before the right child to collect
  root-right-left order, then reverses the list to get left-right-root.

All six traversals visit each node once and leave the tree intact.
