# HW3 — Yisu Wang

Run the checks from the repository root:

```bash
./.venv/bin/python Homeworks/HW3/Yisu_Wang/test_hw3.py
```


`memorize_3s` caches successful results separately for each argument. An entry expires three seconds after the function finishes; cache hits do not extend that time.

The recursive traversals visit nodes in the required order. The iterative versions use stacks: pre-order pushes right before left, in-order tracks the path to the leftmost node, and post-order reverses a root-right-left traversal.
