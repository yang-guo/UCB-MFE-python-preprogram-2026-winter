# HW3 — Jefferson Li

Requires Python 3.12 or later; no extra packages are needed.

Run from this folder:

```bash
python test_hw3.py
```

## Approach

- Each decorated function has its own cache. Results expire three seconds after completion, measured with `time.monotonic()`. Cache hits do not reset expiry; failed calls are not cached.
- Recursive traversals append to a fresh result list in the required order.
- Iterative traversals use stacks: pre-order pushes right before left, in-order saves ancestors while descending left, and post-order marks nodes to record after their children.
