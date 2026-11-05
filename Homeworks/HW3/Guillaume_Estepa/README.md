# HW 3: Advanced Python — Guillaume Estepa

## 1. How to Run

From inside `Homeworks/HW3/Guillaume_Estepa/`, run the test suite using `uv`:

```bash
uv run --no-project python test_hw3.py
```


## 2. Part 1: When Cache Entries Expire

Each function decorated with `memorize_3s` has its own closure dictionary that maps each argument `arg` to a `(result, finished_time)` tuple. When a call is made, we check `arg in cache`—so falsey return values like `0`, `False`, `None`, or `[]` are still recognized as valid cached entries—and compare the current `time.monotonic()` reading against `finished_time`. An entry is fresh only when the elapsed time is strictly less than 3 seconds (`now - finished_time < 3`), meaning an entry at exactly 3.0 seconds or older is expired and triggers a recomputation. Cache hits return the stored result immediately without updating `finished_time`, so hits never extend the expiry window. On a cache miss or expired entry, the completion timestamp is recorded with `time.monotonic()` *after* `func(arg)` returns, and if `func(arg)` raises an exception, it propagates before touching the cache so failed calls are never stored.

## 3. Part 2: How the Iterative Traversals Track Pending Work

* **Pre-order (`Root, Left, Right`):** The stack holds the roots of subtrees waiting to be visited. In each step we pop a node, record its value, and push its right child before its left child so that the LIFO stack visits the left subtree first.
* **In-order (`Left, Root, Right`):** The stack holds ancestor nodes whose left subtrees are currently being explored. Using a `current` pointer, we walk left as far as possible while pushing each node onto the stack; once `current` hits `None`, we pop the top ancestor, visit it, and advance `current` to its right child.
* **Post-order (`Left, Right, Root`):** We run a modified pre-order loop that pushes the left child before the right child, visiting nodes in `Root, Right, Left` order. Reversing that collected sequence at the end yields the exact `Left, Right, Root` post-order traversal.
