# HW 3: Advanced Python (Shashank Mishra)

## How to run

From this folder:

```bash
python test_hw3.py
```

On Windows without `python` on the PATH, use `py -3 test_hw3.py`. All 16 checks pass (`OK`).

## Part 1: `memorize_3s`

`memorize_3s` is a decorator. When it is applied to a function, it creates one dictionary, `cache`, for that function only, so different decorated functions never share results. The dictionary maps each argument to a tuple `(result, finished_at)`, where `finished_at` is the `time.monotonic()` reading taken right after the function returned.

**When entries expire.** On each call the wrapper checks whether the argument is in `cache`. If it is, it computes `time.monotonic() - finished_at`. If that is strictly less than 3 seconds, the stored result is returned. At exactly 3 seconds or later, the entry counts as expired, so the function is called again and the entry is overwritten with the new result and a new timestamp.

- The timestamp is taken **after** `func(arg)` returns, so a slow function still gets its full three seconds.
- A cache hit returns the stored result without touching the timestamp, so hits never extend the expiry.
- The check uses `arg in cache`, not the truthiness of the result, so falsey results (`0`, `False`, `None`, `""`, `[]`) are cached too.
- If `func` raises, the line that stores the result never runs, so failures are not cached and the exception propagates.
- Expired entries are only replaced when the same argument is requested again (no background cleanup).

## Part 2: Binary tree traversals

Each traversal has a recursive and an iterative version. All return a new list and do not change the tree. An empty tree (`None`) returns `[]`.

**Recursive versions.** Each one calls itself on `root.left` and `root.right` and combines the results in the right order: pre-order is root, left, right; in-order is left, root, right; post-order is left, right, root. The base case is `root is None`, which returns `[]`.

**Iterative versions.** These use a list as a stack, so there is no recursion and they work on trees deeper than Python's recursion limit. The stack holds the nodes whose work is still pending.

- **Pre-order:** push the root. Repeatedly pop a node, record its value, then push its right child and then its left child. Because a stack is last-in-first-out, the left child is popped first, giving root, left, right.
- **In-order:** keep a `current` pointer. Walk left as far as possible, pushing every node passed on the way; these are the ancestors whose own value and right subtree are still pending. Then pop a node, record its value, and move to its right child. Repeat until `current` is `None` and the stack is empty.
- **Post-order:** run the pre-order idea but push the left child first and then the right child, which produces root, right, left. Reversing that list gives left, right, root.
