# HW 3: Advanced Python (Vasileios Dimitropoulos)

My solutions are in `hw3.py`. `test_hw3.py` is the supplied test file, unchanged.

## How to run the checks

The checks use only the standard library, so any Python 3.12+ works. From this folder:

```bash
python test_hw3.py
```

I used a local virtual environment called `HW3` (Python 3.12.9, created with `uv venv HW3 --python 3.12.9`). It is listed in `.gitignore` and is not part of the submission. With it, the command on Windows (PowerShell) is:

```powershell
.\HW3\Scripts\python.exe test_hw3.py
```

All 16 checks pass and the run ends with `OK`.

## Part 1: Memoization with expiry (`memorize_3s`)

Each time `memorize_3s` decorates a function, it creates a new dictionary. That dictionary is the cache for that function only, so two decorated functions never share results. It maps each argument to a pair: `(result, time the result was computed)`.

**When does a cache entry expire?** The timestamp is read with `time.monotonic()` right *after* the underlying function returns. So if a computation is slow, the 3 seconds start from when it finishes, not from when it started. On a later call with the same argument:

- if fewer than 3 seconds have passed, the cached result is returned;
- at exactly 3 seconds or later, the function runs again and the new result and time replace the old entry.

A few details:

- **Cache hits never change the timestamp**, so calling the function often does not keep an entry alive longer.
- **Each argument expires on its own.** Computing `f(2)` has no effect on when `f(1)` expires.
- **Falsey results are cached correctly.** I check `arg in cache` instead of whether the stored value is truthy, so results like `0`, `None` or `[]` count as cache hits.
- **Failed calls are not cached.** If the function raises an exception, it reaches the caller before the line that writes to the cache, so nothing is stored and the next call tries again.

Expired entries are not removed in the background. An entry is only checked, and replaced if needed, when that same argument is requested again.

## Part 2: Binary tree traversal

### Recursive versions

Each function creates an empty `result` list and a small nested helper `visit(node)`. The helper returns immediately for `None`. Otherwise it appends the node's value and visits both children. The three orders differ only in *when* the value is appended:

| Traversal | Order inside `visit` |
| --------- | -------------------- |
| Pre-order | append, visit left, visit right |
| In-order | visit left, append, visit right |
| Post-order | visit left, visit right, append |

### Iterative versions: how pending work is tracked

In the recursive versions, Python's call stack quietly remembers which nodes still need attention. The iterative versions keep that memory in an explicit stack (a plain Python list, using `append` to push and `pop` to pop). This stack is an ordinary list in memory, not a chain of function calls, so very deep trees do not hit Python's recursion limit.

- **Pre-order.** The stack holds subtrees that still have to be traversed. It starts with the root. Each step pops a node, records its value, then pushes the **right** child and then the **left** child. The stack is last-in, first-out, so the left subtree is fully handled before the right one.

- **In-order.** A `current` pointer walks down to the left, pushing every node it passes. The stack therefore holds ancestors that are waiting for their left side to finish. When `current` runs out, the top of the stack has nothing left to its left, so it is popped and recorded. Then `current` moves to that node's right child and the process repeats.

- **Post-order.** This works like in-order (walk left, pushing nodes), but a node can only be recorded after **both** of its subtrees. So I look at the top of the stack without removing it:
  - If the node has a right child that has not been traversed yet, I move into that right subtree first.
  - Otherwise both sides are done, so I pop the node, record it, and remember it as `last_visited`.

  `last_visited` is what tells the two situations apart. If the node's right child is the node I just recorded, I am coming *back up* from the right subtree, and the parent is ready to be recorded. The check compares node objects with `is`, not their values, so trees with repeated values work correctly.

None of the traversals modify the tree. Each call builds and returns a brand-new list, and an empty tree (`None`) returns `[]`.
