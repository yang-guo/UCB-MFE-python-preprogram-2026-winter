# HW3 — Mauricio Gutierrez

This submission implements a cache decorator and six binary tree traversals.
The requirements are in the [official assignment](../README.md).

```text
Homeworks/HW3/Mauricio_Gutierrez/
    hw3.py       Implementation
    test_hw3.py  Supplied tests
    README.md    Run instructions and explanation
```

## Run the tests

Use Python 3.12 or later. The code uses the Python standard library.

1. Open a terminal in this submission folder.
2. Run the supplied tests.

```shell
python test_hw3.py
```

With the existing course environment, run this command from the repository root.

```powershell
.\.venv\Scripts\python.exe .\Homeworks\HW3\Mauricio_Gutierrez\test_hw3.py
```

Validation on November 7, 2026 used Python 3.12.9. All 16 tests passed with `OK`.
The personal test file matches the supplied test file.

## Part 1 — Cache with expiry

`memorize_3s` takes a function and returns a wrapper. The wrapper accepts one
positional, hashable argument and returns the function's result.

```text
Argument -> Check cache
                |-- Fresh entry -> Return stored result
                |-- Missing or expired entry -> Compute -> Store -> Return
```

I use a dictionary inside `memorize_3s`, outside `wrapper`. Each decorated function
has its own dictionary. The wrapper keeps access to it between calls.
Each argument is a key. Its entry is `(result, completion_time)`.

I call `time.monotonic()` after the function returns successfully. An entry is
valid while its age is less than three seconds. At exactly three seconds, or
later, the wrapper computes and stores a new result. Reusing an entry preserves
its completion time. Each argument has a separate expiry time.

For example, a result completed at time 100 is valid at 102.9 and expires at 103.
The key membership check lets the cache store results such as `0`, `False`,
`None`, and an empty list. If the function raises an exception, the exception
propagates before the cache assignment. That failed call stores no new entry.

## Part 2 — Binary tree traversals

Each function receives a `Node` or `None`. It returns a new list of string values.
An empty tree returns `[]`. Each call uses its own result list.

```text
Tree root -> Recursive calls or explicit stack -> Ordered list of values
```

For this example, the three orders produce different lists.

```text
    M
   / \
  Z   A
```

| Traversal | Order | Result |
| --- | --- | --- |
| Pre-order | Root, Left, Right | `["M", "Z", "A"]` |
| In-order | Left, Root, Right | `["Z", "M", "A"]` |
| Post-order | Left, Right, Root | `["Z", "A", "M"]` |

### Recursive functions

Each call handles one subtree. The `root is None` check stops recursion at a
missing child. Each call creates a separate local list. The parent uses
`extend()` to combine lists returned by its children. A single node value enters
the list as one item. Its position gives the required traversal order.

### Iterative functions

I use a Python list as a stack. `append()` adds a node reference. `pop()` removes
and returns the last reference added. Removing that reference leaves the tree's
child connections intact. The result list stores node values.

- **Pre-order.** The stack holds nodes waiting to be processed. I pop a node and
  record its value. I add its right child first, then its left child. This makes
  the left child the next node removed.
- **In-order.** I save nodes while moving left. When `current` reaches `None`, I
  pop the pending node, record its value, and move to its right child. The stack
  remembers ancestors while their left subtrees are processed.
- **Post-order.** I save nodes while moving left. Then I inspect the top node
  with `stack[-1]`. If its right subtree is unfinished, I move right and keep the
  parent in the stack. Otherwise, I pop the parent, record its value, and update
  `last_visited`. Comparing node identity with `is` distinguishes nodes with
  equal values. Because post-order records a subtree's root last, `last_visited`
  tells me when the pending node's right subtree is complete.

The iterative functions use loops and stacks. Their pending work is stored in
lists, so they can process trees deeper than Python's normal recursion limit.
All traversals preserve the input tree and follow its connections.

