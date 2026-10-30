# HW 3: Advanced Python

**Due: Thursday, November 5, 2026, at 11:59 p.m. Pacific Time (PST, UTC-8).**

This is 14 calendar days after the final Advanced Python lecture on October 22, following the [course schedule and late policy](../../README.md#homework). The course's three free late days and maximum of three days late apply.

This assignment covers decorators, caching, recursion, and stacks from [Lecture 5](../../Lectures/Lecture%205/README.md) and [Lecture 6](../../Lectures/Lecture%206/README.md). It adapts the January cohort's two-part HW 3 for this course.

## How to approach the assignment

**Write the implementation yourself.** These are core problem-solving skills used in coding interviews. Use AI as a tutor: ask about unfamiliar concepts, explain your understanding, and ask it to quiz you. Do not give the assignment to AI or ask it to write your solution. After your implementation works, you may use AI to review it and suggest further practice.

## Part 1: Memoization with expiry

Memoization stores the result of a function call so repeated inputs can reuse it. Implement the `memorize_3s` decorator in [hw3.py](hw3.py). Keep this function name for compatibility with the assignment.

Requirements:

- The wrapped function accepts **one positional, hashable argument**. Each argument has its own cached result and timestamp; different decorated functions have separate caches.
- A cached result is valid for **less than 3 seconds** after the underlying function successfully returns. At exactly 3 seconds or later, call the function again and cache the new result.
- Cache hits do not extend the expiry time. Return the underlying function's result, including falsey values such as `0`, `False`, `None`, or an empty list.
- Use `time.monotonic()` to measure elapsed time. Keep the starter's `import time` and look up `time.monotonic()` when reading the clock so the supplied tests can control it.
- If the underlying function raises an exception, let it propagate and do not cache that failed call.

For example, if `f(1)` finishes at time 100, a call at 102.9 uses its cache; a call at 103 recomputes it. Caching `f(2)` in between does not change `f(1)`'s expiry. The clock values are elapsed-time readings, not dates.

You do not need to support keyword/multiple arguments, unhashable inputs, async functions, thread safety, cache-size limits, or background cleanup. It is enough to check an entry's expiry when that argument is requested again. Use your own cache rather than a library cache decorator.

## Part 2: Binary tree traversal

The supplied `Node` stores a value in `val` and children in `left` and `right`. A missing child, or an empty tree, is `None`.

```text
        F
       / \
      B   G
     / \   \
    A   D   I
       / \ /
      C  E H
```

| Traversal | Visit order | Result for this tree |
| --------- | ----------- | -------------------- |
| Pre-order | Root, Left, Right | F, B, A, D, C, E, G, I, H |
| In-order | Left, Root, Right | A, B, C, D, E, F, G, H, I |
| Post-order | Left, Right, Root | A, C, E, D, B, H, I, G, F |

Implement each traversal twice using the six supplied function names:

1. **Recursively**, with recursive function calls.
2. **Iteratively**, with a loop and explicit stack(s), without calling recursive helpers or the recursive versions.

Return a new list of node values; return `[]` for an empty tree. Do not change the tree or retain results between calls. Inputs are finite binary trees with string values, including duplicate values. They are **not necessarily binary search trees**, so sorting values is not a traversal. Iterative versions should also handle trees deeper than Python's normal recursion limit; do not raise that limit. Recursive versions are tested only on small trees.

## Starter files and checks

- [hw3.py](hw3.py): `Node` plus seven functions for you to implement.
- [test_hw3.py](test_hw3.py): automated checks using Python's standard-library `unittest`.

Use Python 3.12 or later. No extra packages, database, API keys, or network access are needed to run the checks with an installed Python interpreter.

Create a branch `hw_3_<FIRST_NAME>_<LAST_NAME>` from the current course `main`. Create `Homeworks/HW3/<FIRST_NAME>_<LAST_NAME>/` and copy **both** Python files into it. Leave the shared files unchanged.

From your personal folder, run:

```bash
python test_hw3.py
```

On systems where Python 3 is named `python3`, use `python3 test_hw3.py`; on Windows you may use `py -3 test_hw3.py`. If using the course's uv setup, `uv run --no-project python test_hw3.py` runs these standard-library checks without syncing the full course dependencies.

The untouched starter intentionally reports `NotImplementedError` failures. Once all seven functions are implemented correctly, the suite ends with `OK`. Running `hw3.py` alone only defines the functions; it does not run the checks.

The tests control the clock instead of sleeping. They check cache hits, per-argument expiry, the exact three-second boundary, separate functions, falsey results, failed calls, and slow computations. Tree checks include the example above, empty/single-node trees, left/right chains, unsorted and repeated values, repeat calls, unchanged inputs, and deep iterative traversal. Passing these checks is necessary; graders also review that each traversal uses the requested recursive or iterative approach.

## Submission

```text
Homeworks/HW3/<FIRST_NAME>_<LAST_NAME>/
    hw3.py          # Your implementation
    test_hw3.py     # Supplied checks, unchanged
    README.md       # Run command and a short explanation of your approach
```

In your README, explain when cache entries expire and how your iterative traversals track pending work. You may add extra tests in another file.

Run the checks on your personal copy, commit only your submission folder, and open a pull request into the course repository's `main`. Title it `HW3: <First Last>` and assign `kevinramlal` and `raghuramkowdeed`. Do not commit virtual environments or change other students' work.

Grading is pass/no-pass based on correct behavior, the required recursive/iterative approaches, and clear, readable code you can explain.

## References

- [Python: monotonic elapsed-time clock](https://docs.python.org/3.12/library/time.html#time.monotonic)
- [Python: using lists as stacks](https://docs.python.org/3.12/tutorial/datastructures.html#using-lists-as-stacks)
- [Python: running unittest tests](https://docs.python.org/3.12/library/unittest.html#command-line-interface)
