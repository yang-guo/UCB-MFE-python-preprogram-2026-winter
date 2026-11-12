# HW 3 – Terence Chiu

## Running the tests

From this folder:

```bash
python test_hw3.py
```

## Memoization (`memorize_3s`)

Cache lives in the decorator and stores the function output results as well as the timestamp the function outputs.

Cache entries expire 3 seconds after the function has completed. On a cache hit, the time doesn't get overwritten and the original 3 seconds is still in place, since that is the real age of the function output. At exactly 3.0 seconds it recalculates.

## Iterative traversals

What each stack holds:

- Pre-order: the right children that are being postponed
- In-order: the parents that are getting postponed because the left child is visited first
- Post-order: the parents that are getting postponed because the left child is visited first. It then checks if the right child is `last_visited` or absent, in which it pops, or it goes down the right child.

Iterative approach handles deep trees because it relies on a Python list object instead of Python's call stack which is used by the recursive approach and is capped.