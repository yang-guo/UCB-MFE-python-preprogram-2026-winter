# HW3

> **PLEASE NOTE README WAS CONVERTED TO MARKDOWN FROM TEXT BY CLAUDE**

## Run Command

```bash
python test_hw3.py
```

## Recursion

My approach to the recursion was quite simple:

1. Check for NoneTypes.
2. Then return the given order of left root right, root left right, or left right root.
3. Whenever combining left root and right in any permutation, call the recursion on the left and right branches of the tree in case they have deeper levels. Note the root by definition is a value and does not have a deeper value.

## Iteration

The iteration was extremely challenging.

1. For root left right, I traversed down the left branch saving the root as I went until I got to the end.
2. Then I also saved the right branches in a stack and iterated over the right branches until the program completed.

For the second two iterative functions I figured out the trick eventually. I needed to make one ordered LIFO Stack of the branches of the trees sorted in each step by the specified program order. I overthought it by a mile.

## Caching

My approach for the caching was in hindsight simple but to get there was challenging.

- I printed the cache and a regex formed `cache_name` variable to ensure that different functions were getting different caches.
- Fortunately this happened by default.
- I decided to store the time with the result in a list. Originally I had considered having two caches with the same keys but one with time values one with the result but decided against it to minimize moving parts.
- The `<3` seconds almost tripped me up as I'd originally put `<= 3`
