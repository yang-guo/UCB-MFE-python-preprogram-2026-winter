# HW3: Advanced Python

Author: Ziang Gao

## How to run the tests

Open a terminal in this folder. Then run:

```
python test_hw3.py
```

The tests are in `test_hw3.py`. I did not change this file. All tests must show `OK`.

## Part 1: `memorize_3s`

### How it works

`memorize_3s` is a decorator. It saves the result of a function for 3 seconds.

- The decorator makes one dictionary, `cache`, when it wraps a function.
- The inner function `wrapper` is a closure. It can read `cache` on every call.
- Each key in `cache` is an argument. Each value is a tuple: `(result, expiry_time)`.
- On each call, `wrapper` does these steps:
  1. It reads the clock with `time.monotonic()`.
  2. If the argument is in `cache` and the clock is less than `expiry_time`, it returns the saved result. The wrapped function does not run.
  3. Otherwise, it runs the wrapped function.
  4. It reads the clock again. It saves the result with `expiry_time = finish time + 3`.
  5. It returns the result.

### When cache entries expire

- The 3 seconds start when the wrapped function **finishes**. They do not start when the function starts. A slow function does not use up its own 3 seconds.
- An entry is valid while `clock < finish time + 3`.
- At exactly 3 seconds, the entry is expired. The function runs again.
- A cache hit does **not** make the entry last longer. The expiry time stays the same.
- Each argument has its own expiry time.

### Other design choices

- I use `arg in cache` to check for a saved result. I do not check if the result is true or false. So results such as `0`, `None`, and `""` are also saved.
- If the wrapped function raises an error, the error goes to the caller. The line that saves the result comes after `func(arg)`, so a failed call is never saved.
- I call `time.monotonic()` through the `time` module. The tests can then replace the clock. `time.monotonic()` is also not changed when the computer clock changes.

## Part 2: Binary tree traversal

### Recursive versions

Each recursive function returns a new list. It does these steps:

1. If the node is `None`, it returns an empty list.
2. It calls itself for `root.left` and for `root.right`.
3. It joins the two lists and the value `root.val` with `extend` and `append`.

The three orders differ only in the position of one line, `result.append(root.val)`:

| Order | Steps |
| ----- | ----- |
| Pre-order | root, left, right |
| In-order | left, root, right |
| Post-order | left, right, root |

Python remembers the work that is still waiting on its own call stack. A very deep tree can pass the recursion limit. So the tests use the recursive versions only on small trees.

### Iterative versions: how pending work is tracked

The iterative versions do not call themselves. Each one uses a loop and an explicit stack. The stack is a Python list. `append` pushes a node and `pop` removes the last node. The stack holds the nodes that still wait to be visited. Because it is a normal list, deep trees do not reach the recursion limit.

- **Pre-order.** The stack starts with the root. In each loop step, I pop a node and add its value to the result. Then I push the right child first and the left child second. The left child is on top, so it is popped first.
- **In-order.** I go left as far as I can. I push every node on the way, because each of these nodes must wait until its left side is done. When there is no left child, I pop a node, add its value, and move to its right child. Then I repeat.
- **Post-order.** I use two stacks. The first stack makes the order root, right, left. Each node that I pop from it moves to the second stack. I push the left child first and the right child second, so the right child is popped first. Then I pop the second stack, which reverses the order to left, right, root. I add each value to the result.

### Other notes

- Each function builds a new list. It does not change the tree. It does not keep results between calls.
- An empty tree (`None`) returns `[]`.
- The traversals do not sort values, so they also work for trees that are not binary search trees.
