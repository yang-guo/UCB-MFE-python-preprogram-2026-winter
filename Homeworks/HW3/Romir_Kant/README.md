# HW 3: Romir Kant

## How to run

From this folder, run:

```bash
python test_hw3.py
```

On Windows, use `py -3 test_hw3.py`. All 16 tests pass.

## Part 1: `memorize_3s`

I make a `cache` dictionary inside `memorize_3s`, so every function I decorate gets its own cache. The wrapper can still use it after `memorize_3s` returns. For each argument, I save the result and the time the function finished.

- When the function is called, I check if the argument is in the cache and less than 3 seconds have passed. If so, I return the saved result.
- At 3 seconds or more, the entry is expired. I call the function again and save the new result and time.
- Using a saved result doesn't reset its time, so it still expires 3 seconds after it was first computed.
- I check `arg in cache` instead of checking the saved value, so results like `0`, `None`, or `[]` still count as cached.
- I read the clock after the function finishes, so a slow function still gets the full 3 seconds.
- I only save the result after the function returns. If it raises an error, nothing gets saved and the error goes to the caller.

## Part 2: Tree traversals

**Recursive:** if the node is `None`, return `[]`. Otherwise, combine the node's value with the lists from the left and right children in the right order.

**Iterative:** I use a list as a stack to keep track of what still needs to be visited. Because they don't use recursion, they work on very deep trees too.

- **Pre-order:** pop a node and add its value. Then push the right child and then the left child, so the left child comes off the stack first. If a popped node is `None`, skip it.
- **In-order:** keep going left and pushing each node until there are no more. Then pop a node, add its value, and go to its right child.
- **Post-order:** like pre-order, but push left and then right. This visits nodes as Root, Right, Left. I add each value to the front of the list, which flips it into Left, Right, Root.
