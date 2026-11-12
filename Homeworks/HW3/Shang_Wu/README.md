# HW 3: Shang Wu

## How to run

From this folder:

```bash
python test_hw3.py
```

Use `python3 test_hw3.py` if Python 3 is named `python3` on your system. The suite ends with `OK` when all checks pass.

## Part 1: `memorize_3s`

The decorator returns a `wrapper` function that closes over two dictionaries, `cached_result` and `cached_epoch`. Both are keyed by the argument, so each argument has its own result and timestamp. They are created once per decoration, so each decorated function has its own cache.

**When cache entries expire:** an entry is valid for less than 3 seconds after the underlying function returns.

- The timestamp is read with `time.monotonic()` after `func(arg)` returns, so a slow computation's running time does not use up any of the 3 seconds.
- At exactly 3 seconds or later, the function is called again and its new result and timestamp replace the old ones.
- Cache hits do not update the timestamp, so repeated calls do not extend the expiry.
- Expiry is only checked when an argument is requested again. Expired entries stay in the dictionaries until then.

**Falsey results and failed calls:** every result that `func(arg)` returns is cached, including `0`, `False`, `None`, and `[]`, so the code does not check the result's value. If `func(arg)` raises an exception, the exception propagates before the store lines run, so the failed call is not cached.

## Part 2: Iterative traversals

Each iterative traversal uses a list called `buffer` as a stack to track pending work, using `append` to push and `pop` to remove the most recent item. Because they use a loop rather than recursive calls, they can handle trees deeper than Python's recursion limit.

- **Pre-order:** the stack holds nodes that have not been visited yet. After visiting a node, I push its right child and then its left child. The left child is on top of the stack, so it is visited first.
- **In-order:** the stack holds nodes whose left subtree is still being processed. The inner `while` loop walks down the left side, pushing each node. I then pop a node, record its value, and move to its right child. The outer loop continues while there is a current node or the stack is not empty.
- **Post-order:** I visit nodes in Root, Right, Left order by pushing the left child and then the right child, so the right child is popped first. Reversing that list at the end gives Left, Right, Root.

## Final

I believe the test is not completely right. More specifically, for Part 1, if the return value is [], we should not cache. However, if I really follow that way, the test will fail. Per the 'wrong' suggestion from Claude Code, I
remove the following check to make the test passed.

The following is the "right" one in fact. However it will fail the unit test code.
```
        curr_result = func(arg)
	if curr_result:
	        cached_epoch[arg] = time.monotonic()
        	cached_result[arg] = curr_result
```
