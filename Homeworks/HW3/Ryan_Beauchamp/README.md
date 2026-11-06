# HW3: Ryan Beauchamp

## How to run

From this folder:

```bash
python test_hw3.py
```

(With uv: `uv run --no-project python test_hw3.py`.)


## Part 1: `memorize_3s`

The decorator keeps a dictionary inside the decorator itself, so each decorated function gets its own cache. The key is the argument, and the value is the function's result plus the time it was saved.

On each call, the wrapper checks whether the argument is already in the dictionary and whether its saved time is less than 3 seconds old. If both are true, it returns the saved result. Otherwise it calls the original function, saves the new result with the current time, and returns it.

**When entries expire:** the clock is `time.monotonic()`, read right after the function returns, not when the argument was passed in. An entry is valid while it is strictly less than 3 seconds old, so at exactly 3.0 seconds it is recomputed. Expiry is only checked when that argument is requested again, and nothing is cleaned up in the background. Cache hits don't change the saved time, so repeated hits can't keep an entry alive. If the function raises an error, the error passes through before anything is stored, so failed calls aren't cached. Results like `0`, `None` or `[]` are cached correctly because the wrapper checks whether the argument is in the dictionary, not whether the stored result is truthy.


## Part 2: tree traversals

**Recursive versions:** each one uses a small nested helper that appends to a fresh list created inside the function. The helper first checks whether the node is `None` and stops if so. The three orders only differ in their order for which the append happens relative to the two recursive calls.

**Iterative versions:** each one uses initially uses an empty Python list as a stack of nodes that still need to be constructed.

- **Pre-order:** start with the root on the stack. Pop a node, record its value, and push its right child and then its left child. The last one pushed comes out first, so the left side is visited first.
- **In-order:** keep a "current" node. Walk left, pushing each node on the way, because those nodes still have to be recorded after their left side is done. When there's nothing more on the left, pop a node, record it, and move to its right child. The loop ends when the current node is `None` and the stack is empty.
- **Post-order:** this is essentially the pre-order with the pushes swapped/rearranged, to where it visits Root, Right, Left. Reversing that list gives Left, Right, Root, which is what the HW3 README described and asked for.

