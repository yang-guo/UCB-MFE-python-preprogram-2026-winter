# HW3: Advanced Python

## How to run the tests
Execute the following command from this directory:
```bash
python test_hw3.py
```

## Part 1: Memoization Approach
To implement `memorize_3s`, I used a dictionary to map each argument to a tuple: `(result, completion_timestamp)`. 
* **Expiry:** When a cached argument is called, I check if `time.monotonic() - completion_timestamp < 3.0`. If true, the cached result is returned. If exactly 3 seconds or more have passed, the cache is considered expired, and the function is recomputed.
* By storing the result in a tuple, falsey values (like `0`, `False`, `None`) are safely cached and retrieved. 
* Exceptions naturally propagate without updating the cache because the assignment to the dictionary happens only after the function returns successfully.

## Part 2: Iterative Traversal Approach
For the iterative tree traversals, I used a Python `list` to simulate a stack to track pending work and avoid recursion limits on deep trees.
* **Pre-order (Root, Left, Right):** We pop a node, record its value, then push its `right` child followed by its `left` child. Pushing right first ensures the left child is popped and processed first.
* **In-order (Left, Root, Right):** We use a nested loop to traverse down to the leftmost leaf, pushing all nodes along the way. Then we pop, record the value, and step to the right child.
* **Post-order (Left, Right, Root):** We pop a node, record its value, and push its `left` child then `right` child. This produces a Root-Right-Left order. Reversing the final result list gives the correct Left-Right-Root post-order traversal.
