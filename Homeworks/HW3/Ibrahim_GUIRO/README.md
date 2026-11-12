# HMW3 : Ibrahim GUIRO

From `Homeworks/HW3/Ibrahim_GUIRO` run 

```bash 
python test_hw3.py
```

The `memorize_3s` decorator uses a dictionary to store each argument's result and completion timestamp. Each decorated function has its own cache. 
Entries expire three seconds after the function successfully returns.

A cached result is reused only while its age is less than three seconds.
Cache hits do not reset expiry. Elapsed time is measured with
`time.monotonic()`. Failed calls are not cached.

The recursive traversals combine the current node's value with recursive results from its children in the required order. 

The iterative versions use loops and explicit stacks to track pending work.
- Pre-order pushes the right child before the left, so the left is processed first.
- In-order stores ancestors while descending left, then visits each node before exploring its right subtree.
- Post-order builds Root, Right, Left order, then reverses the result.
