# HW3

## Approach

The memoization decorator caches results separately for each argument. A cached result expires when it is at least three seconds old. Failed function calls are not cached.

The recursive traversals process the left and right subtrees directly. The iterative traversals use a stack to track nodes still to visit. The in-order traversal also uses a current pointer to go left before processing a node. The iterative post-order traversal reverses its collected order to produce left-right-root order.