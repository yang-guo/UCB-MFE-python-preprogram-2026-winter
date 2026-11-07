# HW3 – Anouksha Bansal

## Approach

### Part 1: Memoization with expiry

The `memorize_3s` decorator stores each argument's computed result along with the time at which the function successfully finishes computing it. When the same argument is passed again, the cached result is returned if fewer than three seconds have elapsed. Once three seconds have elapsed, the function is recomputed and the cache entry is replaced. Cache hits do not reset the expiration timer. If the original function raises an exception, its result is not cached.

### Part 2: Binary tree traversal

The recursive traversal functions call themselves on the left and right children in the appropriate order.

The iterative traversal functions use stacks to keep track of nodes that still need to be processed. For preorder traversal, the right child is pushed before the left child so that the left child is processed first. For inorder traversal, nodes are pushed while moving down the left side of the tree, then processed before moving to the right child. For postorder traversal, two stacks are used to reverse the processing order so that each node is visited after its children.

All traversal functions return a new list of node values, and an empty tree returns an empty list.