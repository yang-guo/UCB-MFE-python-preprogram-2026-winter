HW3 - Pierre-Francois Pinelli

To run the tests from this folder:

`python3 test_hw3.py`

1. Memorize 3s

For this exercise, I implemented a decorator that stores the result of a function for 3 seconds, so we don't need to compute it again if the function is called with the same argument.

I used a dictionary inside the decorator to store the results. Each function has its own cache, where I save the argument, the result and the time when the function finished, using `time.monotonic()`.

When the function is called, I first check if the argument is already in the cache. If it is and less than 3 seconds have passed, I return the stored result. Otherwise, I call the function again and update the cache.

The 3 seconds start when the function finishes, and not when it starts. Also, using a cached result does not reset the timer.

Finally, I used `x in cache` instead of checking the result directly, so values like `0`, `False` or `None` are also cached. If the function raises an error, nothing is stored.

2. Binary Tree Traversals

For this part, I implemented the three types of binary tree traversal, using both recursive and iterative methods.

* Pre-order: Root -> Left -> Right
* In-order: Left -> Root -> Right
* Post-order: Left -> Right -> Root

Recursive method

The recursive method is quite simple. If the node is `None`, I return an empty list. Otherwise, I call the function on the left and right children and combine the results with the value of the current node.

The only difference between the three functions is the position of the root value in the list.

Iterative method

For the iterative method, I chose to use a stack with the same structure for all three traversals.

The stack contains tuples `(node, ready)`. The boolean `ready` tells us if the node should be added to the result or if we still need to process its children.

If `ready` is `False`, I push the node and its children back into the stack. If it is `True`, I add its value to the result. If the node is `None`, I simply skip it.

Since a stack works with the LIFO principle (Last In, First Out), I need to push the elements in the reverse order of the traversal:

* Pre-order: Right -> Left -> Root
* In-order: Right -> Root -> Left
* Post-order: Root -> Right -> Left

This way, I can use almost the same code for the three functions, only changing the order in which the elements are pushed into the stack.

The main advantage of this method is that it also works for very deep trees, where the recursive version could reach Python's recursion limit.
