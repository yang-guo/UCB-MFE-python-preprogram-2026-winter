# HW3 — Theo Basseras

## Run the tests

From this folder, run:

    python -m unittest -v test_hw3

## Cache

To measure time, I use `time.monotonic()`, as the assignment requires.

I first read the clock and compare it with the time saved when the same
argument was last computed (if it has been computed before). If less than
3 seconds have passed, I return the cached value.

Otherwise, if nothing is cached for this argument or 3 seconds or more have
passed, I compute the value again. Each new computation replaces the saved
value and time, so the next comparison uses the latest computation.

I read the clock again after the computation. This second reading matters
because the function may take a while to run, and the 3 seconds start when
it finishes. Lastly, I save the value and this time, and return the value.

Remark: I use a dictionary because it can check whether an argument is in it,
and retrieve its saved result and time, in O(1) on average.

Each argument has its own saved result and time, and each decorated function
has its own dictionary. Returning a cached value does not reset its time.
If the function raises an exception, nothing is cached, because saving happens
only after the computation succeeds.

## Recursion

In each recursive function, if the current node is `None`, I return an
empty list. This is what stops the recursion when there is no child.

For pre-order, as the name says, the current node's value comes first, then
the left subtree, then the right one. I put the current value in a list,
then concatenate the lists returned by the recursive calls on the left and
right children.

For in-order, the order is left, value, right. I take the list returned by
the recursive call on the left child, then add the current value, then the
list returned by the call on the right child.

For post-order, the order is left, right, value. I concatenate the results
of the recursive calls on the left and right children, then add the current
value at the end.

Each recursive call applies the same order to its own subtree.

## Iterations

### Pre-order

The iterative versions use a list as a stack. For pre-order, I first put the
root on the stack. Then, while the stack is not empty, I pop a node and add
its value to the result.

I push the right child first and the left child last, because the last node
pushed is the first one popped. This way the left subtree is explored before
the right one, which gives exactly pre-order.

### In-order

For in-order, a `current` variable walks through the tree: it goes down to
the leftmost node by following the left children.

I save the nodes on the stack on the way down. Then I pop the top node and
add its value to the result. I move to its right child and repeat the same
logic. The parents stay on the stack, waiting until their left subtrees have
been explored.

The loop continues while there is a current node or there are still nodes
waiting on the stack.

### Post-order

For post-order, I use the same loop-and-stack idea as in-order, but I swap
the left and right children. I also add the current value on the way down,
instead of after popping.

I save the current node on the stack, add its value to the result, and go
down to its right child. After popping a node, I move to its left child and
repeat.

This gives Value, Right, Left. Lastly, I return the reversed list,
`result[::-1]`, which gives Left, Right, Value: exactly post-order.

All iterative functions use loops and stacks without recursion. Each
traversal returns a new list without changing the tree. For an empty tree,
the result is an empty list.
