hw3.py is the implementation of the decorator in Part 1 and six binary tree traversals in Part 2. Please run test_hw3.py to view the test result.


## Below is my approach explanation

## Part 1: Memoization with expiry

The entry with a argument in the cache will expire if the wrapped function with the this argument input has not been called for 3 seconds.


## Part 2: Binary tree traversal

For all three functions, pending worked will be saved into stack by order. After everything we have saved something to result, we will go to the top layer of stack and loop the process. The detail is as below:

1. pre_order_iteration: 

Two lists are created first: result and stack. First of all, root is added into stack. Then, for each round, the top layer of the stack will be saved to the result list and its children will be added to stack (right will be appended first so the left will be on the top of right and will be added to result first).


2. in_order_iteration:

Two lists are created first: result and stack. Then there will be a variable named current_node. We will start from setting root F as our current_node, then it will be added to stack (stands for the route that we have traveled) and update the current_node with its left child until we come to the left end. At the left end, the left child will be None so we will save it to result, which is the top layer of stack. Then we update the current_node with the right sibling and loop this process again (if right child is None then go back to the new top layer of stack).


3. post_order_iteration:

Three lists are created first: result, stack_1, stack_2. Starting from setting the current_node to the root F (initial top layer of stack_1), we will save it to stack_2 and then save its children (if exist, and left is saved before right) to stack_1. Next we will update the current_node with the new top layer of stack_1, which was the rightmost child of the node that was just written into stack_2, and then we will continue the loop. After clearing stack_1, stack_2 will be the final result in reverse order. 
