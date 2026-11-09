# HW3 README

## Run command: When inside the valerie_macisaac folder, run python3 test_hw3.py


## Explanation of when the cache entries expire: 

In my code, the cache entries expire after 3.0 seconds. If the function is called within 3 seconds of when the cache has stored it for that given argument, then it uses the stored cache values. Otherwise, it recomputes the function for that given argument. 

## Explanation of how iterative traversals track pending work: 

The recursive methods remember which nodes have been explored. 

For the pre-order iterative method, there is a stack that holds the unvisited trees, with the initial stack holding root. As the pre-order function iterates, the top node in the stack is popped and appended to the results, following by appending the right and then left (so the left is on top) and continuing to the next iteration. 

For the in-order iterative method, the current node explores left, appending each left parent root into the stack until it reaches the final left node. It then explores the right node and continues. 

For the post-order iterative method, the function explores a similar method as the pre-order, solving for root, right, left and then reversing the output list. 
