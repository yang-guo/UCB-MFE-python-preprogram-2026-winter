HW3: Oscar Sallusti

How to run:

Folder:
Homeworks/HW3/Oscar_Sallusti/
python3 test_hw3.py

memorize_3s:

The memorize_3s function takes whatever function is included as an input, and returns a new function that wraps around the inputed function. When the wrapper function is called, it caches the function output as well as a timestamp. If the function raises an error, nothing is cached. The timestamp is taken after the function returns. The next time the function is called, if the cached output was computed less than 3 seconds ago, it returns this cached result. Calling a cached result doesn’t change the timestamp of when the result was calculated. If the stored answer is 3 or more seconds old, or there is no stored answer, it calls on the inputed function to calculate a new answer with a new timestamp. 
 
Each decorated function has its own cache, because the cache dict is created inside memorize_3s. Falsey results like 0 and None are still cached.

Recursive functions:

The recursive functions pre_order_recursion, in_order_recursion, and post_order_recursion all work to recursively map a node tree. These determine where root.val takes place, and where in the sequence the node is recorded.

With pre_order_recursion, a node is recorded before any connected nodes below it. This causes parent nodes to always be recorded before children.

Meanwhile, in_order_recursion records each nodes between the left and right subtrees. 

Finally, post_order_recursion records a node after its subtrees are recorded. This is a bottom up approach that always records child nodes before parent nodes.

Iterative functions:

The three iterative functions more explicitly state the same operation as the three recursive functions. The main difference is that as they’re iterative, the list [stack] holds the nodes in the tree that still need to be processed, while the list [result] collects values in visit order.

For pre_order_iteration, it starts at the top node, records it right away and then pushes first the right child and then the left child to the stack. This ensures that the next node the function goes to is this left child node, which is then recorded and removed from the stack, while it’s right then left nodes are added. This process continues until every node has been mapped and the stack list is empty.

For in_order_iteration, the function doesn’t record nodes straight away. Starting at the top, it first adds the parent and child node to the stack, then travels to the left most node node. This repeats until it reaches a node with nothing more to the left. It then records this node, and removes it from the stack. It then travels to that node’s right child and repeats the process.

For post_order_iteration, The first stack holds nodes still to visit. The second holds visited nodes, collected in reverse of the order you need. The sequence of the function is: Do a pre-order traversal with the children swapped, so it visits Root, Right, Left. Then: Reverse the result. That gives Left, Right, Root, which is post-order.

One benefit of iterative functions over recursive ones, is that they can work on very deep trees. If a tree has thousands and thousands of chained nodes, a recursive function would call itself within an already running recursive function too many times, using too much memory and leading to an error. With the iterative method where each output is stored as a list, memory issues are much less of a concern as the function is only called once.
