"""HW 3 starter. Implement the seven functions, then run test_hw3.py."""

import time


def memorize_3s(func):
    """Cache each argument's successful result for three seconds after completion.

    The wrapped function takes one positional, hashable argument. Use
    time.monotonic(); cache hits do not reset expiry. See README.md for details.
    """
    # remember calculations for 3 seconds
    # use a dictionary that for each input remembers the output and the timestamp it occurs
    # if timestamp is more than 3 seconds ago, remove it from the cache and recalculate

    cache = {} #dictionary to hold eahc input, output, and timestamp

    def wrapper(x):
        
    # first check if the input is already cached
        if x in cache:
            output, timestamp = cache[x]

            # return the cached output only if less than 3 seconds have passed
            if time.monotonic() - timestamp < 3:
                return output
            else:
                # if more than 3 seconds, delete the expired entry
                del cache[x]

        if x not in cache:
            # calc the output and store it in dictionary with a new timestamp, input is the key
            output = func(x)
            cache[x] = (output, time.monotonic())

        return output

    return wrapper
    
    #raise NotImplementedError("Implement memorize_3s")


class Node:
    """A binary-tree node; missing children are None. Leave this class unchanged."""

    def __init__(self, key: str, left=None, right=None):
        self.val = key
        self.left = left
        self.right = right


def pre_order_recursion(root: Node | None) -> list[str]:
    """Return Root, Left, Right order using recursion; None returns []."""

    # returns emtpy list for empty trees
    if root is None:
        return []
    else:
        results = [root.val]  #list

        if root.left is not None:
            results += pre_order_recursion(root.left)
        if root.right is not None:
            results += pre_order_recursion(root.right)

    return results
        
    
    #raise NotImplementedError("Implement pre_order_recursion")


def pre_order_iteration(root: Node | None) -> list[str]:
    """Return Root, Left, Right order using a loop and stack(s)."""
    #raise NotImplementedError("Implement pre_order_iteration")

    stack = [] # stack is places we visit, with next node to visit at the top of the stack
    results = [] # results is what is returned

    if root is None:
        return []

    stack.append(root)
    
    while len(stack)>0:

        current_node = stack.pop() # current node is last node in our stack (most recently added) since we start at root
        results.append(current_node.val) # add its value to  results
        
        if current_node.right is not None: # have to add right first so then left is on the top and first to be visited
            stack.append(current_node.right) 
            
        if current_node.left is not None:
            stack.append(current_node.left) 

    return results

        
    

    


def in_order_recursion(root: Node | None) -> list[str]:
    """Return Left, Root, Right order using recursion; None returns []."""
    
    # returns emtpy list for empty trees
    if root is None:
        return []
        
    else:
        results = []
        if root.left is not None:
            results += in_order_recursion(root.left)
        results += [root.val]
        
        if root.right is not None:
            results += in_order_recursion(root.right)

    return results
    #raise NotImplementedError("Implement in_order_recursion")


def in_order_iteration(root: Node | None) -> list[str]:
    """Return Left, Root, Right order using a loop and stack(s)."""
    #raise NotImplementedError("Implement in_order_iteration")

    # need to go down entire left side of tree, record bottom node, then return root, then right node

    
    stack = [] # stack is places we visit, with next node to visit at the top of the stack
    visited = [] # list of already visited nodes
    results = [] # results is what is returned

    if root is None:
        return []

    stack.append(root)
    
    while len(stack)>0:

        current_node = stack.pop() # current node is last node in our stack (most recently added)

        if current_node in visited: # this means that the node was a parent of another node we've already dealt with
            results.append(current_node.val)
        else:
            visited.append(current_node)

            if current_node.right is not None: 
                stack.append(current_node.right) # we will visit the right branch, but first must add the root

            stack.append(current_node) # visit current node before the right side, so current_node must be most recently added to stack

             # visit left node before the current node, so left node must be most recently added to stack
            if current_node.left is not None:
                stack.append(current_node.left)
            
    return results


def post_order_recursion(root: Node | None) -> list[str]:
    """Return Left, Right, Root order using recursion; None returns []."""

    # returns emtpy list for empty trees
    if root is None:
        return []
        
    else:
        results = []
        if root.left is not None:
            results += post_order_recursion(root.left)
        
        if root.right is not None:
            results += post_order_recursion(root.right)
        results += [root.val]

    return results
    #raise NotImplementedError("Implement post_order_recursion")


def post_order_iteration(root: Node | None) -> list[str]:
    """Return Left, Right, Root order using a loop and stack(s)."""
    #raise NotImplementedError("Implement post_order_iteration")

    # need to go down entire left side of tree, record bottom node, then return right, then node

    
    stack = [] # stack is places we visit, with next node to visit at the top of the stack
    visited = [] # list of already visited nodes
    results = [] # results is what is returned

    if root is None:
        return []

    stack.append(root)
    
    while len(stack)>0:

        current_node = stack.pop() # current node is last node in our stack (most recently added)

        if current_node in visited: # this means that the node was a parent of another node we've already dealt with
            results.append(current_node.val)
        else:
            visited.append(current_node)

            
            stack.append(current_node) # visit current node after left and right
            
            if current_node.right is not None: 
                stack.append(current_node.right) # we will visit the right branch, but first must visit the left

            

             # visit left node first, so left node must be most recently added to stack
            if current_node.left is not None:
                stack.append(current_node.left)
            
    return results