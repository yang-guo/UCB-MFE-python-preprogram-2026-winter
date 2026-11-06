"""HW 3 starter. Implement the seven functions, then run test_hw3.py."""

import time


def memorize_3s(func):
    """Cache each argument's successful result for three seconds after completion.

    The wrapped function takes one positional, hashable argument. Use
    time.monotonic(); cache hits do not reset expiry. See README.md for details.
    """
    cache_name = f"cache_{func}" # To make sure each function has a different cache
   
    cache = {}    
    def wrapper(n):
        start = time.monotonic()
        if n in cache:
            if (time.monotonic() - cache[n][1]) < 3:
                return(cache[n][0])
            else:
                result = func(n)
                cache[n] = [result, time.monotonic()]
                return cache[n][0]
        elif n not in cache:
            result = func(n)
            cache[n] = [result, time.monotonic()]
            return cache[n][0]
    #print(cache)
    #print(cache_name) # Different functions indeed had different names
    return wrapper




class Node:
    """A binary-tree node; missing children are None. Leave this class unchanged."""    
    def __init__(self, key: str, left=None, right=None):
        self.val = key
        self.left = left
        self.right = right


def pre_order_recursion(root: Node | None) -> list[str]:
    """Return Root, Left, Right order using recursion; None returns []."""
    # raise NotImplementedError("Implement pre_order_recursion")
    if root is None:
        return([])
    if root.left is None:
        if root.right is None: 
            return([root.val])
        else:
            output = [root.val] + (pre_order_recursion(root.right))
            return(output)
    else:
        if root.right is None: 
            return([root.val]  + pre_order_recursion(root.left))
        else:
            output = [root.val] + pre_order_recursion(root.left) + (pre_order_recursion(root.right))
            return(output)


def pre_order_iteration(root: Node | None) -> list[str]:
    """Return Root, Left, Right order using a loop and stack(s)."""
    if root is None:
        return([])
    full_output = []
    right_nodes = []
    current_node = root
    while type(current_node) != type(None):
            full_output = full_output + [current_node.val]
            if type(current_node.right) != type(None): # Save right nodes in order LIFO
                right_nodes = right_nodes + [current_node.right]
    
            if type(current_node.left) != type(None): # Iterate down left nodes if they exist
                    current_node = current_node.left
            
            elif len(right_nodes) != 0:
                    current_node = right_nodes.pop() # If no more left, use most recent right
            else: # If out of right nodes (& Out of left as the first if has already failed - just break the loop which setting current to null will do)
                current_node = current_node.left
    return(full_output)


def in_order_recursion(root: Node | None) -> list[str]:
    """Return Left, Root, Right order using recursion; None returns []."""
    if root is None:
        return([])
    if root.left is None:
        if root.right is None: 
            return([root.val])
        else:
            output = [root.val] + (in_order_recursion(root.right))
            return(output)
    else:
        if root.right is None: 
            return(in_order_recursion(root.left) + [root.val])
        else:
            output = in_order_recursion(root.left) + [root.val]  + (in_order_recursion(root.right))
            return(output)


def in_order_iteration(root: Node | None) -> list[str]:
    """Return Left, Root, Right order using a loop and stack(s)."""
    if root is None:
        return([])
    full_output = []
    right_nodes = []
    root_values = []
    current_node = root
    while type(current_node) != type(None):
            if type(current_node.right) != type(None): # Save right nodes in order LIFO
                right_nodes = right_nodes + [current_node.right]
    
            if type(current_node.left) != type(None): # Iterate down left nodes if they exist, saving the roots as we go
                    root_values = root_values + [current_node.val]
                    current_node = current_node.left
            else:
                full_output = full_output + [current_node.val]
                if len(root_values) != 0: # Imagine we're at the end of the branch. We just added the left, now the root, and then we move to the most recent right which will repeat and traverse down the left again and so on.
                    full_output = full_output + [root_values.pop()]    
                if len(right_nodes) != 0: # If the left values don't exist we're at the far left of the tree. Save, and then save the corresponding root
                    current_node = right_nodes.pop() # If no more left, use most recent right
                else:
                    if len(root_values) != 0: # Imagine we're at the end of the branch. We just added the left, now the root, and then we move to the most recent right which will repeat and traverse down the left again and so on.
                        full_output = full_output + root_values
                    current_node = current_node.left
    return(full_output)





def post_order_recursion(root: Node | None) -> list[str]:
    """Return Left, Right, Root order using recursion; None returns []."""
    if root is None:
        return([])
    elif root.left is None:
        if root.right is None: 
            return([root.val])
        else:
            output = post_order_recursion(root.right) + [root.val]
            return(output)
    else:
        if root.right is None: 
            return(post_order_recursion(root.left) + [root.val])
        else:
            output = post_order_recursion(root.left) + (post_order_recursion(root.right)) + [root.val] 
            return(output)

def post_order_iteration(root: Node | None) -> list[str]:
    """Return Left, Right, Root order using a loop and stack(s)."""
    raise NotImplementedError("Implement post_order_iteration")
