"""HW3 learning starter. Fill each TODO, then run the supplied tests.

Keep all seven function names. Replace each NotImplementedError with your code.
Read README.md for the work order and test commands.
"""

import time


def memorize_3s(func):
    """Return a function that caches each argument's result for three seconds.

    Input: a function that takes one positional, hashable argument.
    Output: a callable that returns the original function's result.
    Flow: argument -> cache check -> reuse or compute -> result.
    """
    cache = {}
    # Think: Where must it live so each decorated function has its own cache?
    def wrapper(arg):

        ### My approach

        if arg not in cache:
            result = func(arg)
            cache[arg] = (result,time.monotonic())

            return result
        else:
            if time.monotonic() - cache[arg][1] < 3: 
                return cache[arg][0]
            else:
                result = func(arg)
                cache[arg] = (result,time.monotonic())
                
                return result
    return wrapper

    


class Node:
    """A binary-tree node; missing children are None. Leave this class unchanged."""

    def __init__(self, key: str, left=None, right=None):
        self.val = key
        self.left = left
        self.right = right


def pre_order_recursion(root: Node | None) -> list[str]:
    """Return Root, Left, Right order using recursion; None returns []."""


    if root is None :
        return []
    else:
        values = [root.val]

        values.extend(pre_order_recursion(root.left))
        values.extend(pre_order_recursion(root.right))

    return values
    
        
    

def pre_order_iteration(root: Node | None) -> list[str]:
    """Return Root, Left, Right order using a loop and stack(s)."""

    if root is None:
        return []
    
    stack = [root]
    result_list = []

    while stack:
        current = stack.pop()
        result_list.append(current.val)

        if current.right is not None:
            stack.append(current.right)
        if current.left is not None:
            stack.append(current.left)

    return result_list 
        


def in_order_recursion(root: Node | None) -> list[str]:
    """Return Left, Root, Right order using recursion; None returns []."""
   
    if root is None :
        return []
    else:
        values = []

        values.extend(in_order_recursion(root.left))
        values.append(root.val)
        values.extend(in_order_recursion(root.right))

    return values


def in_order_iteration(root: Node | None) -> list[str]:
    """Return Left, Root, Right order using a loop and stack(s)."""
    
    if root is None:
        return []
    
    stack = []
    current = root
    result_list = []

    while current is not None or stack:

        if current is not None:
            stack.append(current)
            current = current.left
        else:
            current = stack.pop()
            result_list.append(current.val)
            current = current.right       

    return result_list 


def post_order_recursion(root: Node | None) -> list[str]:
    """Return Left, Right, Root order using recursion; None returns []."""

    if root is None :
        return []
    else:
        values = []

        values.extend(post_order_recursion(root.left))
        values.extend(post_order_recursion(root.right))
        values.append(root.val)

    return values    



def post_order_iteration(root: Node | None) -> list[str]:
    """Return Left, Right, Root order using a loop and stack(s)."""
    
    if root is None:
        return []
    
    stack = []
    current = root
    result_list = []

    last_visited = None

    while current is not None or stack:

        if current is not None:
            stack.append(current)
            current = current.left
            
        else:
            pending = stack[-1]
            if pending.right is not None and last_visited is not pending.right:
                current = pending.right
            else:    
                stack.pop()
                result_list.append(pending.val)
                last_visited = pending
            
    return result_list
