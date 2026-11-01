"""HW 3 starter. Implement the seven functions, then run test_hw3.py."""

import time


def memorize_3s(func):
    """Cache each argument's successful result for three seconds after completion.

    The wrapped function takes one positional, hashable argument. Use
    time.monotonic(); cache hits do not reset expiry. See README.md for details.
    """

    cache = {} 
    def wrapper(x):
        current_time = time.monotonic()
        if x in cache:
            result, timestamp = cache[x]
            if current_time - timestamp < 3:
                return result
        result = func(x)
        current_time = time.monotonic()
        cache[x] = (result, current_time)
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

    
    
    return [] if root is None else [root.val] + pre_order_recursion(root.left) + pre_order_recursion(root.right)



def pre_order_iteration(root: Node | None) -> list[str]:
    """Return Root, Left, Right order using a loop and stack(s)."""
    stack=[root] if root is not None else []
    result=[]

    while stack!=[]:
        node=stack.pop()
        result.append(node.val)
        if node.right is not None:
            stack.append(node.right)
        if node.left is not None:
            stack.append(node.left)
    return result



def in_order_recursion(root: Node | None) -> list[str]:
    """Return Left, Root, Right order using recursion; None returns []."""
    
    return [] if root is None else in_order_recursion(root.left) + [root.val] + in_order_recursion(root.right)


def in_order_iteration(root: Node | None) -> list[str]:
    """Return Left, Root, Right order using a loop and stack(s)."""
    stack = []
    result = []
    current = root

    while current is not None or stack:
        while current is not None:
            stack.append(current)
            current = current.left
        current = stack.pop()
        result.append(current.val)
        current = current.right


    return result



def post_order_recursion(root: Node | None) -> list[str]:
    """Return Left, Right, Root order using recursion; None returns []."""
    
    return [] if root is None else post_order_recursion(root.left) + post_order_recursion(root.right) + [root.val]


def post_order_iteration(root: Node | None) -> list[str]:
    """Return Left, Right, Root order using a loop and stack(s)."""
    stack = []
    result = []
    current = root

    
    while current is not None or stack != []:
        while current is not None:
            stack.append(current)
            result.append(current.val)
            current = current.right
        current = stack.pop()
        current = current.left     
        
    return result[::-1]  

