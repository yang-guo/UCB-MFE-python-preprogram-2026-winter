"""HW 3 starter. Implement the seven functions, then run test_hw3.py."""

import time
from functools import wraps

def memorize_3s(func):
    """Cache each argument's successful result for three seconds after completion."""
    cache = {}
    
    @wraps(func)
    def wrapper(arg):
        now = time.monotonic()
        
        # return cached result if within 3s
        if arg in cache:
            res, timestamp = cache[arg]
            if now - timestamp < 3.0:
                return res
                
        # otherwise compute, cache, and return
        res = func(arg)
        cache[arg] = (res, time.monotonic())
        return res
        
    return wrapper


class Node:
    """A binary-tree node; missing children are None. Leave this class unchanged."""
    def __init__(self, key: str, left=None, right=None):
        self.val = key
        self.left = left
        self.right = right


def pre_order_recursion(root: Node | None) -> list[str]:
    """Return Root, Left, Right order using recursion; None returns []."""
    if not root:
        return []
    return [root.val] + pre_order_recursion(root.left) + pre_order_recursion(root.right)


def pre_order_iteration(root: Node | None) -> list[str]:
    """Return Root, Left, Right order using a loop and stack(s)."""
    if not root:
        return []
        
    stack, res = [root], []
    while stack:
        node = stack.pop()
        res.append(node.val)
        # push right first so left is processed first
        if node.right:
            stack.append(node.right)
        if node.left:
            stack.append(node.left)
            
    return res


def in_order_recursion(root: Node | None) -> list[str]:
    """Return Left, Root, Right order using recursion; None returns []."""
    if not root:
        return []
    return in_order_recursion(root.left) + [root.val] + in_order_recursion(root.right)


def in_order_iteration(root: Node | None) -> list[str]:
    """Return Left, Root, Right order using a loop and stack(s)."""
    stack, res = [], []
    curr = root
    
    while curr or stack:
        # go left as far as possible
        while curr:
            stack.append(curr)
            curr = curr.left
            
        curr = stack.pop()
        res.append(curr.val)
        
        # move to right child
        curr = curr.right
        
    return res


def post_order_recursion(root: Node | None) -> list[str]:
    """Return Left, Right, Root order using recursion; None returns []."""
    if not root:
        return []
    return post_order_recursion(root.left) + post_order_recursion(root.right) + [root.val]


def post_order_iteration(root: Node | None) -> list[str]:
    """Return Left, Right, Root order using a loop and stack(s)."""
    if not root:
        return []
        
    stack, res = [root], []
    while stack:
        node = stack.pop()
        res.append(node.val)
        # push left then right (reverse of pre-order)
        if node.left:
            stack.append(node.left)
        if node.right:
            stack.append(node.right)
            
    # reverse output to get Left, Right, Root
    return res[::-1]
