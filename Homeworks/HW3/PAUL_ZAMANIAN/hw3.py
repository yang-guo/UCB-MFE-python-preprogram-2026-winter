"""HW 3 starter. Implement the seven functions, then run test_hw3.py."""

import time
from functools import wraps


def memorize_3s(func):
    """Cache each argument's successful result for three seconds after completion.

    The wrapped function takes one positional, hashable argument. Use
    time.monotonic(); cache hits do not reset expiry. See README.md for details.
    """
    cache = {}
    
    @wraps(func)
    def wrapper(arg):
        now = time.monotonic()
        
        if arg in cache:
            result, timestamp = cache[arg]
            
            if now - timestamp < 3:
                return result
        
        # if arg not in cache
        result = func(arg)
        
        # save to cache
        cache[arg] = (result, time.monotonic())
        
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
    if not root:
        return []
    return (
        [root.val] + pre_order_recursion(root.left) + pre_order_recursion(root.right)
    )



def pre_order_iteration(root: Node | None) -> list[str]:
    """Return Root, Left, Right order using a loop and stack(s)."""
    if not root:
        return []

    res = []
    stack = [root]

    while stack:
        current = stack.pop()
        res.append(current.val)
        
        for el in [current.right, current.left]:
            if el:
                stack.append(el)

    return res
        


def in_order_recursion(root: Node | None) -> list[str]:
    """Return Left, Root, Right order using recursion; None returns []."""
    if not root:
            return []
    return (
        in_order_recursion(root.left) + [root.val] + in_order_recursion(root.right)
    )


def in_order_iteration(root: Node | None) -> list[str]:
    """Return Left, Root, Right order using a loop and stack(s)."""
    res = []
    stack = []
    current = root

    while current or stack:
        while current:
            stack.append(current)
            current = current.left

        current = stack.pop()
        res.append(current.val)

        current = current.right

    return res


def post_order_recursion(root: Node | None) -> list[str]:
    """Return Left, Right, Root order using recursion; None returns []."""
    if not root:
        return []
    return (
        post_order_recursion(root.left) + post_order_recursion(root.right) + [root.val]
    )


def post_order_iteration(root: Node | None) -> list[str]:
    """Return Left, Right, Root order using a loop and stack(s)."""
    if not root:
        return []

    res = []
    stack = [root]

    while stack:
        current = stack.pop()
        res.append(current.val)
        
        for el in [current.left, current.right]:
            if el:
                stack.append(el)

    return res[::-1]