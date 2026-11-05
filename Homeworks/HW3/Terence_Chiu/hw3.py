"""HW 3 starter. Implement the seven functions, then run test_hw3.py."""

import time


def memorize_3s(func):
    """Cache each argument's successful result for three seconds after completion.

    The wrapped function takes one positional, hashable argument. Use
    time.monotonic(); cache hits do not reset expiry. See README.md for details.
    """
    cache = {}
    timeout = 3

    def wrapper(arg):
        if arg in cache and time.monotonic() - cache[arg][1] < timeout:
            return cache[arg][0]

        result = func(arg)
        t = time.monotonic()
        cache[arg] = [result, t]
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
    if root is None:
        return []
    
    return [root.val] + pre_order_recursion(root.left) + pre_order_recursion(root.right)


def pre_order_iteration(root: Node | None) -> list[str]:
    """Return Root, Left, Right order using a loop and stack(s)."""
    if root is None:
        return []
    
    stack = []
    result = []
    current = root
    while current is not None:
        result.append(current.val)
        if current.right:
            stack.append(current.right)
        if current.left:
            current = current.left
        elif stack:
            current = stack.pop()
        else:
            current = None
    return result

def in_order_recursion(root: Node | None) -> list[str]:
    """Return Left, Root, Right order using recursion; None returns []."""
    if root is None:
        return []
    
    return in_order_recursion(root.left) + [root.val] + in_order_recursion(root.right)

def in_order_iteration(root: Node | None) -> list[str]:
    """Return Left, Root, Right order using a loop and stack(s)."""
    if root is None:
        return []
    
    stack = []
    result = []
    current = root
    while current or stack:
        while current:
            stack.append(current)
            current = current.left
        current = stack.pop()
        result.append(current.val)
        current = current.right
    return result


def post_order_recursion(root: Node | None) -> list[str]:
    """Return Left, Right, Root order using recursion; None returns []."""
    if root is None:
        return []
    return post_order_recursion(root.left) + post_order_recursion(root.right) + [root.val]

def post_order_iteration(root: Node | None) -> list[str]:
    """Return Left, Right, Root order using a loop and stack(s)."""
    if root is None:
        return []

    stack = []
    result = []
    current = root
    last_visited = None
    
    while current or stack:
        # go left
        while current:
            stack.append(current)
            current = current.left
        # check right
        peek = stack[-1]
        if peek.right and peek.right is not last_visited:
            current = peek.right
        else:
            # visit own
            result.append(stack.pop().val)
            last_visited = peek
            current = None
    return result