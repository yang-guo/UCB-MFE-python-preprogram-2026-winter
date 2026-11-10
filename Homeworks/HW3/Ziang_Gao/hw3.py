"""HW 3 starter. Implement the seven functions, then run test_hw3.py."""

import time


def memorize_3s(func):
    """Cache each argument's successful result for three seconds after completion.

    The wrapped function takes one positional, hashable argument. Use
    time.monotonic(); cache hits do not reset expiry. See README.md for details.
    """
    
    cache = {}
    
    def wrapper(arg):
        current_time = time.monotonic()
        if arg in cache:
            result, expiry = cache[arg]
            if current_time < expiry:
                return result
        result = func(arg)
        cache[arg] = (result, time.monotonic() + 3)
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
    result = []
    if root is None:
        return result
    result.append(root.val)
    result.extend(pre_order_recursion(root.left))
    result.extend(pre_order_recursion(root.right))
    return result


def pre_order_iteration(root: Node | None) -> list[str]:
    """Return Root, Left, Right order using a loop and stack(s)."""
    result = []
    if root is None:
        return result
    stack = [root]
    while stack:
        node = stack.pop()
        result.append(node.val)
        if node.right:
            stack.append(node.right)
        if node.left:
            stack.append(node.left)
    return result
    


def in_order_recursion(root: Node | None) -> list[str]:
    """Return Left, Root, Right order using recursion; None returns []."""
    result = []
    if root is None:
        return result
    result.extend(in_order_recursion(root.left))
    result.append(root.val)
    result.extend(in_order_recursion(root.right))
    return result


def in_order_iteration(root: Node | None) -> list[str]:
    """Return Left, Root, Right order using a loop and stack(s)."""
    result = []
    stack = []
    current = root
    while stack or current:
        while current:
            stack.append(current)
            current = current.left
        current = stack.pop()
        result.append(current.val)
        current = current.right
    return result


def post_order_recursion(root: Node | None) -> list[str]:
    """Return Left, Right, Root order using recursion; None returns []."""
    result = []
    if root is None:
        return result
    result.extend(post_order_recursion(root.left))
    result.extend(post_order_recursion(root.right))
    result.append(root.val)
    return result

def post_order_iteration(root: Node | None) -> list[str]:
    """Return Left, Right, Root order using a loop and stack(s)."""
    result = []
    if root is None:
        return result
    stack1 = [root]
    stack2 = []
    while stack1:
        node = stack1.pop()
        stack2.append(node)
        if node.left:
            stack1.append(node.left)
        if node.right:
            stack1.append(node.right)
    while stack2:
        node = stack2.pop()
        result.append(node.val)
    return result