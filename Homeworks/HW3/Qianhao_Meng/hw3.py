"""HW 3 starter. Implement the seven functions, then run test_hw3.py."""

import time

def memorize_3s(func):
    """Cache each argument's successful result for three seconds after completion.

    The wrapped function takes one positional, hashable argument. Use
    time.monotonic(); cache hits do not reset expiry. See README.md for details.
    """
    cache = {}
    def wrapper(x):
        # reuse the result if it is still valid
        if x in cache:
            result, finished_at = cache[x]
            if time.monotonic() - finished_at < 3:
                return result
        # recompute, then cache the result and complete time if successful
        result = func(x)
        cache[x] = (result, time.monotonic())
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
    def _pre_order(node):
        if node is not None:
            result.append(node.val)
            _pre_order(node.left)
            _pre_order(node.right)
    _pre_order(root)
    return result

def pre_order_iteration(root: Node | None) -> list[str]:
    """Return Root, Left, Right order using a loop and stack(s)."""
    result = []
    if root is not None:
        stack = [root]
    else: 
        return result
    while stack:
        node = stack.pop()
        result.append(node.val)
        # Push right first so the left child is popped first
        if node.right is not None:
            stack.append(node.right)
        if node.left is not None:
            stack.append(node.left)
    return result

def in_order_recursion(root: Node | None) -> list[str]:
    """Return Left, Root, Right order using recursion; None returns []."""
    result = []

    def _in_order(node):
        if node is not None:
            _in_order(node.left)
            result.append(node.val)
            _in_order(node.right)

    _in_order(root)
    return result


def in_order_iteration(root: Node | None) -> list[str]:
    """Return Left, Root, Right order using a loop and stack(s)."""
    result = []
    stack = []
    node = root 
    while node is not None or stack:
        while node is not None:
            stack.append(node)
            node = node.left
        node = stack.pop()
        result.append(node.val)
        node = node.right
    return result


def post_order_recursion(root: Node | None) -> list[str]:
    """Return Left, Right, Root order using recursion; None returns []."""
    result = []

    def _post_order(node):
        if node is not None:
            _post_order(node.left)
            _post_order(node.right)
            result.append(node.val)

    _post_order(root)
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
        if node.left is not None:
            stack1.append(node.left)
        if node.right is not None:
            stack1.append(node.right)
    while stack2:
        node = stack2.pop()
        result.append(node.val)
    return result
