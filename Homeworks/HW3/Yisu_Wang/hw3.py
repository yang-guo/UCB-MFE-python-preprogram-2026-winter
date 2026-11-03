"""HW 3 starter. Implement the seven functions, then run test_hw3.py."""

import time


def memorize_3s(func):
    """Cache each argument's successful result for three seconds after completion.

    The wrapped function takes one positional, hashable argument. Use
    time.monotonic(); cache hits do not reset expiry. See README.md for details.
    """
    cache = {}
    def wrapper(arg):
        if arg in cache:
            value, finished_at = cache[arg]
            if time.monotonic() - finished_at < 3:
                return value
        value = func(arg)
        cache[arg] = (value, time.monotonic())
        return value
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
    def visit(node):
        if node is None:
            return
        result.append(node.val)
        visit(node.left)
        visit(node.right)
    visit(root)
    return result


def pre_order_iteration(root: Node | None) -> list[str]:
    """Return Root, Left, Right order using a loop and stack(s)."""
    result, stack = [], [root] if root is not None else []
    while stack:
        node = stack.pop()
        result.append(node.val)
        if node.right is not None:
            stack.append(node.right)
        if node.left is not None:
            stack.append(node.left)
    return result


def in_order_recursion(root: Node | None) -> list[str]:
    """Return Left, Root, Right order using recursion; None returns []."""
    result = []
    def visit(node):
        if node is None:
            return
        visit(node.left)
        result.append(node.val)
        visit(node.right)
    visit(root)
    return result


def in_order_iteration(root: Node | None) -> list[str]:
    """Return Left, Root, Right order using a loop and stack(s)."""
    result, stack, node = [], [], root
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
    def visit(node):
        if node is None:
            return
        visit(node.left)
        visit(node.right)
        result.append(node.val)
    visit(root)
    return result


def post_order_iteration(root: Node | None) -> list[str]:
    """Return Left, Right, Root order using a loop and stack(s)."""
    result, stack = [], [root] if root is not None else []
    while stack:
        node = stack.pop()
        result.append(node.val)
        if node.left is not None:
            stack.append(node.left)
        if node.right is not None:
            stack.append(node.right)
    return result[::-1]
