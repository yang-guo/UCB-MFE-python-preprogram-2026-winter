"""HW3: expiring memoization and binary-tree traversals."""

import time
from functools import wraps


def memorize_3s(func):
    """Cache each argument's successful result for three seconds after completion.

    The wrapped function takes one positional, hashable argument. Use
    time.monotonic(); cache hits do not reset expiry. See README.md for details.
    """
    cache = {}

    @wraps(func)
    def wrapped(arg, /):
        if arg in cache:
            result, finished_at = cache[arg]
            if time.monotonic() - finished_at < 3:
                return result

        result = func(arg)
        cache[arg] = (result, time.monotonic())
        return result

    return wrapped


class Node:
    """A binary-tree node; missing children are None. Leave this class unchanged."""

    def __init__(self, key: str, left=None, right=None):
        self.val = key
        self.left = left
        self.right = right


def pre_order_recursion(root: Node | None) -> list[str]:
    """Return Root, Left, Right order using recursion; None returns []."""
    values = []

    def visit(node):
        if node is None:
            return
        values.append(node.val)
        visit(node.left)
        visit(node.right)

    visit(root)
    return values


def pre_order_iteration(root: Node | None) -> list[str]:
    """Return Root, Left, Right order using a loop and stack(s)."""
    if root is None:
        return []

    values = []
    stack = [root]
    while stack:
        node = stack.pop()
        values.append(node.val)
        # Push right first so that left is popped first.
        if node.right is not None:
            stack.append(node.right)
        if node.left is not None:
            stack.append(node.left)
    return values


def in_order_recursion(root: Node | None) -> list[str]:
    """Return Left, Root, Right order using recursion; None returns []."""
    values = []

    def visit(node):
        if node is None:
            return
        visit(node.left)
        values.append(node.val)
        visit(node.right)

    visit(root)
    return values


def in_order_iteration(root: Node | None) -> list[str]:
    """Return Left, Root, Right order using a loop and stack(s)."""
    values = []
    stack = []
    node = root
    while node is not None or stack:
        while node is not None:
            stack.append(node)
            node = node.left
        node = stack.pop()
        values.append(node.val)
        node = node.right
    return values


def post_order_recursion(root: Node | None) -> list[str]:
    """Return Left, Right, Root order using recursion; None returns []."""
    values = []

    def visit(node):
        if node is None:
            return
        visit(node.left)
        visit(node.right)
        values.append(node.val)

    visit(root)
    return values


def post_order_iteration(root: Node | None) -> list[str]:
    """Return Left, Right, Root order using a loop and stack(s)."""
    if root is None:
        return []

    values = []
    stack = [root]
    while stack:
        node = stack.pop()
        values.append(node.val)
        if node.left is not None:
            stack.append(node.left)
        if node.right is not None:
            stack.append(node.right)
    # Reverse Root, Right, Left to get Left, Right, Root.
    values.reverse()
    return values
