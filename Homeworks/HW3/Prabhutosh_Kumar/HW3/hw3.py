"""HW3: expiring memoization and recursive/iterative binary-tree traversals."""

import time
from functools import wraps


def memorize_3s(func):
    """Cache each argument's successful result for less than three seconds.

    Each wrapper owns its cache. Expiry starts after computation completes;
    cache hits do not refresh the timestamp and exceptions are not cached.
    """
    cache = {}

    @wraps(func)
    def wrapped(argument):
        # Check membership, not result truthiness: None and zero are cacheable.
        if argument in cache:
            result, completed_at = cache[argument]
            if time.monotonic() - completed_at < 3:
                return result

        result = func(argument)
        # This is reached only if the call succeeds, including slow calls.
        cache[argument] = (result, time.monotonic())
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
    """Return Root, Left, Right order using an explicit pending-node stack."""
    result = []
    stack = [root] if root is not None else []
    while stack:
        node = stack.pop()
        result.append(node.val)
        # Last in, first out: push right first so left is visited first.
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
    """Return Left, Root, Right order using a stack of pending ancestors."""
    result = []
    stack = []
    current = root
    while current is not None or stack:
        # Defer ancestors until their entire left subtree is visited.
        while current is not None:
            stack.append(current)
            current = current.left
        current = stack.pop()
        result.append(current.val)
        current = current.right
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
    """Return Left, Right, Root order using explicit expansion/visit tasks."""
    result = []
    stack = [(root, False)] if root is not None else []
    while stack:
        node, expanded = stack.pop()
        if expanded:
            result.append(node.val)
        else:
            # Defer the root until both children finish; LIFO runs left first.
            stack.append((node, True))
            if node.right is not None:
                stack.append((node.right, False))
            if node.left is not None:
                stack.append((node.left, False))
    return result
