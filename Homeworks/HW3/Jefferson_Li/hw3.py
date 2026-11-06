"""HW3: expiring memoization and binary-tree traversals."""

import time


def memorize_3s(func):
    """Cache each argument's successful result for three seconds after completion.

    The wrapped function takes one positional, hashable argument. Use
    time.monotonic(); cache hits do not reset expiry. See README.md for details.
    """
    cache = {}

    def wrapper(x):
        if x in cache:
            result, finished_at = cache[x]
            if time.monotonic() - finished_at < 3:
                return result
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
    values = []

    def helper(root):
        if not root:
            return

        values.append(root.val)
        helper(root.left)
        helper(root.right)

    helper(root)
    return values


def pre_order_iteration(root: Node | None) -> list[str]:
    """Return Root, Left, Right order using a loop and stack(s)."""
    values = []
    stack = [root] if root is not None else []

    while stack:
        node = stack.pop()
        values.append(node.val)

        if node.right is not None:
            stack.append(node.right)
        if node.left is not None:
            stack.append(node.left)

    return values


def in_order_recursion(root: Node | None) -> list[str]:
    """Return Left, Root, Right order using recursion; None returns []."""
    values = []

    def helper(node):
        if node is None:
            return

        helper(node.left)
        values.append(node.val)
        helper(node.right)

    helper(root)
    return values


def in_order_iteration(root: Node | None) -> list[str]:
    """Return Left, Root, Right order using a loop and stack(s)."""
    values = []
    stack = []
    cur = root

    while cur is not None or stack:
        while cur is not None:
            stack.append(cur)
            cur = cur.left
        cur = stack.pop()
        values.append(cur.val)
        cur = cur.right

    return values


def post_order_recursion(root: Node | None) -> list[str]:
    """Return Left, Right, Root order using recursion; None returns []."""
    values = []

    def helper(node):
        if node is None:
            return

        helper(node.left)
        helper(node.right)
        values.append(node.val)

    helper(root)
    return values


def post_order_iteration(root: Node | None) -> list[str]:
    """Return Left, Right, Root order using a loop and stack(s)."""
    values = []
    stack = [(root, False)] if root is not None else []

    while stack:
        node, ready = stack.pop()
        if ready:
            values.append(node.val)
        else:
            stack.append((node, True))
            if node.right is not None:
                stack.append((node.right, False))
            if node.left is not None:
                stack.append((node.left, False))

    return values
