"""HW 3 starter. Implement the seven functions, then run test_hw3.py."""

import time


def memorize_3s(func):
    """Cache each argument's successful result for three seconds after completion.

    The wrapped function takes one positional, hashable argument. Use
    time.monotonic(); cache hits do not reset expiry. See README.md for details.
    """
    cache = {}

    def wrapped(arg):
        if arg in cache:
            saved_result, finished_at = cache[arg]
            if time.monotonic() - finished_at < 3:
                return saved_result

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
    if root is None:
        return []

    left_values = pre_order_recursion(root.left)
    right_values = pre_order_recursion(root.right)
    return [root.val] + left_values + right_values


def pre_order_iteration(root: Node | None) -> list[str]:
    """Return Root, Left, Right order using a loop and stack(s)."""
    values = []
    stack = [root]

    while stack:
        node = stack.pop()
        if node is None:
            continue

        values.append(node.val)
        stack.append(node.right)
        stack.append(node.left)

    return values


def in_order_recursion(root: Node | None) -> list[str]:
    """Return Left, Root, Right order using recursion; None returns []."""
    if root is None:
        return []

    left_values = in_order_recursion(root.left)
    right_values = in_order_recursion(root.right)
    return left_values + [root.val] + right_values


def in_order_iteration(root: Node | None) -> list[str]:
    """Return Left, Root, Right order using a loop and stack(s)."""
    values = []
    stack = []
    current = root

    while current is not None or stack:
        while current is not None:
            stack.append(current)
            current = current.left

        current = stack.pop()
        values.append(current.val)
        current = current.right

    return values


def post_order_recursion(root: Node | None) -> list[str]:
    """Return Left, Right, Root order using recursion; None returns []."""
    if root is None:
        return []

    left_values = post_order_recursion(root.left)
    right_values = post_order_recursion(root.right)
    return left_values + right_values + [root.val]


def post_order_iteration(root: Node | None) -> list[str]:
    """Return Left, Right, Root order using a loop and stack(s)."""
    values = []
    stack = [root]

    while stack:
        node = stack.pop()
        if node is None:
            continue

        values.append(node.val)
        stack.append(node.left)
        stack.append(node.right)

    values.reverse()
    return values


