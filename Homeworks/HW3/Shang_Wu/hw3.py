"""HW 3 starter. Implement the seven functions, then run test_hw3.py."""

import time

def memorize_3s(func):
    """Cache each argument's successful result for three seconds after completion.

    The wrapped function takes one positional, hashable argument. Use
    time.monotonic(); cache hits do not reset expiry. See README.md for details.
    """
    cached_epoch = {}
    cached_result = {}
    def wrapper(arg):
        curr_epoch = time.monotonic()
        if arg in cached_epoch and curr_epoch - cached_epoch[arg] < 3:
            return cached_result[arg]
        curr_result = func(arg)
        cached_epoch[arg] = time.monotonic()
        cached_result[arg] = curr_result
        return curr_result
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
    ans = []
    buffer = []
    buffer.append(root)
    while buffer:
        node = buffer.pop()
        ans.append(node.val)
        if node.right is not None:
            buffer.append(node.right)
        if node.left is not None:
            buffer.append(node.left)
    return ans

def in_order_recursion(root: Node | None) -> list[str]:
    """Return Left, Root, Right order using recursion; None returns []."""
    if root is None:
        return []
    return in_order_recursion(root.left) + [root.val] + in_order_recursion(root.right)

def in_order_iteration(root: Node | None) -> list[str]:
    """Return Left, Root, Right order using a loop and stack(s)."""
    ans = []
    buffer = []
    current = root

    while current is not None or buffer:
        while current is not None:
            buffer.append(current)
            current = current.left

        current = buffer.pop()
        ans.append(current.val)

        current = current.right

    return ans

def post_order_recursion(root: Node | None) -> list[str]:
    """Return Left, Right, Root order using recursion; None returns []."""
    if root is None:
        return []
    return post_order_recursion(root.left) + post_order_recursion(root.right) + [root.val]

def post_order_iteration(root: Node | None) -> list[str]:
    """Return Left, Right, Root order using a loop and stack(s)."""
    if root is None:
        return []
    ans = []
    buffer = []
    buffer.append(root)
    while buffer:
        node = buffer.pop()
        ans.append(node.val)
        if node.left is not None:
            buffer.append(node.left)
        if node.right is not None:
            buffer.append(node.right)
    return ans[::-1]
