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
        t = time.monotonic()
        if arg in cache:
            val, ts = cache[arg]
            if t - ts < 3.0:
                return val

        res = func(arg)
        cache[arg] = (res, time.monotonic())
        return res

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
    return [root.val] + pre_order_recursion(root.left) + pre_order_recursion(root.right)


def pre_order_iteration(root: Node | None) -> list[str]:
    """Return Root, Left, Right order using a loop and stack(s)."""
    if not root:
        return []
    res, stack = [], [root]
    while stack:
        n = stack.pop()
        res.append(n.val)
        if n.right:
            stack.append(n.right)
        if n.left:
            stack.append(n.left)
    return res


def in_order_recursion(root: Node | None) -> list[str]:
    """Return Left, Root, Right order using recursion; None returns []."""
    if not root:
        return []
    return in_order_recursion(root.left) + [root.val] + in_order_recursion(root.right)


def in_order_iteration(root: Node | None) -> list[str]:
    """Return Left, Root, Right order using a loop and stack(s)."""
    res, stack = [], []
    curr = root
    while curr or stack:
        while curr:
            stack.append(curr)
            curr = curr.left
        curr = stack.pop()
        res.append(curr.val)
        curr = curr.right
    return res


def post_order_recursion(root: Node | None) -> list[str]:
    """Return Left, Right, Root order using recursion; None returns []."""
    if not root:
        return []
    return post_order_recursion(root.left) + post_order_recursion(root.right) + [root.val]


def post_order_iteration(root: Node | None) -> list[str]:
    """Return Left, Right, Root order using a loop and stack(s)."""
    if not root:
        return []
    res, stack = [], []
    curr = root
    last = None
    while curr or stack:
        if curr:
            stack.append(curr)
            curr = curr.left
        else:
            peek = stack[-1]
            if peek.right and last is not peek.right:
                curr = peek.right
            else:
                res.append(peek.val)
                last = stack.pop()
    return res