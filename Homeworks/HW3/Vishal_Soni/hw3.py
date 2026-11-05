"""HW 3: memoization and binary tree traversals."""

import time


def memorize_3s(func):
    """Cache each argument's successful result for three seconds after completion."""
    cache = {}

    def wrapper(arg):
        # Check the age of this argument's result without resetting its timer.
        if arg in cache:
            result, finished_at = cache[arg]
            if time.monotonic() - finished_at < 3:
                return result

        # Failed calls raise normally and never replace the cached value.
        result = func(arg)
        cache[arg] = (result, time.monotonic())
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

    res = []
    stack = [root]
    while stack:
        cur = stack.pop()
        res.append(cur.val)
        # Push right first so the left child is visited first.
        if cur.right is not None:
            stack.append(cur.right)
        if cur.left is not None:
            stack.append(cur.left)
    return res


def in_order_recursion(root: Node | None) -> list[str]:
    """Return Left, Root, Right order using recursion; None returns []."""
    if root is None:
        return []
    return in_order_recursion(root.left) + [root.val] + in_order_recursion(root.right)


def in_order_iteration(root: Node | None) -> list[str]:
    """Return Left, Root, Right order using a loop and stack(s)."""
    res = []
    stack = []
    cur = root
    while cur is not None or stack:
        # Save the path left, then visit a node and move to its right subtree.
        while cur is not None:
            stack.append(cur)
            cur = cur.left
        cur = stack.pop()
        res.append(cur.val)
        cur = cur.right
    return res


def post_order_recursion(root: Node | None) -> list[str]:
    """Return Left, Right, Root order using recursion; None returns []."""
    if root is None:
        return []
    return post_order_recursion(root.left) + post_order_recursion(root.right) + [root.val]


def post_order_iteration(root: Node | None) -> list[str]:
    """Return Left, Right, Root order using a loop and stack(s)."""
    if root is None:
        return []

    res = []
    stack = [(root, False)]
    while stack:
        cur, visited = stack.pop()
        if visited:
            res.append(cur.val)
        else:
            # Revisit the parent after both children have been processed.
            stack.append((cur, True))
            if cur.right is not None:
                stack.append((cur.right, False))
            if cur.left is not None:
                stack.append((cur.left, False))
    return res
