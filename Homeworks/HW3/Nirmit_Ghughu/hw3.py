"""HW 3 starter. Implement the seven functions, then run test_hw3.py."""

import time


def memorize_3s(func):
    """Cache each argument's successful result for three seconds after completion.

    The wrapped function takes one positional, hashable argument. Use
    time.monotonic(); cache hits do not reset expiry. See README.md for details.
    """
    cache = {}  # arg -> (result, time the call finished); one dict per decorated function

    def wrapper(arg):
        if arg in cache:  # membership, not truthiness, so falsey results still count as hits
            result, finished_at = cache[arg]
            if time.monotonic() - finished_at < 3:
                return result
        result = func(arg)  # an exception propagates here and nothing is cached
        cache[arg] = (result, time.monotonic())  # expiry starts once the call has finished
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
    result = []
    stack = [root] if root is not None else []
    while stack:
        node = stack.pop()
        result.append(node.val)
        # Push right first so the left child is popped (visited) first.
        if node.right is not None:
            stack.append(node.right)
        if node.left is not None:
            stack.append(node.left)
    return result


def in_order_recursion(root: Node | None) -> list[str]:
    """Return Left, Root, Right order using recursion; None returns []."""
    if root is None:
        return []
    return in_order_recursion(root.left) + [root.val] + in_order_recursion(root.right)


def in_order_iteration(root: Node | None) -> list[str]:
    """Return Left, Root, Right order using a loop and stack(s)."""
    result = []
    stack = []  # ancestors whose own value and right subtree are still pending
    current = root
    while current is not None or stack:
        while current is not None:  # walk as far left as possible
            stack.append(current)
            current = current.left
        node = stack.pop()  # leftmost unvisited node
        result.append(node.val)
        current = node.right
    return result


def post_order_recursion(root: Node | None) -> list[str]:
    """Return Left, Right, Root order using recursion; None returns []."""
    if root is None:
        return []
    return post_order_recursion(root.left) + post_order_recursion(root.right) + [root.val]


def post_order_iteration(root: Node | None) -> list[str]:
    """Return Left, Right, Root order using a loop and stack(s)."""
    result = []
    stack = [(root, False)] if root is not None else []  # (node, children already expanded?)
    while stack:
        node, expanded = stack.pop()
        if expanded:
            result.append(node.val)  # both subtrees are done, so the node comes last
            continue
        # Pushed in reverse so they pop as: left subtree, right subtree, then the node itself.
        stack.append((node, True))
        if node.right is not None:
            stack.append((node.right, False))
        if node.left is not None:
            stack.append((node.left, False))
    return result
