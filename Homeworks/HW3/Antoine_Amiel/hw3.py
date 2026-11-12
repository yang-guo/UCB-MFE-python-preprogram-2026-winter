"""HW 3: memoization with expiry and binary tree traversals."""

import time
from functools import wraps

CACHE_DURATION_S = 3.0


def memorize_3s(func):
    """Cache each argument's successful result for three seconds after completion.

    The wrapped function takes one positional, hashable argument. Use
    time.monotonic(); cache hits do not reset expiry. See README.md for details.
    """
    cache = {}  # arg -> (result, time the call finished)

    @wraps(func)
    def wrapper(arg):
        if arg in cache:
            result, finished_at = cache[arg]
            if time.monotonic() - finished_at < CACHE_DURATION_S:
                return result
        # Missing or expired entry: recompute. If func raises, the exception
        # propagates and the line storing the result is never reached.
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


# ---------------------------------------------------------------------------
# Recursive traversals
# ---------------------------------------------------------------------------


def pre_order_recursion(root: Node | None) -> list[str]:
    """Return Root, Left, Right order using recursion; None returns []."""
    if root is None:
        return []
    return [root.val] + pre_order_recursion(root.left) + pre_order_recursion(root.right)


def in_order_recursion(root: Node | None) -> list[str]:
    """Return Left, Root, Right order using recursion; None returns []."""
    if root is None:
        return []
    return in_order_recursion(root.left) + [root.val] + in_order_recursion(root.right)


def post_order_recursion(root: Node | None) -> list[str]:
    """Return Left, Right, Root order using recursion; None returns []."""
    if root is None:
        return []
    return post_order_recursion(root.left) + post_order_recursion(root.right) + [root.val]


# ---------------------------------------------------------------------------
# Iterative traversals (explicit stacks, no recursion)
# ---------------------------------------------------------------------------


def pre_order_iteration(root: Node | None) -> list[str]:
    """Return Root, Left, Right order using a loop and stack(s)."""
    result = []
    stack = [root] if root is not None else []
    while stack:
        node = stack.pop()
        result.append(node.val)
        # Push right first so that left is popped (visited) first.
        if node.right is not None:
            stack.append(node.right)
        if node.left is not None:
            stack.append(node.left)
    return result


def in_order_iteration(root: Node | None) -> list[str]:
    """Return Left, Root, Right order using a loop and stack(s)."""
    result = []
    stack = []  # nodes whose left subtree is being explored
    node = root
    while stack or node is not None:
        # Walk as far left as possible, remembering each node on the way down.
        while node is not None:
            stack.append(node)
            node = node.left
        # Left subtree done: visit the node, then explore its right subtree.
        node = stack.pop()
        result.append(node.val)
        node = node.right
    return result


def post_order_iteration(root: Node | None) -> list[str]:
    """Return Left, Right, Root order using a loop and stack(s)."""
    result = []
    stack = [root] if root is not None else []
    # Build Root, Right, Left order, which is exactly post-order reversed.
    while stack:
        node = stack.pop()
        result.append(node.val)
        if node.left is not None:
            stack.append(node.left)
        if node.right is not None:
            stack.append(node.right)
    result.reverse()
    return result
