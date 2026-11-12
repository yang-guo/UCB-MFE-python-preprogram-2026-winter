"""HW 3: expiring memoization and recursive/iterative tree traversals."""

import time
from functools import wraps


def memorize_3s(func):
    """Cache each argument's successful result for three seconds after completion.

    The wrapped function takes one positional, hashable argument. Use
    time.monotonic(); cache hits do not reset expiry. See README.md for details.
    """
    # A separate dictionary is created for each decorated function.
    cache = {}

    @wraps(func)
    def wrapper(arg):
        # Check membership rather than truthiness: even None is a valid result.
        if arg in cache:
            result, completed_at = cache[arg]
            if time.monotonic() - completed_at < 3:
                return result

        # Exceptions propagate before the cache can be updated. Start the timer
        # after computation, so a slow function still gets three cached seconds.
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
    result = []
    stack = [root] if root is not None else []

    while stack:
        node = stack.pop()
        result.append(node.val)
        # Push right first so the last-in, first-out stack visits left first.
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
    result = []
    stack = []
    current = root

    while current is not None or stack:
        # Save ancestors while descending to the next leftmost node.
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
    """Return Left, Right, Root order using a loop and stack(s)."""
    result = []
    stack = [(root, False)] if root is not None else []

    while stack:
        node, ready_to_visit = stack.pop()
        if ready_to_visit:
            result.append(node.val)
        else:
            # Revisit this node after its children. Stack order makes the left
            # subtree finish first, then the right subtree, then the node.
            stack.append((node, True))
            if node.right is not None:
                stack.append((node.right, False))
            if node.left is not None:
                stack.append((node.left, False))

    return result
