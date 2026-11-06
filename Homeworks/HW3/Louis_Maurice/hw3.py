"""HW 3: memoization with expiry and binary-tree traversals."""

import time
from functools import wraps


def memorize_3s(func):
    """Cache each argument's successful result for three seconds after completion.

    The wrapped function takes one positional, hashable argument. Use
    time.monotonic(); cache hits do not reset expiry. See README.md for details.
    """
    cache = {}  # key is the argument, value is (result, time the result was computed)

    @wraps(func)
    def wrapper(arg):
        if arg not in cache or time.monotonic() - cache[arg][1] >= 3:
            result = func(arg)  # if func raises, nothing is stored in the cache
            cache[arg] = (result, time.monotonic())
        return cache[arg][0]
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

    def _preorder(node):
        if node:
            result.append(node.val)      # Visit node
            _preorder(node.left)         # Go left
            _preorder(node.right)        # Go right

    _preorder(root)
    return result


def pre_order_iteration(root: Node | None) -> list[str]:
    """Return Root, Left, Right order using a loop and stack(s)."""
    stack = [root]
    result = []

    while stack:
        node = stack.pop()
        if node:
            result.append(node.val)      # Visit node
            # Add moves in reverse order (for same sequence as recursive)
            stack.append(node.right)
            stack.append(node.left)

    return result


def in_order_recursion(root: Node | None) -> list[str]:
    """Return Left, Root, Right order using recursion; None returns []."""
    result = []

    def _inorder(node):
        if node:
            _inorder(node.left)          # Go left
            result.append(node.val)      # Visit node
            _inorder(node.right)         # Go right

    _inorder(root)
    return result


def in_order_iteration(root: Node | None) -> list[str]:
    """Return Left, Root, Right order using a loop and stack(s)."""
    stack = [(root, False)]  # (node, visited)
    result = []

    while stack:
        node, visited = stack.pop()
        if node:
            if visited:
                result.append(node.val)  # Visit node
            else:
                # Add moves in reverse order (for same sequence as recursive)
                stack.append((node.right, False))
                stack.append((node, True))
                stack.append((node.left, False))

    return result


def post_order_recursion(root: Node | None) -> list[str]:
    """Return Left, Right, Root order using recursion; None returns []."""
    result = []

    def _postorder(node):
        if node:
            _postorder(node.left)        # Go left
            _postorder(node.right)       # Go right
            result.append(node.val)      # Visit node

    _postorder(root)
    return result


def post_order_iteration(root: Node | None) -> list[str]:
    """Return Left, Right, Root order using a loop and stack(s)."""
    stack = [(root, False)]  # (node, visited)
    result = []

    while stack:
        node, visited = stack.pop()
        if node:
            if visited:
                result.append(node.val)  # Visit node
            else:
                # Add moves in reverse order (for same sequence as recursive)
                stack.append((node, True))
                stack.append((node.right, False))
                stack.append((node.left, False))

    return result
