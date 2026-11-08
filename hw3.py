"""HW 3: memoization with expiry and binary-tree traversals."""

import time


def memorize_3s(func):
    """Cache each argument's result for less than three seconds after completion."""
    cache = {}

    def wrapper(arg):
        if arg in cache:
            result, finished_at = cache[arg]
            if time.monotonic() - finished_at < 3:
                return result
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
    """Return Root, Left, Right order using a stack of pending nodes."""
    result = []
    stack = [root] if root is not None else []
    while stack:
        node = stack.pop()
        result.append(node.val)
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
    """Return Left, Root, Right order using a stack of waiting ancestors."""
    result = []
    stack = []
    node = root
    while node is not None or stack:
        while node is not None:
            stack.append(node)
            node = node.left
        node = stack.pop()
        result.append(node.val)
        node = node.right
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
    """Return Left, Right, Root order using nodes paired with visit flags."""
    result = []
    stack = [(root, False)] if root is not None else []
    while stack:
        node, ready = stack.pop()
        if ready:
            result.append(node.val)
        else:
            stack.append((node, True))
            if node.right is not None:
                stack.append((node.right, False))
            if node.left is not None:
                stack.append((node.left, False))
    return result
