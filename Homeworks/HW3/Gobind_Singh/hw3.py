"""HW 3 starter. Implement the seven functions, then run test_hw3.py."""

import time


def memorize_3s(func):
    """Cache each argument's successful result for three seconds after completion.

    The wrapped function takes one positional, hashable argument. Use
    time.monotonic(); cache hits do not reset expiry. See README.md for details.
    """
    # one cache per decorated function: arg -> (result, time the call finished)
    cache = {}

    def wrapper(arg):
        now = time.monotonic()
        if arg in cache:
            value, finished_at = cache[arg]
            # a hit just returns the old value; it does not move finished_at
            if now - finished_at < 3:
                return value

        value = func(arg)
        # stamp after the call. if func raises we never store anything
        cache[arg] = (value, time.monotonic())
        return value

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

    def walk(node):
        if node is None:
            return
        result.append(node.val)
        walk(node.left)
        walk(node.right)

    walk(root)
    return result


def pre_order_iteration(root: Node | None) -> list[str]:
    """Return Root, Left, Right order using a loop and stack(s)."""
    result = []
    stack = []
    if root is not None:
        stack.append(root)

    while len(stack) > 0:
        node = stack.pop()
        result.append(node.val)
        # push right first so left is the next one we pop
        if node.right is not None:
            stack.append(node.right)
        if node.left is not None:
            stack.append(node.left)

    return result


def in_order_recursion(root: Node | None) -> list[str]:
    """Return Left, Root, Right order using recursion; None returns []."""
    result = []

    def walk(node):
        if node is None:
            return
        walk(node.left)
        result.append(node.val)
        walk(node.right)

    walk(root)
    return result


def in_order_iteration(root: Node | None) -> list[str]:
    """Return Left, Root, Right order using a loop and stack(s)."""
    result = []
    stack = []
    node = root

    while node is not None or len(stack) > 0:
        # go all the way left, saving the nodes we pass
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

    def walk(node):
        if node is None:
            return
        walk(node.left)
        walk(node.right)
        result.append(node.val)

    walk(root)
    return result


def post_order_iteration(root: Node | None) -> list[str]:
    """Return Left, Right, Root order using a loop and stack(s)."""
    # second value is whether both children have already been pushed
    result = []
    stack = []
    if root is not None:
        stack.append((root, False))

    while len(stack) > 0:
        node, seen_children = stack.pop()
        if seen_children:
            result.append(node.val)
            continue
        stack.append((node, True))
        # push right then left so left is handled first
        if node.right is not None:
            stack.append((node.right, False))
        if node.left is not None:
            stack.append((node.left, False))

    return result
