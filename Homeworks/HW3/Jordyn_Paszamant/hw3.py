"""HW 3 starter. Implement the seven functions, then run test_hw3.py."""

import time


def memorize_3s(func):
    """Cache each argument's successful result for three seconds after completion.

    The wrapped function takes one positional, hashable argument. Use
    time.monotonic(); cache hits do not reset expiry. See README.md for details.
    """
    cache = {}  # arg -> (result, time when func finished)

    def wrapper(arg):
        if arg in cache:
            result, saved_time = cache[arg]
            age = time.monotonic() - saved_time
            if age < 3:
                return result

        # not cached, or expired: run the real function
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
    result = [root.val]
    result = result + pre_order_recursion(root.left)
    result = result + pre_order_recursion(root.right)
    return result


def pre_order_iteration(root: Node | None) -> list[str]:
    """Return Root, Left, Right order using a loop and stack(s)."""
    result = []
    if root is None:
        return result

    stack = [root]
    while len(stack) > 0:
        node = stack.pop()
        result.append(node.val)
        # push right first so left comes off the stack first
        if node.right is not None:
            stack.append(node.right)
        if node.left is not None:
            stack.append(node.left)
    return result


def in_order_recursion(root: Node | None) -> list[str]:
    """Return Left, Root, Right order using recursion; None returns []."""
    if root is None:
        return []
    result = in_order_recursion(root.left)
    result.append(root.val)
    result = result + in_order_recursion(root.right)
    return result


def in_order_iteration(root: Node | None) -> list[str]:
    """Return Left, Root, Right order using a loop and stack(s)."""
    result = []
    stack = []
    current = root

    while current is not None or len(stack) > 0:
        # keep going left, saving each node on the way down
        while current is not None:
            stack.append(current)
            current = current.left

        # nothing more to the left, so visit the node on top
        current = stack.pop()
        result.append(current.val)

        # then do its right subtree
        current = current.right
    return result


def post_order_recursion(root: Node | None) -> list[str]:
    """Return Left, Right, Root order using recursion; None returns []."""
    if root is None:
        return []
    result = post_order_recursion(root.left)
    result = result + post_order_recursion(root.right)
    result.append(root.val)
    return result


def post_order_iteration(root: Node | None) -> list[str]:
    """Return Left, Right, Root order using a loop and stack(s)."""
    result = []
    if root is None:
        return result

    # stack1 does the walking, stack2 collects nodes in Root, Right, Left order.
    # Popping stack2 afterwards gives Left, Right, Root.
    stack1 = [root]
    stack2 = []
    while len(stack1) > 0:
        node = stack1.pop()
        stack2.append(node)
        if node.left is not None:
            stack1.append(node.left)
        if node.right is not None:
            stack1.append(node.right)

    while len(stack2) > 0:
        node = stack2.pop()
        result.append(node.val)
    return result