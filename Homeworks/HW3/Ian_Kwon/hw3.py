"""HW 3 starter. Implement the seven functions, then run test_hw3.py."""

import time


def memorize_3s(func):
    """Cache each argument's successful result for three seconds after completion.

    The wrapped function takes one positional, hashable argument. Use
    time.monotonic(); cache hits do not reset expiry. See README.md for details.
    """
    cache = {}

    def wrapper(arg):
        if arg in cache:
            result, timestamp = cache[arg]

            if time.monotonic() - timestamp < 3:
                return result
        
        result = func(arg)
        timestamp = time.monotonic()

        cache[arg] = (result, timestamp)

        return result
    
    return wrapper
    raise NotImplementedError("Implement memorize_3s")


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

    return (
        [root.val]
        + pre_order_recursion(root.left)
        + pre_order_recursion(root.right)
    )
    raise NotImplementedError("Implement pre_order_recursion")


def pre_order_iteration(root: Node | None) -> list[str]:
    """Return Root, Left, Right order using a loop and stack(s)."""

    if root is None:
        return []

    result = []
    stack = [root]

    while stack:
        node = stack.pop()

        result.append(node.val)

        if node.right is not None:
            stack.append(node.right)

        if node.left is not None:
            stack.append(node.left)

    return result
    raise NotImplementedError("Implement pre_order_iteration")


def in_order_recursion(root: Node | None) -> list[str]:
    """Return Left, Root, Right order using recursion; None returns []."""

    if root is None:
        return []

    return (
        in_order_recursion(root.left)
        + [root.val]
        + in_order_recursion(root.right)
    )
    raise NotImplementedError("Implement in_order_recursion")


def in_order_iteration(root: Node | None) -> list[str]:
    """Return Left, Root, Right order using a loop and stack(s)."""

    result = []
    stack = []
    current = root

    while current is not None or stack:

        while current is not None:
            stack.append(current)
            current = current.left

        current = stack.pop()
        result.append(current.val)

        current = current.right

    return result
    raise NotImplementedError("Implement in_order_iteration")


def post_order_recursion(root: Node | None) -> list[str]:
    """Return Left, Right, Root order using recursion; None returns []."""

    if root is None:
        return []

    return (
        post_order_recursion(root.left)
        + post_order_recursion(root.right)
        + [root.val]
    )
    raise NotImplementedError("Implement post_order_recursion")


def post_order_iteration(root: Node | None) -> list[str]:
    """Return Left, Right, Root order using a loop and stack(s)."""

    if root is None:
        return []

    stack1 = [root]
    stack2 = []

    while stack1:
        node = stack1.pop()
        stack2.append(node)

        if node.left is not None:
            stack1.append(node.left)

        if node.right is not None:
            stack1.append(node.right)

    result = []

    while stack2:
        result.append(stack2.pop().val)

    return result
    raise NotImplementedError("Implement post_order_iteration")
