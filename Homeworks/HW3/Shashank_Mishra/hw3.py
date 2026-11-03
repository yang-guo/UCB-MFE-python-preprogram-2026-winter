"""HW 3: memoization with expiry and binary tree traversals."""

import time


def memorize_3s(func):
    """Cache each argument's successful result for three seconds after completion.

    The wrapped function takes one positional, hashable argument. Use
    time.monotonic(); cache hits do not reset expiry. See README.md for details.
    """
    cache = {}

    def wrapper(arg):
        # Check if the cache hit is strictly under the 3-second limit
        if arg in cache:
            cached_result, timestamp = cache[arg]
            if time.monotonic() - timestamp < 3.0:
                return cached_result

        # Recompute on cache miss, expiry, or exactly 3 seconds
        result = func(arg)

        # Update cache with the new result and completion time
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
    result.extend(pre_order_recursion(root.left))
    result.extend(pre_order_recursion(root.right))
    return result


def pre_order_iteration(root: Node | None) -> list[str]:
    """Return Root, Left, Right order using a loop and stack(s)."""
    if root is None:
        return []

    stack = [root]
    result = []

    while stack:
        node = stack.pop()
        result.append(node.val)

        # Push the right child first so the left child is popped and processed first
        if node.right is not None:
            stack.append(node.right)
        if node.left is not None:
            stack.append(node.left)

    return result


def in_order_recursion(root: Node | None) -> list[str]:
    """Return Left, Root, Right order using recursion; None returns []."""
    if root is None:
        return []

    result = []
    result.extend(in_order_recursion(root.left))
    result.append(root.val)
    result.extend(in_order_recursion(root.right))
    return result


def in_order_iteration(root: Node | None) -> list[str]:
    """Return Left, Root, Right order using a loop and stack(s)."""
    stack = []
    result = []
    current = root

    while current is not None or stack:
        # Traverse to the leftmost node
        while current is not None:
            stack.append(current)
            current = current.left

        # Process the node and move to its right subtree
        current = stack.pop()
        result.append(current.val)
        current = current.right

    return result


def post_order_recursion(root: Node | None) -> list[str]:
    """Return Left, Right, Root order using recursion; None returns []."""
    if root is None:
        return []

    result = []
    result.extend(post_order_recursion(root.left))
    result.extend(post_order_recursion(root.right))
    result.append(root.val)
    return result


def post_order_iteration(root: Node | None) -> list[str]:
    """Return Left, Right, Root order using a loop and stack(s)."""
    if root is None:
        return []

    stack = [root]
    result = []

    # Process as Root, Right, Left
    while stack:
        node = stack.pop()
        result.append(node.val)

        if node.left is not None:
            stack.append(node.left)
        if node.right is not None:
            stack.append(node.right)

    # Reverse to achieve Left, Right, Root order
    return result[::-1]
