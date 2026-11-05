"""HW 3 starter. Implement the seven functions, then run test_hw3.py."""

import functools
import time


def memorize_3s(func):
    """Cache each argument's successful result for three seconds after completion.

    The wrapped function takes one positional, hashable argument. Use
    time.monotonic(); cache hits do not reset expiry. See README.md for details.
    """
    # 1. Created ONCE per decorated function (separate cache per function)
    # Maps: arg -> (result, finished_time)
    cache = {}

    @functools.wraps(func)
    def wrapper(arg):
        # 2. Check the clock when the call arrives
        now = time.monotonic()

        # 3. Check for a fresh cache hit (strictly less than 3 seconds old)
        if arg in cache:
            cached_result, finished_time = cache[arg]
            if now - finished_time < 3:
                # Return immediately without updating finished_time
                return cached_result

        # 4. Cache miss or expired: call the underlying function
        # If func(arg) raises an exception, execution stops here and nothing is cached
        result = func(arg)

        # 5. Record the timestamp AFTER func(arg) finishes
        completed_at = time.monotonic()
        cache[arg] = (result, completed_at)
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
    return (
        [root.val]
        + pre_order_recursion(root.left)
        + pre_order_recursion(root.right)
    )


def pre_order_iteration(root: Node | None) -> list[str]:
    """Return Root, Left, Right order using a loop and stack(s)."""
    if root is None:
        return []

    result = []
    stack = [root]

    while stack:
        node = stack.pop()
        result.append(node.val)

        # Push right first so left sits on top of the stack and is popped next
        if node.right is not None:
            stack.append(node.right)
        if node.left is not None:
            stack.append(node.left)

    return result


def in_order_recursion(root: Node | None) -> list[str]:
    """Return Left, Root, Right order using recursion; None returns []."""
    if root is None:
        return []
    return (
        in_order_recursion(root.left)
        + [root.val]
        + in_order_recursion(root.right)
    )


def in_order_iteration(root: Node | None) -> list[str]:
    """Return Left, Root, Right order using a loop and stack(s)."""
    result = []
    stack = []
    current = root

    while current is not None or stack:
        # 1. Walk left from current as far as possible, pushing each node passed
        while current is not None:
            stack.append(current)
            current = current.left

        # 2. Pop a node: its left subtree is completely done, so visit it now
        node = stack.pop()
        result.append(node.val)

        # 3. Move current to that node's right child and repeat
        current = node.right

    return result


def post_order_recursion(root: Node | None) -> list[str]:
    """Return Left, Right, Root order using recursion; None returns []."""
    if root is None:
        return []
    return (
        post_order_recursion(root.left)
        + post_order_recursion(root.right)
        + [root.val]
    )


def post_order_iteration(root: Node | None) -> list[str]:
    """Return Left, Right, Root order using a loop and stack(s)."""
    if root is None:
        return []

    result = []
    stack = [root]

    while stack:
        node = stack.pop()
        result.append(node.val)

        # Push left first so right is on top of the stack and visited first (Root, Right, Left)
        if node.left is not None:
            stack.append(node.left)
        if node.right is not None:
            stack.append(node.right)

    # Reverse (Root, Right, Left) to get (Left, Right, Root)
    result.reverse()
    return result
