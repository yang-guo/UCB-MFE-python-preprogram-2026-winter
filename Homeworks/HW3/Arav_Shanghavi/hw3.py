"""HW 3 starter. Implement the seven functions, then run test_hw3.py."""

import time

def memorize_3s(func):
    """Cache each argument's successful result for three seconds after completion.

    The wrapped function takes one positional, hashable argument. Use
    time.monotonic(); cache hits do not reset expiry. See README.md for details.
    """
    cache_table = {}

    def inner(arg):
        current_time =  time.monotonic()

        if arg in cache_table:
            result, finished_time = cache_table[arg]

            if current_time - finished_time < 3:
                return result

        result = func(arg)
        finished_at = time.monotonic()

        cache_table[arg] = (result, finished_at)

        return result

    return inner


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

    if root is None:
        return []
    lists = []
    collected = []
    lists.append(root)

    while lists:
        popped_value =  lists.pop()
        collected.append(str(popped_value.val))
        if popped_value.right:
            lists.append(popped_value.right)
        if popped_value.left:
            lists.append(popped_value.left)

    return collected

def in_order_recursion(root: Node | None) -> list[str]:
    """Return Left, Root, Right order using recursion; None returns []."""
    if not root:
        return []

    return in_order_recursion(root.left) + [root.val] + in_order_recursion(root.right)


def in_order_iteration(root: Node | None) -> list[str]:
    """Return Left, Root, Right order using a loop and stack(s)."""
    if root is None:
        return []

    stack = []
    result = []
    current = root

    while current is not None or stack:
        while current is not None:
            stack.append(current)
            current = current.left
        current = stack.pop()
        result.append(current.val)
        current = current.right

    return result

def post_order_recursion(root: Node | None) -> list[str]:
    """Return Left, Right, Root order using recursion; None returns []."""
    if not root:
        return []

    return post_order_recursion(root.left) + post_order_recursion(root.right) + [root.val]


def post_order_iteration(root: Node | None) -> list[str]:
    """Return Left, Right, Root order using a loop and stack(s)."""
    if root is None:
        return []

    stack1 = []
    stack2 = []
    result = []
    stack1.append(root)

    while stack1:
        current = stack1.pop()
        stack2.append(current)

        if current.left:
            stack1.append(current.left)

        if current.right:
            stack1.append(current.right)

    while stack2:
        result.append(stack2.pop().val)

    return result