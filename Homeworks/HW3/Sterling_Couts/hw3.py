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
            result, completion_time = cache[arg]
            if time.monotonic() - completion_time < 3:
                return result

        result = func(arg)
        completion_time = time.monotonic()
        cache[arg] = (result, completion_time)
        return result

    return wrapper




class Node:
    """A binary-tree node; missing children are None. Leave this class unchanged."""

    def __init__(self, key: str, left=None, right=None):
        self.val = key
        self.left = left
        self.right = right


def pre_order_recursion(root: Node | None) -> list[str]:
    """Return node values in Root, Left, Right order."""
    if root is None:
        return []

    left_values = pre_order_recursion(root.left)
    right_values = pre_order_recursion(root.right)

    return [root.val] + left_values + right_values



def pre_order_iteration(root: Node | None) -> list[str]:
    """Return Root, Left, Right order using an explicit stack."""
    if root is None:
        return []

    values = []
    pending = [root]

    while pending:
        node = pending.pop()
        values.append(node.val)

        # Push right first so the left subtree is processed first.
        if node.right is not None:
            pending.append(node.right)

        if node.left is not None:
            pending.append(node.left)

    return values



def in_order_recursion(root: Node | None) -> list[str]:
    """Return node values in Left, Root, Right order."""
    if root is None:
        return []

    left_values = in_order_recursion(root.left)
    right_values = in_order_recursion(root.right)

    return left_values + [root.val] + right_values



def in_order_iteration(root: Node | None) -> list[str]:
    """Return Left, Root, Right order using an explicit stack."""
    values = []
    pending = []
    current = root

    while current is not None or pending:
        # Save ancestors while descending to the leftmost node.
        while current is not None:
            pending.append(current)
            current = current.left

        current = pending.pop()
        values.append(current.val)

        # Explore this node's right subtree next.
        current = current.right

    return values


def post_order_recursion(root: Node | None) -> list[str]:
    """Return node values in Left, Right, Root order."""
    if root is None:
        return []

    left_values = post_order_recursion(root.left)
    right_values = post_order_recursion(root.right)

    return left_values + right_values + [root.val]



def post_order_iteration(root: Node | None) -> list[str]:
    """Return Left, Right, Root order using an explicit stack."""
    if root is None:
        return []

    values = []
    pending = [(root, False)]

    while pending:
        node, ready_to_record = pending.pop()

        if ready_to_record:
            values.append(node.val)
        else:
            # Schedule this node after its children.
            pending.append((node, True))

            if node.right is not None:
                pending.append((node.right, False))

            if node.left is not None:
                pending.append((node.left, False))

    return values

