"""HW3: memoization with expiry and recursive/iterative tree traversals."""

import time


def memorize_3s(func):
    """Cache each argument's successful result for three seconds after completion.

    The wrapped function takes one positional, hashable argument. Use
    time.monotonic(); cache hits do not reset expiry. See README.md for details.
    """
    cache = {}

    def wrapper(argument):
        if argument in cache:
            cached_result, completion_time = cache[argument]
            if time.monotonic() - completion_time < 3:
                return cached_result

        result = func(argument)
        completion_time = time.monotonic()

        cache[argument] = (result, completion_time)

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
    values = []

    def visit(node):
        if node is None:
            return

        values.append(node.val)
        visit(node.left)
        visit(node.right)

    visit(root)
    return values


def pre_order_iteration(root: Node | None) -> list[str]:
    """Return Root, Left, Right order using a loop and stack(s)."""
    if root is None:
        return []

    values = []
    pending_nodes = [root]

    while pending_nodes:
        node = pending_nodes.pop()
        values.append(node.val)

        # Push right first so that the left child is visited next.
        if node.right is not None:
            pending_nodes.append(node.right)
        if node.left is not None:
            pending_nodes.append(node.left)

    return values


def in_order_recursion(root: Node | None) -> list[str]:
    """Return Left, Root, Right order using recursion; None returns []."""
    values = []

    def visit(node):
        if node is None:
            return

        visit(node.left)
        values.append(node.val)
        visit(node.right)

    visit(root)
    return values


def in_order_iteration(root: Node | None) -> list[str]:
    """Return Left, Root, Right order using a loop and stack(s)."""
    values = []
    pending_nodes = []
    node = root

    while node is not None or pending_nodes:
        # Save each node until its left subtree has been visited.
        while node is not None:
            pending_nodes.append(node)
            node = node.left

        node = pending_nodes.pop()
        values.append(node.val)
        node = node.right

    return values


def post_order_recursion(root: Node | None) -> list[str]:
    """Return Left, Right, Root order using recursion; None returns []."""
    values = []

    def visit(node):
        if node is None:
            return

        visit(node.left)
        visit(node.right)
        values.append(node.val)

    visit(root)
    return values


def post_order_iteration(root: Node | None) -> list[str]:
    """Return Left, Right, Root order using a loop and stack(s)."""
    if root is None:
        return []

    values = []
    pending_nodes = [(root, False)]

    while pending_nodes:
        node, ready_to_record = pending_nodes.pop()

        if ready_to_record:
            values.append(node.val)
        else:
            # Record the parent only after its left and right subtrees.
            pending_nodes.append((node, True))
            if node.right is not None:
                pending_nodes.append((node.right, False))
            if node.left is not None:
                pending_nodes.append((node.left, False))

    return values
