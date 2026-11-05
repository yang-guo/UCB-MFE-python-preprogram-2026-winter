"""HW 3 starter. Implement the seven functions, then run test_hw3.py."""

import time


def memorize_3s(func):
    """Cache each argument's successful result for three seconds after completion.

    The wrapped function takes one positional, hashable argument. Use
    time.monotonic(); cache hits do not reset expiry. See README.md for details.
    """
    cache = {}
    def wrapper(arg):
        curr_time = time.monotonic()
        if arg in cache:
            result, exp_time = cache[arg]
            if curr_time < exp_time:
                return result
        
        result = func(arg)
        finish_time = time.monotonic()
        cache[arg] = (result, finish_time + 3)

        return(result)
    return wrapper
    # raise NotImplementedError("Implement memorize_3s")


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
    # raise NotImplementedError("Implement pre_order_recursion")


def pre_order_iteration(root: Node | None) -> list[str]:
    """Return Root, Left, Right order using a loop and stack(s)."""
    result = []
    stack = []

    if root is None:
        return result
    stack.append(root)

    while len(stack)>0:
        node = stack.pop()
        result.append(node.val)
        if node.right is not None:
            stack.append(node.right)
        if node.left is not None:
            stack.append(node.left)

    return result
    # raise NotImplementedError("Implement pre_order_iteration")


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
    # raise NotImplementedError("Implement in_order_recursion")


def in_order_iteration(root: Node | None) -> list[str]:
    """Return Left, Root, Right order using a loop and stack(s)."""
    result = []
    stack = []
    current = root

    while current is not None or len(stack) > 0:
        while current is not None:
            stack.append(current)
            current = current.left
        current = stack.pop()
        result.append(current.val)
        current = current.right

    return result
    # raise NotImplementedError("Implement in_order_iteration")


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
    # raise NotImplementedError("Implement post_order_recursion")


def post_order_iteration(root: Node | None) -> list[str]:
    """Return Left, Right, Root order using a loop and stack(s)."""
    result = []
    stack = []

    if root is None:
        return result
    stack.append(root)

    while len(stack) > 0:
        node = stack.pop()
        result.append(node.val)
        if node.left is not None:
            stack.append(node.left)
        if node.right is not None:
            stack.append(node.right)

    result.reverse()
    return result
    # raise NotImplementedError("Implement post_order_iteration")
