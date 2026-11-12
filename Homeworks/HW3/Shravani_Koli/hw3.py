"""HW 3 starter. Implement the seven functions, then run test_hw3.py."""

import time


def memorize_3s(func):
    """Cache each argument's successful result for three seconds after completion.

    The wrapped function takes one positional, hashable argument. Use
    time.monotonic(); cache hits do not reset expiry. See README.md for details.
    """
    cache = {}
    def inner_func(arg):
        if arg in cache:
            result, timestmp = cache[arg]
            if time.monotonic() - timestmp <3:
                return result
                
        result = func(arg)
        cache[arg] = (result, time.monotonic())

        return result
    return inner_func
    # raise NotImplementedError("Implement memorize_3s")


class Node:
    """A binary-tree node; missing children are None. Leave this class unchanged."""

    def __init__(self, key: str, left=None, right=None):
        self.val = key
        self.left = left
        self.right = right


def pre_order_recursion(root: Node | None) -> list[str]:
    """Return Root, Left, Right order using recursion; None returns []."""
    # raise NotImplementedError("Implement pre_order_recursion")
    if root is None:
        return []

    return ([root.val] + pre_order_recursion(root.left) +pre_order_recursion(root.right))


def pre_order_iteration(root: Node | None) -> list[str]:
    """Return Root, Left, Right order using a loop and stack(s)."""
    # raise NotImplementedError("Implement pre_order_iteration")
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


def in_order_recursion(root: Node | None) -> list[str]:
    """Return Left, Root, Right order using recursion; None returns []."""
    # raise NotImplementedError("Implement in_order_recursion")
    if root is None:
        return []
    return (in_order_recursion(root.left) +[root.val]+ in_order_recursion(root.right))



def in_order_iteration(root: Node | None) -> list[str]:
    """Return Left, Root, Right order using a loop and stack(s)."""
    # raise NotImplementedError("Implement in_order_iteration")
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


def post_order_recursion(root: Node | None) -> list[str]:
    """Return Left, Right, Root order using recursion; None returns []."""
    # raise NotImplementedError("Implement post_order_recursion")
    if root is None:
        return []
    return (post_order_recursion(root.left) + post_order_recursion(root.right) +[root.val])



def post_order_iteration(root: Node | None) -> list[str]:
    """Return Left, Right, Root order using a loop and stack(s)."""
    # raise NotImplementedError("Implement post_order_recursion")
    if root is None:
        return []
    result = []
    stack = [root]
    
    while stack:
        node = stack.pop()
        result.append(node.val)
        if node.left is not None:
            stack.append(node.left)
        if node.right is not None:
            stack.append(node.right)

    result.reverse()
    return result
    # raise NotImplementedError("Implement post_order_iteration")
