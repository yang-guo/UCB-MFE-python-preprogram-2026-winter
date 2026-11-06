"""HW 3 starter. Implement the seven functions, then run test_hw3.py."""

import functools
import time


def memorize_3s(func):
    """Cache each argument's successful result for three seconds after completion.

    The wrapped function takes one positional, hashable argument. Use
    time.monotonic(); cache hits do not reset expiry. See README.md for details.
    """
    cache = {}

    @functools.wraps(func)
    def wrapper_function(arg):
        if arg in cache:
            output, finished = cache[arg]
            if time.monotonic() < finished + 3:
                return output

        output = func(arg)
        cache[arg] = (output, time.monotonic())
        return output

    return wrapper_function


class Node:
    """A binary-tree node; missing children are None. Leave this class unchanged."""

    def __init__(self, key: str, left=None, right=None):
        self.val = key
        self.left = left
        self.right = right


def pre_order_recursion(root: Node | None) -> list[str]:
    """Return Root, Left, Right order using recursion; None returns []."""
    if root == None:
        return []

    output = [root.val]
    output += pre_order_recursion(root.left)
    output += pre_order_recursion(root.right)
    return output


def pre_order_iteration(root: Node | None) -> list[str]:
    """Return Root, Left, Right order using a loop and stack(s)."""
    output = []
    stack = [root]

    while stack: 
        node = stack.pop()
        if node is None: 
            continue
        output.append(node.val)
        stack.append(node.right)
        stack.append(node.left)

    return output


def in_order_recursion(root: Node | None) -> list[str]:
    """Return Left, Root, Right order using recursion; None returns []."""
    if root == None:
        return []

    output = []
    output += in_order_recursion(root.left)
    output += [root.val]
    output += in_order_recursion(root.right)
    return output


def in_order_iteration(root: Node | None) -> list[str]:
    """Return Left, Root, Right order using a loop and stack(s)."""
    output = []
    stack = []

    node = root
    while node is not None or stack:
        if node is not None: 
            stack.append(node)
            node = node.left

        else: 
            node = stack.pop()
            output.append(node.val)
            node = node.right
    return output 

def post_order_recursion(root: Node | None) -> list[str]:
    """Return Left, Right, Root order using recursion; None returns []."""
    if root == None: 
        return []

    output = post_order_recursion(root.left) + post_order_recursion(root.right)
    output.append(root.val)
    return output


def post_order_iteration(root: Node | None) -> list[str]:
    """Return Left, Right, Root order using a loop and stack(s)."""
    if root == None: 
        return []

    output = []
    stack = [root]

    while stack: 
        node = stack.pop()
        output.insert(0, node.val)

        if node.left is not None: 
            stack.append(node.left)

        if node.right is not None:
            stack.append(node.right)

    return output
