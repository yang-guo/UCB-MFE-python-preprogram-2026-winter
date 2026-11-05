"""HW 3 starter. Implement the seven functions, then run test_hw3.py."""

import time


def memorize_3s(func):
    """Cache each argument's successful result for three seconds after completion.
    The wrapped function takes one positional, hashable argument. Use
    time.monotonic(); cache hits do not reset expiry. See README.md for details.
    """
    cache={}

    def wrapper(arg):
        if arg in cache and (time.monotonic() - cache[arg][1] < 3):
            return cache[arg][0]
        value = func(arg)
        cache[arg] = [value, time.monotonic()]
        return value
    return wrapper


class Node:
    """A binary-tree node; missing children are None. Leave this class unchanged."""

    def __init__(self, key: str, left=None, right=None):
        self.val = key
        self.left = left
        self.right = right


def pre_order_recursion(root: Node | None) -> list[str]:
    """Return Root, Left, Right order using recursion; None returns []."""
    order_list = []

    def _preorder(node):
        if node is not None:
            order_list.append(node.val)
            _preorder(node.left)
            _preorder(node.right)

    _preorder(root)
    return order_list

def pre_order_iteration(root: Node | None) -> list[str]:
    """Return Root, Left, Right order using a loop and stack(s)."""
    order_list = []
    stack = [root]
    while stack:
        node = stack.pop()
        if node is not None:
            order_list.append(node.val)
            stack.append(node.right)
            stack.append(node.left)
            
    return order_list


def in_order_recursion(root: Node | None) -> list[str]:
    """Return Left, Root, Right order using recursion; None returns []."""
    order_list = []

    def _inorder(node):
        if node is not None:
            _inorder(node.left)
            order_list.append(node.val)
            _inorder(node.right)
    
    _inorder(root)
    return order_list


def in_order_iteration(root: Node | None) -> list[str]:
    order_list = []
    stack = []
    current_node = root
    while stack or current_node:
        while current_node is not None:
            stack.append(current_node)
            current_node = current_node.left
        node = stack.pop()
        order_list.append(node.val)
        current_node = node.right
    return order_list


def post_order_recursion(root: Node | None) -> list[str]:
    """Return Left, Right, Root order using recursion; None returns []."""
    order_list = []

    def _postorder(node):
        if node is not None:
            _postorder(node.left)
            _postorder(node.right)
            order_list.append(node.val)
    
    _postorder(root)
    return order_list


def post_order_iteration(root: Node | None) -> list[str]:
    """Return Left, Right, Root order using a loop and stack(s)."""
    raise NotImplementedError("Implement post_order_iteration")
