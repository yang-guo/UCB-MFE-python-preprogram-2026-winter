"""HW 3 starter. Implement the seven functions, then run test_hw3.py."""

import time


def memorize_3s(func):
    cache = {}

    def wrapper(r):
        if r in cache and time.monotonic() - cache[r][0] < 3:
            return cache[r][1]
        result = func(r)
        cache[r] = [time.monotonic(), result]
        return result

    return wrapper


class Node:
    """A binary-tree node; missing children are None. Leave this class unchanged."""

    def __init__(self, key: str, left=None, right=None):
        self.val = key
        self.left = left
        self.right = right


def pre_order_recursion(root):
    if root is None:
        return []
    else:
        return [root.val] + pre_order_recursion(root.left) + pre_order_recursion(root.right)


def pre_order_iteration(root):
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
    if root is None:
        return []
    else:
        return in_order_recursion(root.left) + [root.val] + in_order_recursion(root.right)
    

def in_order_iteration(root):
    result = []
    stack = []
    node = root
    while stack or node is not None:
        while node is not None:
            stack.append(node)
            node = node.left
        node = stack.pop()
        result.append(node.val)
        node = node.right
    return result


def post_order_recursion(root: Node | None) -> list[str]:
    if root is None:
        return []
    else:
        return post_order_recursion(root.left) + post_order_recursion(root.right) + [root.val]
    


def post_order_iteration(root):
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
    return result[::-1]