"""HW 3 starter. Implement the seven functions, then run test_hw3.py."""

import time


def memorize_3s(func):
    cache = {}

    def wrapper(arg):
        now = time.monotonic()

        if arg in cache:
            value, saved_time = cache[arg]
            if now - saved_time < 3:
                return value

        value = func(arg)
        cache[arg] = (value, time.monotonic())

        return value

    return wrapper


class Node:
    """A binary-tree node; missing children are None. Leave this class unchanged."""

    def __init__(self, key: str, left=None, right=None):
        self.val = key
        self.left = left
        self.right = right


def pre_order_recursion(root):
    result = []

    def walk(node):
        if node is None:
            return

        result.append(node.val)
        walk(node.left)
        walk(node.right)

    walk(root)
    return result


def pre_order_iteration(root):
    result = []

    if root is None:
        return result

    stack = [root]

    while stack:
        node = stack.pop()
        result.append(node.val)

        if node.right is not None:
            stack.append(node.right)

        if node.left is not None:
            stack.append(node.left)

    return result


def in_order_recursion(root):
    result = []

    def walk(node):
        if node is None:
            return

        walk(node.left)
        result.append(node.val)
        walk(node.right)

    walk(root)
    return result


def in_order_iteration(root):
    result = []
    current = root
    stack = []

    while current is not None or stack:
        if current is not None:
            stack.append(current)
            current = current.left
        else:
            current = stack.pop()
            result.append(current.val)
            current = current.right

    return result


def post_order_recursion(root):
    result = []

    def walk(node):
        if node is None:
            return

        walk(node.left)
        walk(node.right)
        result.append(node.val)

    walk(root)
    return result


def post_order_iteration(root):
    result = []
    stack = [root] if root is not None else []

    while stack:
        node = stack.pop()
        result.append(node.val)

        if node.left is not None:
            stack.append(node.left)

        if node.right is not None:
            stack.append(node.right)

    result.reverse()
    return result
