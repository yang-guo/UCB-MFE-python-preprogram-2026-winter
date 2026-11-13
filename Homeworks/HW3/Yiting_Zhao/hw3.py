
import time


# Part 1: Memoization with 3-second expiry

def memorize_3s(func):
    cache = {}

    def wrapper(x):
        if x in cache and time.monotonic() - cache[x][1] < 3:
            return cache[x][0]

        result = func(x)
        cache[x] = (result, time.monotonic())
        return result

    return wrapper


# Part 2: Binary Tree

class Node:
    def __init__(self, key, left=None, right=None):
        self.val = key
        self.left = left
        self.right = right


# Recursive traversals

def pre_order_recursion(root):
    if root is None:
        return []

    return [root.val] + pre_order_recursion(root.left) + pre_order_recursion(root.right)


def in_order_recursion(root):
    if root is None:
        return []

    return in_order_recursion(root.left) + [root.val] + in_order_recursion(root.right)


def post_order_recursion(root):
    if root is None:
        return []

    return post_order_recursion(root.left) + post_order_recursion(root.right) + [root.val]


# Iterative traversals

def pre_order_iteration(root):
    if root is None:
        return []

    result = []
    stack = [root]

    while stack:
        node = stack.pop()
        result.append(node.val)

        if node.right:
            stack.append(node.right)
        if node.left:
            stack.append(node.left)

    return result


def in_order_iteration(root):
    result = []
    stack = []
    node = root

    while node or stack:
        while node:
            stack.append(node)
            node = node.left

        node = stack.pop()
        result.append(node.val)
        node = node.right

    return result


def post_order_iteration(root):
    if root is None:
        return []

    result = []
    stack = [root]

    while stack:
        node = stack.pop()
        result.append(node.val)

        if node.left:
            stack.append(node.left)
        if node.right:
            stack.append(node.right)

    return result[::-1]
