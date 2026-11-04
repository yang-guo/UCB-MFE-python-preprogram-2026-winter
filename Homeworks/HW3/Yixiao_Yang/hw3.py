import time
from functools import wraps


def memorize_3s(func):
    cache = {}

    @wraps(func)
    def wrapper(arg):
        now = time.monotonic()
        if arg in cache:
            value, completed_at = cache[arg]
            if now - completed_at < 3:
                return value

        # A failed call raises normally and does not change the cache.
        result = func(arg)
        # Measure after completion: the computation itself is not part of the TTL.
        cache[arg] = (result, time.monotonic())
        return result

    return wrapper


class Node:

    def __init__(self, key: str, left=None, right=None):
        self.val = key
        self.left = left
        self.right = right


def pre_order_recursion(root: Node | None) -> list[str]:
   
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
    
    if root is None:
        return []

    values = []
    stack = [root]
    while stack:
        node = stack.pop()
        values.append(node.val)
        # The stack is LIFO: push right before left.
        if node.right is not None:
            stack.append(node.right)
        if node.left is not None:
            stack.append(node.left)
    return values


def in_order_recursion(root: Node | None) -> list[str]:
    
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
    
    values = []
    stack = []
    current = root

    while current is not None or stack:
        while current is not None:
            stack.append(current)
            current = current.left

        current = stack.pop()
        values.append(current.val)
        current = current.right

    return values


def post_order_recursion(root: Node | None) -> list[str]:
    
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
    
    if root is None:
        return []

    # Process Root, Right, Left, then reverse for Left, Right, Root.
    reverse_values = []
    stack = [root]
    while stack:
        node = stack.pop()
        reverse_values.append(node.val)
        if node.left is not None:
            stack.append(node.left)
        if node.right is not None:
            stack.append(node.right)
    return reverse_values[::-1]
