import time
from functools import wraps


def memorize_3s(func):
    """
    Decorator that caches the return value of a single-argument function
    for strictly less than 3 seconds using time.monotonic().
    """
    cache = {}  # Maps arg -> (result, finish_timestamp)

    @wraps(func)
    def wrapper(arg):
        now = time.monotonic()
        if arg in cache:
            result, timestamp = cache[arg]
            if now - timestamp < 3.0:
                return result

        # Compute result; if an exception occurs, it propagates naturally without caching
        result = func(arg)
        finish_time = time.monotonic()
        cache[arg] = (result, finish_time)
        return result

    return wrapper


class Node:
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


# ==========================================
# Pre-order Traversal (Root, Left, Right)
# ==========================================

def pre_order_recursion(root: Node | None) -> list:
    if root is None:
        return []
    return [root.val] + pre_order_recursion(root.left) + pre_order_recursion(root.right)


def pre_order_iteration(root: Node | None) -> list:
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


# ==========================================
# In-order Traversal (Left, Root, Right)
# ==========================================

def in_order_recursion(root: Node | None) -> list:
    if root is None:
        return []
    return in_order_recursion(root.left) + [root.val] + in_order_recursion(root.right)


def in_order_iteration(root: Node | None) -> list:
    result = []
    stack = []
    curr = root
    while curr is not None or stack:
        while curr is not None:
            stack.append(curr)
            curr = curr.left
        curr = stack.pop()
        result.append(curr.val)
        curr = curr.right
    return result


# ==========================================
# Post-order Traversal (Left, Right, Root)
# ==========================================

def post_order_recursion(root: Node | None) -> list:
    if root is None:
        return []
    return post_order_recursion(root.left) + post_order_recursion(root.right) + [root.val]


def post_order_iteration(root: Node | None) -> list:
    if root is None:
        return []
    result = []
    stack = []
    curr = root
    last_visited = None

    while curr is not None or stack:
        if curr is not None:
            stack.append(curr)
            curr = curr.left
        else:
            peek_node = stack[-1]
            if peek_node.right is not None and last_visited is not peek_node.right:
                curr = peek_node.right
            else:
                result.append(peek_node.val)
                last_visited = stack.pop()
    return result