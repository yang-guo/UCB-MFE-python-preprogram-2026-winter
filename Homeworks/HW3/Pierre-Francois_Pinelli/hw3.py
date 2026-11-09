import time


def memorize_3s(func):
    cache = {}  # each function has its own cache

    def wrapper(x):
        if x in cache:
            result, finished_at = cache[x]
            if time.monotonic() - finished_at < 3:
                return result  # return the result if it is still valid

        result = func(x)
        cache[x] = (result, time.monotonic())  # save the result and the time
        return result

    return wrapper


class Node:
    """A binary-tree node; missing children are None. Leave this class unchanged."""
    
    def __init__(self, key: str, left=None, right=None):
        self.val = key
        self.left = left
        self.right = right


def pre_order_recursion(root: Node | None) -> list[str]:
    if root is None:
        return []
    return [root.val] + pre_order_recursion(root.left) + pre_order_recursion(root.right)


def pre_order_iteration(root: Node | None) -> list[str]:
    result = []
    stack = [(root, False)]  # False means the node is not ready to be added

    while stack:
        node, ready = stack.pop()
        if node is None:
            continue
        if ready:
            result.append(node.val)
        else:
            # reverse order because the stack is LIFO
            stack.append((node.right, False))
            stack.append((node.left, False))
            stack.append((node, True))

    return result


def in_order_recursion(root: Node | None) -> list[str]:
    if root is None:
        return []
    return in_order_recursion(root.left) + [root.val] + in_order_recursion(root.right)


def in_order_iteration(root: Node | None) -> list[str]:
    result = []
    stack = [(root, False)]

    while stack:
        node, ready = stack.pop()
        if node is None:
            continue
        if ready:
            result.append(node.val)
        else:
            # push in reverse order -> left, root, right
            stack.append((node.right, False))
            stack.append((node, True))
            stack.append((node.left, False))

    return result


def post_order_recursion(root: Node | None) -> list[str]:
    if root is None:
        return []
    return post_order_recursion(root.left) + post_order_recursion(root.right) + [root.val]


def post_order_iteration(root: Node | None) -> list[str]:
    result = []
    stack = [(root, False)]

    while stack:
        node, ready = stack.pop()
        if node is None:
            continue
        if ready:
            result.append(node.val)
        else:
            # push in reverse order -> left, right, root
            stack.append((node, True))
            stack.append((node.right, False))
            stack.append((node.left, False))

    return result
