"""HW 3 starter. Implement the seven functions, then run test_hw3.py."""

import time


def memorize_3s(func):
    """Cache each argument's successful result for three seconds after completion.

    The wrapped function takes one positional, hashable argument. Use
    time.monotonic(); cache hits do not reset expiry. See README.md for details.
    """
    cache = {}
    def wrapper(x):
        if x in cache:
            result, stamp = cache[x]
            elapsed = time.monotonic() - stamp
            if elapsed < 3: 
                return result
        result = func(x)
        cache[x] = (result, time.monotonic())
        return result

    return wrapper


class Node:
    """A binary-tree node; missing children are None. Leave this class unchanged."""

    def __init__(self, key: str, left=None, right=None):
        self.val = key
        self.left = left
        self.right = right


def pre_order_recursion(root: Node | None) -> list[str]:
    """Return Root, Left, Right order using recursion; None returns []."""
    if root is None:
        return []
    left_side = pre_order_recursion(root.left)
    right_side =pre_order_recursion(root.right)
    return [root.val] + left_side + right_side


def pre_order_iteration(root: Node | None) -> list[str]:
    """Return Root, Left, Right order using a loop and stack(s)."""
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
    result = []

    def walk(node):
        if node is None: return
        walk(node.left)
        result.append(node.val)
        walk(node.right)
    walk(root)
    return result

def in_order_iteration(root: Node | None) -> list[str]:
    """Return Left, Root, Right order using a loop and stack(s)."""
    result = []
    stack = []
    current = root

    while current is not None or stack:
        while current is not None:
            # keep tracking the left most ones until hit empty node
            stack.append(current)
            current = current.left
        # self-note when to remind me the logic
        # when the left side hit empty node, then pop the left most one, and switch to the right
        # so in the base case after the left one is poped, since there's no right one, stack pops
        # its parent node, gets recorded, and then move to the parent node's right one
        node = stack.pop()
        result.append(node.val)
        # then handles the right in the same way: left -> parent -> right
        current = node.right
    return result


def post_order_recursion(root: Node | None) -> list[str]:
    """Return Left, Right, Root order using recursion; None returns []."""
    # similar to in_order_recursion
    result = [] 
    
    def walk(node):
        if node is None: return
        walk(node.left)
        walk(node.right)
        result.append(node.val)
    walk(root)
    return result



def post_order_iteration(root: Node | None) -> list[str]:
    """Return Left, Right, Root order using a loop and stack(s)."""
    # similar to pre_order_iteration
    if root is None:
        return []
    result = []
    stack = [root]
    while stack:
        node = stack.pop()
        result.append(node.val)

        # flip the order of the if statements, so the order of traversal 
        # becomes root -> right -> left
        if node.left is not None:
            stack.append(node.left)
        if node.right is not None:
            stack.append(node.right)

    # now flip the result list so the order is left -> right -> root
    result.reverse()
    return result   
