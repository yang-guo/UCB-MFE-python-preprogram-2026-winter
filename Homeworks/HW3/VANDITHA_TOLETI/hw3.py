"""HW 3 starter. Implement the seven functions, then run test_hw3.py."""

import time
from functools import wraps

def memorize_3s(func):
    """Cache each argument's successful result for 3 seconds after completion.

    The wrapped function takes one positional, hashable argument.
    Cache hits do not reset expiry.
    """

    cache = {}

    @wraps(func)
    def wrapper(arg):
        # Get the current monotonic time
        current_time = time.monotonic()

        # Check whether a valid cached result exists
        if arg in cache:
            result, expiry = cache[arg]

            if current_time < expiry:
                return result

            # Remove expired cache entry
            del cache[arg]

        # Execute the original function
        result = func(arg)

        # Set expiry 3 seconds after function completion
        expiry = time.monotonic() + 3

        # Cache the successful result
        cache[arg] = (result, expiry)

        return result

    return wrapper


class Node:
    """A binary-tree node; missing children are None. Leave this class unchanged."""

    def __init__(self, key: str, left=None, right=None):
        self.val = key
        self.left = left
        self.right = right


def pre_order_recursion(root: Node | None) -> list[str]:
    """Preorder: Visit Root, then Left subtree, then Right subtree."""

    # Base case: Empty tree
    if root is None:
        return []

    result = [root.val]

    # Recursively traverse left subtree
    result.extend(pre_order_recursion(root.left))

    # Recursively traverse right subtree
    result.extend(pre_order_recursion(root.right))

    return result


def pre_order_iteration(root: Node | None) -> list[str]:
    """Preorder: Iterative DFS using a stack."""

    if root is None:
        return []

    stack = [root]
    result = []

    while stack:
        # Pop the top node and visit it
        node = stack.pop()
        result.append(node.val)

        # Push right first because stack follows LIFO
        if node.right:
            stack.append(node.right)

        # Push left second so it gets processed first
        if node.left:
            stack.append(node.left)

    return result


def in_order_recursion(root: Node | None) -> list[str]:
    """Inorder: Traverse Left subtree, visit Root, traverse Right subtree."""

    if root is None:
        return []

    result = []

    # First traverse left subtree
    result.extend(in_order_recursion(root.left))

    # Then visit root
    result.append(root.val)

    # Finally traverse right subtree
    result.extend(in_order_recursion(root.right))

    return result


def in_order_iteration(root: Node | None) -> list[str]:
    """Inorder: Iterative traversal using a stack."""

    if root is None:
        return []

    stack = []
    result = []
    current = root

    while current or stack:

        # Keep going left and pushing nodes onto stack
        while current:
            stack.append(current)
            current = current.left

        # Visit the leftmost unvisited node
        current = stack.pop()
        result.append(current.val)

        # Now explore its right subtree
        current = current.right

    return result


def post_order_recursion(root: Node | None) -> list[str]:
    """Postorder: Traverse Left, then Right, then visit Root."""

    if root is None:
        return []

    result = []

    # Traverse left subtree
    result.extend(post_order_recursion(root.left))

    # Traverse right subtree
    result.extend(post_order_recursion(root.right))

    # Visit root only after both subtrees
    result.append(root.val)

    return result


def post_order_iteration(root: Node | None) -> list[str]:
    """Postorder: Iterative traversal using two stacks."""

    if root is None:
        return []

    stack1 = [root]
    stack2 = []
    result = []

    # First stack processes nodes
    while stack1:
        node = stack1.pop()
        stack2.append(node)

        # Push left, then right
        if node.left:
            stack1.append(node.left)

        if node.right:
            stack1.append(node.right)

    # Second stack reverses the processing order
    while stack2:
        result.append(stack2.pop().val)

    return result
