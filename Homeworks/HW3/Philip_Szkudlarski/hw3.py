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
    def wrapper(arg):
        now = time.monotonic()

        # Check whether the argument is already cached
        if arg in cache:
            expiry, result = cache[arg]

            # Return cached result if it has not expired
            if now < expiry:
                return result

        # Calculate result if not cached or expired
        result = func(arg)

        # Store result with expiration time starting after completion
        cache[arg] = (time.monotonic() + 3, result)

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

    result = []

    def _preorder(node):
        if node is not None:
            result.append(node.val)
            _preorder(node.left)
            _preorder(node.right)

    _preorder(root)

    return result


def pre_order_iteration(root: Node | None) -> list[str]:
    """Return Root, Left, Right order using a loop and stack(s)."""

    if root is None:
        return []

    result = []
    stack = [root]

    while stack:

        node = stack.pop()
        result.append(node.val)

        # Push right first so left gets processed first
        if node.right is not None:
            stack.append(node.right)

        if node.left is not None:
            stack.append(node.left)

    return result


def in_order_recursion(root: Node | None) -> list[str]:
    """Return Left, Root, Right order using recursion; None returns []."""

    result = []

    def _inorder(node):
        if node is not None:
            _inorder(node.left)
            result.append(node.val)
            _inorder(node.right)

    _inorder(root)

    return result


def in_order_iteration(root: Node | None) -> list[str]:
    """Return Left, Root, Right order using a loop and stack(s)."""

    result = []
    stack = []
    current = root

    while current is not None or stack:

        # Travel as far left as possible
        while current is not None:
            stack.append(current)
            current = current.left

        # Process the leftmost node
        current = stack.pop()
        result.append(current.val)

        # Move to the right subtree
        current = current.right

    return result


def post_order_recursion(root: Node | None) -> list[str]:
    """Return Left, Right, Root order using recursion; None returns []."""

    result = []

    def _postorder(node):
        if node is not None:
            _postorder(node.left)
            _postorder(node.right)
            result.append(node.val)

    _postorder(root)

    return result


def post_order_iteration(root: Node | None) -> list[str]:
    """Return Left, Right, Root order using a loop and stack(s)."""

    if root is None:
        return []

    # stack1 produces Root, Right, Left; stack2 reverses it to Left, Right, Root
    stack1 = [root]
    stack2 = []

    while stack1:

        node = stack1.pop()
        stack2.append(node)

        if node.left is not None:
            stack1.append(node.left)

        if node.right is not None:
            stack1.append(node.right)

    result = []

    while stack2:
        node = stack2.pop()
        result.append(node.val)

    return result
