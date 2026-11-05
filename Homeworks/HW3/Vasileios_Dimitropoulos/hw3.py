"""HW 3: memoization with expiry and binary tree traversals."""

import functools
import time

# How long (in seconds) a cached result stays valid.
CACHE_LIFETIME = 3.0


# ---------------------------------------------------------------------------
# Part 1: Memoization with expiry
# ---------------------------------------------------------------------------

def memorize_3s(func):
    """Cache each argument's successful result for three seconds after completion.

    The wrapped function takes one positional, hashable argument. A cached
    result is reused while it is less than CACHE_LIFETIME seconds old, measured
    with time.monotonic() from when the call finished. Cache hits do not reset
    the expiry, and calls that raise an exception are not cached.
    """
    # Each decorated function gets its own cache, created once per decoration.
    # It maps an argument to a (result, time_the_result_was_computed) pair.
    cache = {}

    @functools.wraps(func)  # keep the original function's name and docstring
    def wrapper(arg):
        if arg in cache:
            result, computed_at = cache[arg]
            # A hit is valid only for strictly less than 3 seconds. We do not
            # update computed_at here, so cache hits never extend the expiry.
            if time.monotonic() - computed_at < CACHE_LIFETIME:
                return result

        # Cache miss or expired entry: call the real function. If it raises,
        # the exception propagates and we never reach the line that caches.
        result = func(arg)

        # Read the clock *after* the call, so expiry is measured from the
        # moment the computation finished (not when it started).
        cache[arg] = (result, time.monotonic())
        return result

    return wrapper


# ---------------------------------------------------------------------------
# Part 2: Binary tree traversal
# ---------------------------------------------------------------------------

class Node:
    """A binary-tree node; missing children are None. Leave this class unchanged."""

    def __init__(self, key: str, left=None, right=None):
        self.val = key
        self.left = left
        self.right = right


# --- Recursive versions ----------------------------------------------------
# Each function builds a fresh result list and fills it with a small nested
# helper. The three versions differ only in *when* the node's own value is
# appended relative to visiting its children.

def pre_order_recursion(root: Node | None) -> list[str]:
    """Return Root, Left, Right order using recursion; None returns []."""
    result = []

    def visit(node):
        if node is None:
            return
        result.append(node.val)  # Root
        visit(node.left)         # Left
        visit(node.right)        # Right

    visit(root)
    return result


def in_order_recursion(root: Node | None) -> list[str]:
    """Return Left, Root, Right order using recursion; None returns []."""
    result = []

    def visit(node):
        if node is None:
            return
        visit(node.left)         # Left
        result.append(node.val)  # Root
        visit(node.right)        # Right

    visit(root)
    return result


def post_order_recursion(root: Node | None) -> list[str]:
    """Return Left, Right, Root order using recursion; None returns []."""
    result = []

    def visit(node):
        if node is None:
            return
        visit(node.left)         # Left
        visit(node.right)        # Right
        result.append(node.val)  # Root

    visit(root)
    return result


# --- Iterative versions ----------------------------------------------------
# A Python list serves as the stack: append() pushes and pop() removes the
# most recently pushed item. The stack holds the nodes whose work is still
# pending, replacing the call stack the recursive versions rely on. Because
# it lives on the heap, deep trees do not hit Python's recursion limit.

def pre_order_iteration(root: Node | None) -> list[str]:
    """Return Root, Left, Right order using a loop and stack(s)."""
    result = []
    stack = [root] if root is not None else []

    while stack:
        node = stack.pop()
        result.append(node.val)  # visit the root first
        # Push right before left: the stack is last-in, first-out, so the
        # left child is popped (and its whole subtree finished) first.
        if node.right is not None:
            stack.append(node.right)
        if node.left is not None:
            stack.append(node.left)

    return result


def in_order_iteration(root: Node | None) -> list[str]:
    """Return Left, Root, Right order using a loop and stack(s)."""
    result = []
    stack = []      # ancestors whose left side we are still exploring
    current = root

    while current is not None or stack:
        # Walk as far left as possible, remembering each node on the way.
        while current is not None:
            stack.append(current)
            current = current.left

        # The top node has no unvisited left subtree, so it is next in order.
        node = stack.pop()
        result.append(node.val)

        # Then traverse its right subtree the same way.
        current = node.right

    return result


def post_order_iteration(root: Node | None) -> list[str]:
    """Return Left, Right, Root order using a loop and stack(s)."""
    result = []
    stack = []
    current = root
    last_visited = None  # the node most recently appended to result

    while current is not None or stack:
        # Walk as far left as possible, remembering each node on the way.
        while current is not None:
            stack.append(current)
            current = current.left

        # Look at (but do not yet remove) the deepest pending node.
        node = stack[-1]

        if node.right is not None and node.right is not last_visited:
            # Its right subtree has not been traversed yet: do that first.
            current = node.right
        else:
            # Both subtrees are done, so the node itself can be visited.
            stack.pop()
            result.append(node.val)
            last_visited = node

    return result
