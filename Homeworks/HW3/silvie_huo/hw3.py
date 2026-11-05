"""HW 3 starter. Implement the seven functions, then run test_hw3.py."""

import time


def memorize_3s(func):
    """Cache each argument's successful result for three seconds after completion.

    The wrapped function takes one positional, hashable argument. Use
    time.monotonic(); cache hits do not reset expiry. See README.md for details.
    """

    cache = {}                           # arg -> (result, stored_at)

    def wrapper(x):
        # 1. If x is cached, read the clock and compute its age.
        #    Hit only if age < 3: return the stored result.
        #    (Do NOT update the timestamp on a hit.)
        if x in cache:
            (result, stored_at) = cache[x]
            if time.monotonic() - stored_at < 3:
                return result
        
        # 2. Otherwise (missing OR expired): call func(x).    
        result = func(x)

        # 3. After it returns, read the clock and store (result, now).
        cache[x] = (result,time.monotonic())

        # 4. Return the result.
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
    raise NotImplementedError("Implement pre_order_recursion")


def pre_order_iteration(root: Node | None) -> list[str]:
    """Return Root, Left, Right order using a loop and stack(s)."""
    raise NotImplementedError("Implement pre_order_iteration")


def in_order_recursion(root: Node | None) -> list[str]:
    """Return Left, Root, Right order using recursion; None returns []."""
    raise NotImplementedError("Implement in_order_recursion")


def in_order_iteration(root: Node | None) -> list[str]:
    """Return Left, Root, Right order using a loop and stack(s)."""
    raise NotImplementedError("Implement in_order_iteration")


def post_order_recursion(root: Node | None) -> list[str]:
    """Return Left, Right, Root order using recursion; None returns []."""
    raise NotImplementedError("Implement post_order_recursion")


def post_order_iteration(root: Node | None) -> list[str]:
    """Return Left, Right, Root order using a loop and stack(s)."""
    raise NotImplementedError("Implement post_order_iteration")
