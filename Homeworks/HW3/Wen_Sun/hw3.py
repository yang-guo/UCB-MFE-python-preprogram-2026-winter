"""HW 3 starter. Implement the seven functions, then run test_hw3.py."""

import time

def memorize_3s(func):
    """Cache each argument's successful result for three seconds after completion.

    The wrapped function takes one positional, hashable argument. Use
    time.monotonic(); cache hits do not reset expiry. See README.md for details.
    """
    cache = {}  # arg: (result, finished_at)

    def wrapper(arg):
        now = time.monotonic()
        if arg in cache:
            result, finished_at = cache[arg]
            #A cached result is valid
            if now - finished_at < 3:
                return result
        
        #call the function again
        result = func(arg)  # exceptions propagate before we cache anything
        cache[arg] = (result, time.monotonic())
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
    out: list[str] = []

    def walk(node: Node | None) -> None:
        # base case
        if node is None:
            return
        # root first, then recurse into the left and right subtrees.
        out.append(node.val)
        walk(node.left)
        walk(node.right)

    walk(root)
    return out

def pre_order_iteration(root: Node | None) -> list[str]:
    """Return Root, Left, Right order using a loop and stack(s)."""
    if root is None:
        return []

    out: list[str] = []
    stack: list[Node] = [root]

    while stack:
        # pop the next node and visit it immediately.
        node = stack.pop()
        out.append(node.val)

        # push right before left so the left child is popped first,
        # preserving the Root, Left, Right order.
        if node.right is not None:
            stack.append(node.right)
        if node.left is not None:
            stack.append(node.left)

    return out

def in_order_recursion(root: Node | None) -> list[str]:
    """Return Left, Root, Right order using recursion; None returns []."""
    out: list[str] = []

    def walk(node: Node | None) -> None:
        # base case
        if node is None:
            return
        # recurse left first, then record the root, then recurse right.
        walk(node.left)
        out.append(node.val)
        walk(node.right)

    walk(root)
    return out

def in_order_iteration(root: Node | None) -> list[str]:
    """Return Left, Root, Right order using a loop and stack(s)."""
    out: list[str] = []
    stack: list[Node] = []
    node = root

    # keep going while there is a current node to descend into
    # or pending nodes on the stack to visit.
    while node is not None or stack:
        # descend as far left as possible, remembering the path.
        while node is not None:
            stack.append(node)
            node = node.left

        # the leftmost pending node is the next one in order.
        node = stack.pop()
        out.append(node.val)

        # right subtree (if any) must be processed next.
        node = node.right

    return out

def post_order_recursion(root: Node | None) -> list[str]:
    """Return Left, Right, Root order using recursion; None returns []."""
    out: list[str] = []

    def walk(node: Node | None) -> None:
        # base case
        if node is None:
            return
        # recurse into both children before recording the root.
        walk(node.left)
        walk(node.right)
        out.append(node.val)

    walk(root)
    return out

def post_order_iteration(root: Node | None) -> list[str]:
    """Return Left, Right, Root order using a loop and stack(s)."""
    if root is None:
        return []

    out: list[str] = []
    stack: list[Node] = [root]

    # the mirror of pre-order
    while stack:
        node = stack.pop()
        out.append(node.val)

        # push left before right so the right child is popped first.
        if node.left is not None:
            stack.append(node.left)
        if node.right is not None:
            stack.append(node.right)

    # reversing Root, Right, Left yields Left, Right, Root.
    out.reverse()
    return out