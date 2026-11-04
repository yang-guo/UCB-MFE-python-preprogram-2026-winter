"""HW 3 starter. Implement the seven functions, then run test_hw3.py."""

import time


def memorize_3s(func):
    """
    Decorator on an unknown function func.  Keep a cache of previous copies.  When func
    is called, we can return the cached value if less than 3 seconds old, otherwise we
    need to call func to get fresh value
    """
    
    cache = {}
    def wrapper(arg):
        # Calculate the start time
        start_time = time.monotonic()
        
        if arg in cache:
            # We have a cached copy, don't know if it is fresh
            cached_result, timestamp = cache[arg]
            if start_time - timestamp < 3:
                # It is fresh, we can use it
                return cached_result
    
        # Failed to find a fresh value; compute it and store for next time
        result = func(arg)
        cache[arg] = (result, time.monotonic())

        return result

    return wrapper


class Node:
    """A binary-tree node; missing children are None. Leave this class unchanged."""

    def __init__(self, key: str, left=None, right=None):
        self.val = key
        self.left = left
        self.right = right


#def pre_order_recursion(root: Node | None) -> list[str]:
def pre_order_recursion(root):
    """Return Root, Left, Right order using recursion; None returns []."""
    if root is None:
        return []
    return [root.val] + pre_order_recursion(root.left) + pre_order_recursion(root.right) 


#def pre_order_iteration(root: Node | None) -> list[str]:
def pre_order_iteration(root):
    """Return Root, Left, Right order using a loop and stack(s)."""
    if root is None:
        return []
    result = []
    stack = [root]

    while len(stack) > 0:
        current = stack.pop()
        # Add the parent before it's children
        result.append(current.val)

        # Push the right first (processed later) and then left (processed first)
        if current.right is not None:
            stack.append(current.right)
        if current.left is not None:
            stack.append(current.left) 

    return result


#def in_order_recursion(root: Node | None) -> list[str]:
def in_order_recursion(root):
    """Return Left, Root, Right order using recursion; None returns []."""
    if root is None:
        return []
    return in_order_recursion(root.left) + [root.val] + in_order_recursion(root.right) 


#def in_order_iteration(root: Node | None) -> list[str]:
def in_order_iteration(root):
    """Return Left, Root, Right order using a loop and stack(s)."""
    result = []
    stack = []
    current = root

    # While we have a current node or stack to work with
    while current is not None or len(stack) > 0:
        # Traverse left from the current node until we hit the end
        while current is not None:
            stack.append(current)
            current = current.left
        # Left is exhausted, add myself to the list
        current = stack.pop()
        result.append(current.val)
        # And then go right
        current = current.right

    return result


#def post_order_recursion(root: Node | None) -> list[str]:
def post_order_recursion(root):
    """Return Left, Right, Root order using recursion; None returns []."""
    if root is None:
        return []
    return post_order_recursion(root.left) + post_order_recursion(root.right) + [root.val] 

#def post_order_iteration(root: Node | None) -> list[str]:
def post_order_iteration(root):
    """Return Left, Right, Root order using a loop and stack(s)."""

    if root is None:
        return []
    stack = [root]
    result = []

    while len(stack) > 0:
        current = stack.pop()
        # Add nodes as we process them
        result.append(current.val)
        # We will process the left tree later
        if current.left is not None:
            stack.append(current.left)
        # And the right tree first, because it's post-order
        if current.right is not None:
            stack.append(current.right)

    # Result is backwards because we appended parents before children
    return result[::-1]

