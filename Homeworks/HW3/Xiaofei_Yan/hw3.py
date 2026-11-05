"""HW 3 starter. Implement the seven functions, then run test_hw3.py."""

import time


def memorize_3s(func):
    """Cache each argument's successful result for three seconds after completion.

    The wrapped function takes one positional, hashable argument. Use
    time.monotonic(); cache hits do not reset expiry. See README.md for details.
    """
    cache = {} # cache dictionary to save arg: result, time

    def wrapper(arg):

        # If saved and last called within 3 seconds
        if arg in cache and (time.monotonic() - cache[arg][1]) < 3:
            result = cache[arg][0]

        # If not
        else:
            result = func(arg)
            cache[arg] = [result, time.monotonic()]

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

    if root is not None:
        result.append(root.val)
        result.extend(pre_order_recursion(root.left))
        result.extend(pre_order_recursion(root.right))

    return result


def pre_order_iteration(root: Node | None) -> list[str]:
    """Return Root, Left, Right order using a loop and stack(s)."""
    
    result = [] # To save node val by order
    stack = [] # Act as a stack to be processed. Upper layer is processed first
    stack.append(root)

    while stack != []: # When there is still a quene
        current_node = stack.pop() # the upper layer node to be added and splitted

        if current_node is not None:
            result.append(current_node.val)
            stack.append(current_node.right) # covered by left
            stack.append(current_node.left) # covering right

    return result


def in_order_recursion(root: Node | None) -> list[str]:
    """Return Left, Root, Right order using recursion; None returns []."""
    
    result = []

    if root is not None:
        result.extend(in_order_recursion(root.left))
        result.append(root.val)
        result.extend(in_order_recursion(root.right))

    return result


def in_order_iteration(root: Node | None) -> list[str]:
    """Return Left, Root, Right order using a loop and stack(s)."""
    result = [] # To save node val by order
    stack = [] # Act as a stack to be processed. Upper layer is processed first
    current_node = root

    while current_node is not None or stack != []:
        if current_node is not None: # keep going to left
            stack.append(current_node)
            current_node = current_node.left

        else: # when at left end
            # pop parent to stack right
            p = stack.pop()
            result.append(p.val)
            current_node = p.right 

    return result



def post_order_recursion(root: Node | None) -> list[str]:
    """Return Left, Right, Root order using recursion; None returns []."""
        
    result = []

    if root is not None:
        result.extend(post_order_recursion(root.left))
        result.extend(post_order_recursion(root.right))
        result.append(root.val)

    return result


def post_order_iteration(root: Node | None) -> list[str]:
    """Return Left, Right, Root order using a loop and stack(s)."""
    result = []

    if root is None:
        return result
    
    stack_1 = [root] # to be processed
    stack_2 = [] # reverse order of result

    # Fill up stack_2
    while stack_1 != []:
        current_node = stack_1.pop()
        stack_2.append(current_node)

        # put its children into stack_1
        if current_node.left is not None:
            stack_1.append(current_node.left)  # left will go to stack 2 later and result first

        if current_node.right is not None:
            stack_1.append(current_node.right) 

    # Pour from stack_2 to result
    while stack_2 != []:
        result.append(stack_2.pop().val)

    return result