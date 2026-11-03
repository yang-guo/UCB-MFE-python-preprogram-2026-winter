"""HW 3 starter. Implement the seven functions, then run test_hw3.py."""

import time




def memorize_3s(func):
    """Cache each argument's successful result for three seconds after completion.

    The wrapped function takes one positional, hashable argument. Use
    time.monotonic(); cache hits do not reset expiry. See README.md for details.
    """
    # raise NotImplementedError("Implement memorize_3s")
    cache={func:{}}
    
    def wrapper(arg):
        if arg in cache[func]:
            temp=cache[func][arg]
            if time.monotonic()-temp[1]<3:
                return temp[0]
        res=func(arg)
        cache[func][arg]=(res,time.monotonic())
        return res
    
    return wrapper


class Node:
    """A binary-tree node; missing children are None. Leave this class unchanged."""

    def __init__(self, key: str, left=None, right=None):
        self.val = key
        self.left = left
        self.right = right


def pre_order_recursion(root: Node | None) -> list[str]:
    """Return Root, Left, Right order using recursion; None returns []."""
    # raise NotImplementedError("Implement pre_order_recursion")
    lst=[]
    if root is None:
        return lst
    lst.append(root.val)
    lst.extend(pre_order_recursion(root.left))
    lst.extend(pre_order_recursion(root.right))
    return lst


def pre_order_iteration(root: Node | None) -> list[str]:
    """Return Root, Left, Right order using a loop and stack(s)."""
    # raise NotImplementedError("Implement pre_order_iteration")
    res=[]
    st=[]
    if root is None:
        return res
    st.append(root)
    while len(st)!=0:
        node=st.pop()
        res.append(node.val)
        if node.right is not None:
            st.append(node.right)
        if node.left is not None:
            st.append(node.left)
    return res


def in_order_recursion(root: Node | None) -> list[str]:
    """Return Left, Root, Right order using recursion; None returns []."""
    # raise NotImplementedError("Implement in_order_recursion")
    lst=[]
    if root is None:
        return lst
    lst.extend(in_order_recursion(root.left))
    lst.append(root.val)
    lst.extend(in_order_recursion(root.right))
    return lst


def in_order_iteration(root: Node | None) -> list[str]:
    """Return Left, Root, Right order using a loop and stack(s)."""
    # raise NotImplementedError("Implement in_order_iteration")
    res=[]
    st=[]
    current=root
    while current is not None or len(st)!=0:
        while current is not None:
            st.append(current)
            current=current.left
        current=st.pop()
        res.append(current.val)
        current=current.right
    return res


def post_order_recursion(root: Node | None) -> list[str]:
    """Return Left, Right, Root order using recursion; None returns []."""
    # raise NotImplementedError("Implement post_order_recursion")
    lst=[]
    if root is None:
        return lst
    lst.extend(post_order_recursion(root.left))
    lst.extend(post_order_recursion(root.right))
    lst.append(root.val)
    return lst


def post_order_iteration(root: Node | None) -> list[str]:
    """Return Left, Right, Root order using a loop and stack(s)."""
    # raise NotImplementedError("Implement post_order_iteration")
    if root is None:
        return []
    res=[]
    st=[root]
    while len(st)!=0:
        node=st.pop()
        res.append(node.val)
        if node.left is not None:
            st.append(node.left)
        if node.right is not None:
            st.append(node.right) 
    res.reverse()   
    return res
