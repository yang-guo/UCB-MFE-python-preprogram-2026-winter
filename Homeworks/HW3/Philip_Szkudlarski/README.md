# Homework 3 – Advanced Python

**Philip Szkudlarski**

**Topics:** Decorators, Memoization, Recursion, and Stack-Based Algorithms

## How to run

From this folder (`Homeworks/HW3/Philip_Szkudlarski/`):

```bash
python3 test_hw3.py
```

On systems where Python 3 is named `python`, use `python test_hw3.py`. When all tests pass, the output ends with `OK`.

## Overview

This assignment covers two main topics:

1. Memoization with expiration using Python decorators.
2. Binary tree traversal using both recursive and iterative approaches.

The goal was to implement each function correctly while understanding how decorators, dictionaries, recursion, and stacks can be used to solve computational problems.

---

## Part 1: Memoization with Expiration

### 1. What is memoization?

Memoization is an optimization technique that stores the results of previously executed function calls. When the same input is passed to the function again, the program can return the stored result instead of recalculating it.

This is particularly useful when a function is computationally expensive or when the same inputs occur repeatedly.

For this assignment, I implemented a decorator called `memorize_3s` that caches results for three seconds.

### 2. How the implementation works

The implementation uses a Python dictionary to store cached results:

- **Key:** the function argument.
- **Value:** a tuple of `(expiration timestamp, result)`.

The decorator works as follows:

1. When the decorated function is called, it checks whether the argument already exists in the cache.
2. If the argument exists, it compares the current time against the stored expiration timestamp.
3. If fewer than three seconds have passed, the cached result is returned.
4. If three seconds or more have passed, the original function executes again.
5. After successful execution, the new result is stored with an expiration timestamp three seconds after completion.

A cache hit returns the stored result without updating the timestamp, so repeated calls do not extend the expiry.

The implementation uses `time.monotonic()` rather than `time.time()` because monotonic time measures elapsed time without being affected by system clock adjustments.

### 3. Why does the expiration time start after function execution?

The assignment specifies that a cached result must remain valid for three seconds after the underlying function successfully returns.

For example, suppose a function starts at time 100 and takes two seconds to execute.

- Function starts: 100
- Function finishes: 102
- Cache expires: 105

A call at 104.9 uses the cached result, while a call at 105 must execute the function again.

This is why the implementation calls `time.monotonic()` a second time, after the original function finishes, when computing the expiration timestamp.

### 4. Why use a dictionary?

A dictionary allows constant-time average lookup, insertion, and retrieval.

It also allows each argument to have its own independent expiration timestamp.

For example, caching the result of `f(1)` does not affect the expiration time of `f(2)`.

Each decorated function also receives its own cache because the dictionary is created inside the decorator, once per decorated function.

### 5. Handling edge cases

The implementation accounts for the following requirements:

- **Exact expiration boundary:** A result expires when the elapsed time reaches exactly three seconds, because the check is `now < expiry` (strictly less than).
- **Falsey values:** Results such as `0`, `False`, `None`, and empty lists are cached correctly because the implementation checks dictionary membership (`arg in cache`) rather than whether the result is truthy.
- **Failed function calls:** If the original function raises an exception, the exception propagates before the cache is updated, so no new cache entry is created.
- **Independent arguments:** Each argument has its own result and expiration timestamp.
- **Independent functions:** Each decorated function maintains a separate cache.
- **Slow computations:** The expiration timer starts after successful execution, not before.

---

## Part 2: Binary Tree Traversal

### 1. What is a binary tree?

A binary tree is a data structure in which each node can have at most two children:

- A left child.
- A right child.

In this assignment, the supplied `Node` class contains three attributes:

- `val`: The value stored in the node.
- `left`: Reference to the left child.
- `right`: Reference to the right child.

A missing child, or an empty tree, is represented by `None`.

The assignment requires three different traversal methods, each implemented recursively and iteratively.

### 2. Example binary tree

The assignment provides the following tree:

```text
        F
       / \
      B   G
     / \   \
    A   D   I
       / \ /
      C  E H
```

Every traversal visits each node exactly once; the traversal method determines the order.

### 3. Preorder Traversal

**Order: Root → Left → Right**

Preorder traversal visits the current node first, followed by its entire left subtree, and then its right subtree.

#### Recursive implementation

The recursive implementation follows three steps:

1. Append the current node's value to the result list.
2. Recursively traverse the left child.
3. Recursively traverse the right child.

A base case checks whether the current node is `None`. If so, the function returns without doing anything. This stops the recursion when it reaches a missing child.

#### Iterative implementation

The iterative implementation uses a stack. A stack follows the Last In, First Out (LIFO) principle: the most recently added element is the first one removed.

The stack holds the nodes that are still waiting to be visited. The algorithm works as follows:

1. Push the root onto the stack.
2. Pop the top node and append its value to the result.
3. Push its right child onto the stack (if it exists).
4. Push its left child onto the stack (if it exists).
5. Repeat until the stack is empty.

**Why push the right child first?**

Because a stack is LIFO, the left child must be pushed last so it gets processed first. This produces the required Root → Left → Right order.

#### Result

```text
F, B, A, D, C, E, G, I, H
```

---

### 4. Inorder Traversal

**Order: Left → Root → Right**

Inorder traversal visits the entire left subtree first, then the current node, and finally the right subtree.

#### Recursive implementation

The recursive implementation follows three steps:

1. Recursively traverse the left child.
2. Append the current node's value to the result list.
3. Recursively traverse the right child.

The base case returns when the current node is `None`.

#### Iterative implementation

The iterative implementation uses a stack and a pointer called `current`.

The algorithm works as follows:

1. Start at the root.
2. Continue moving left, pushing each node onto the stack.
3. Once there are no more left children, pop the top node.
4. Append its value to the result.
5. Move to its right child.
6. Repeat until both the current node is `None` and the stack is empty.

**Why do we keep moving left before processing a node?**

Because inorder traversal requires the left subtree to be processed before the current node.

The stack stores nodes that have been encountered but cannot yet be processed because their left subtrees must be visited first.

#### Result

```text
A, B, C, D, E, F, G, H, I
```

Inorder traversal produces sorted values for a binary search tree, but the trees in this assignment are not necessarily binary search trees. Therefore, the algorithm follows the tree structure rather than sorting the values.

---

### 5. Postorder Traversal

**Order: Left → Right → Root**

Postorder traversal visits both children before visiting the current node.

#### Recursive implementation

The recursive implementation follows three steps:

1. Recursively traverse the left child.
2. Recursively traverse the right child.
3. Append the current node's value to the result list.

The base case returns when the current node is `None`.

**Why is the root visited last?**

Postorder traversal processes the children before their parent. This is useful when deleting a tree or calculating the total size of a directory structure.

#### Iterative implementation

The iterative implementation uses two stacks.

The first stack holds the nodes still waiting to be processed, while the second stack collects processed nodes so their order can be reversed.

The algorithm works as follows:

1. Push the root onto the first stack.
2. Pop a node from the first stack and push it onto the second stack.
3. Push its left child, then its right child, onto the first stack (if they exist).
4. Continue until the first stack is empty.
5. Pop all nodes from the second stack and append their values to the result list.

**Why use two stacks?**

Because the right child is pushed last, it is popped first, so nodes leave the first stack in Root → Right → Left order.

The second stack reverses this order, resulting in Left → Right → Root.

This allows postorder traversal to be implemented iteratively without recursive function calls.

#### Result

```text
A, C, E, D, B, H, I, G, F
```

---

### 6. Comparison of the Six Traversal Methods

| Function | Approach | Visit order |
|---|---|---|
| `pre_order_recursion` | Recursion | Root, Left, Right |
| `pre_order_iteration` | One stack | Root, Left, Right |
| `in_order_recursion` | Recursion | Left, Root, Right |
| `in_order_iteration` | One stack | Left, Root, Right |
| `post_order_recursion` | Recursion | Left, Right, Root |
| `post_order_iteration` | Two stacks | Left, Right, Root |

### 7. Time and Space Complexity

Let n be the total number of nodes in the tree and h its height.

All six traversal algorithms run in **O(n)** time, because every node is visited exactly once.

Auxiliary space (not counting the returned list):

- **Recursive versions:** O(h), used by the function call stack.
- **Iterative preorder and inorder:** O(h), used by the explicit stack.
- **Iterative postorder:** O(n), because the second stack eventually holds every node before reversing them.

In the worst case, a highly unbalanced tree has height O(n). A balanced tree has height O(log n).

An important advantage of the iterative implementations is that they do not depend on Python's recursion limit. They can therefore handle trees that are deeper than the normal recursion limit.

---

## Testing and Verification

The supplied `test_hw3.py` file uses Python's standard-library `unittest` framework to verify the implementations. See [How to run](#how-to-run) for the command.

The tests cover:

### Memoization
- Cache hits and expiration.
- Expiration at exactly three seconds.
- Independent expiration for different arguments.
- Separate caches for different decorated functions.
- Falsey return values.
- Failed function calls.
- Functions that take time to execute.

### Binary tree traversal
- The example tree provided in the assignment.
- Empty trees.
- Single-node trees.
- Trees with only left or right children.
- Trees with unsorted and duplicate values.
- Repeated function calls.
- Ensuring the original tree remains unchanged.
- Deep trees that exceed Python's recursion limit (iterative implementations).

---

## Conclusion

This assignment demonstrates how decorators can modify existing function behavior without changing the original function, and how memoization can improve efficiency by avoiding repeated calculations.

It also demonstrates two ways of traversing binary trees. Recursive implementations use function calls to move through the tree, while iterative implementations use explicit stacks to keep track of nodes that still need to be processed.

The main distinction between preorder, inorder, and postorder traversal is the position at which the root is visited relative to its children.

All seven functions were implemented to satisfy the assignment's requirements while maintaining readable code and handling the specified edge cases.
