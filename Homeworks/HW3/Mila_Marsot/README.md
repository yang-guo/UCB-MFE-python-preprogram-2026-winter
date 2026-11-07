# HW3: Mila Marsot

## How to run the tests :
Go to the `Homeworks/HW3/Mila_Marsot` folder and run: `python test_hw3.py`

## Part 1: Cache decorator (`memorize_3s`)
### When does a cached result expire?
For each input, the cache saves the result and the time.
So we do not compute the same thing again if it is still new.

The time is saved after the function ends. If the result is less than 3 seconds old, we use it again. At 3 seconds or more, it is too old, so, we call the function again. Using the cache does not change the saved time.

## Part 2: Binary tree traversal
### Recursive versions
If the node is `None`, return `[]`. Otherwise, join the lists of the node and its two children in the right order.

### Iterative versions : we need to keep track of what is left to do
Instead of recursion, I use a list as a stack: `append` adds a node on top, and `pop` removes the node on top. The stack holds the nodes I still need to visit.

- **Pre-order:** I pop a node and add it to the result. Then I push its right child first and its left child second, so the left child is visited next.

- **In-order:** I go down to the left as far as I can, pushing each node on the way, because a node must wait until its left side is done. When I can't go further left, I pop a node, add it to the result, and move to its right child.

- **Post-order:** each node is pushed twice, with a `done` flag. The first time, I push it back with `done = True` and then push its children on top, so they are handled before it. The second time, its children are already in the result, so I add the node itself.