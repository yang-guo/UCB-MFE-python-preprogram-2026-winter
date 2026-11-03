# HW3: Memoization and Tree Traversal

## Part 1: Memoization

My `memorize_3s` function uses a dictionary to save each input, its result, and the time the function finished. If the same input is used again within three seconds, it returns the saved result. If three seconds have passed, it deletes the old entry and calculates the result again. Each decorated function has its own dictionary. If a calculation fails, its result is not saved.

## Part 2: Binary Tree Traversal

I wrote pre-order, in-order, and post-order traversals using recursion and iteration. The recursive functions call themselves to visit the left and right sides of the tree in the correct order.

The iterative functions use a stack to keep track of nodes that still need to be visited. For pre-order, I add the right child before the left child so the left side is visited first. For in-order and post-order, I also use a `visited` list to know when a node is ready to be added to the results.

## Testing

Run the tests in the folder with `hw3.py` and `test_hw3.py`:

```bash
python test_hw3.py
```

All 16 provided tests passed when I tested my functions in Jupyter Notebook.
