### Part 1
- First check if input has been cached in dictionary key
- If yes, compare check time with saved time, if time >= 3.0 seconds, continue with loop
    - The new results overwrite an existing one if exists
- If time < 3.0 seconds, return cached value 
- Save the time after function run completes
- Then cache the time and result as a tuple in the dictionary
- Passing different functions for the wrapper create separate dictionaries/cache

### Part 2
- Recursion: match return order with traversal pattern
- Iterative Pre-Order: push right first so left child pops first
- Iterative In-Order: use inner loop to access bottom left, then store nodes on a stack, pop the root of each layer before moving to right subtrees
- Iterative Post-order: run pre-prder but reversing right left access order, return reversal of list