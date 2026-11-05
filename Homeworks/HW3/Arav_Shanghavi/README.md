# HW3: Arav Shanghavi

Memoization with a three second expiry, plus pre, in and post order traversals of a binary tree, each written once with recursion and once with a loop.

## How to run

From this folder:

```bash
python test_hw3.py
```

It only needs the standard library. Python 3.12 or newer. The last line should read `OK` after 16 tests.

## When cache entries expire

`memorize_3s` keeps a dictionary called `cache_table`, created fresh each time the decorator is applied. So two decorated functions never share entries. Each key is the argument. Each value is a pair: what the function gave back, plus a `time.monotonic()` stamp I take once the call has completed.

On a call I read the clock first. If the argument is already a key and less than 3 seconds have gone by since its stamp, I hand back the saved value and the real function is skipped. At 3.0 seconds or later the entry counts as stale. I call the real function once more and overwrite the pair with a fresh value and a fresh stamp.

A few details matter here.

- The stamp comes from after the function finishes, not from when the call started. A slow function still gets its full 3 seconds.
- Returning a saved value never touches the stamp. Hits don't buy extra time.
- I test whether the key exists, not whether the saved value looks true. So `0`, `None`, `False`, `""` and `[]` are cached like anything else.
- If the function raises, the line that writes to `cache_table` is never reached. The error reaches whoever called it, and `cache_table` stays as it was.

Think of it like a stock quote. Fine to reuse for a moment, then you go and fetch a new one.

## How the iterative traversals track pending work

All three keep a plain Python list as the stack. `append` adds to its end and `pop` removes from that same end. None of them recurse, so a tree thousands of levels deep is fine. An empty tree returns `[]` before any loop starts.

**Pre-order.** At the start the only thing on the stack is `root`. Each pass pops one node and records it straight away. Then it pushes that node's right child and after that its left child, skipping any that are missing. Left was pushed last, so it is the next one off. The stack holds every subtree I have found but not yet started, and the loop ends when it runs dry.

**In-order.** A node can't be recorded until everything to its left is done. So I follow `left` links downward and drop every node I pass onto the stack, until I fall off the bottom. Then I pop. That node has no unfinished left side, so I record it and jump to its right child to repeat the same slide. The stack is the list of ancestors still waiting for their turn. The loop keeps going while there is either a node to slide down from or something left on the stack.

**Post-order.** I use two stacks. The first behaves much like pre-order, except whatever I pop gets parked on the second stack rather than written to the output, and left children are pushed before right ones. That fills the second stack in Root, Right, Left order. Emptying the second stack from the top flips that around into Left, Right, Root. That is exactly post-order. Here the first stack is work still to explore and the second is finished nodes waiting to be written out.

## Recursive versions

Each one returns `[]` for a missing node. Otherwise it joins three lists in the order the traversal needs: the node's own value in a one item list, and whatever it gets back from calling itself on `left` and on `right`. Nothing is stored between calls, so every call returns a brand new list.
