# HW3: Advanced Python

Author: Ziang Gao

## How to run the tests

Open a terminal in this folder. Then run:

```
python test_hw3.py
```

The tests are in `test_hw3.py`. I did not change this file. All tests must show `OK`.

## Part 1: `memorize_3s`

### How it works

`memorize_3s` is a decorator. It saves the result of a function for 3 seconds.

- The decorator makes one dictionary, `cache`, when it wraps a function.
- The inner function `wrapper` is a closure. It can read `cache` on every call.
- Each key in `cache` is an argument. Each value is a tuple: `(result, expiry_time)`.
- On each call, `wrapper` does these steps:
  1. It reads the clock with `time.monotonic()`.
  2. If the argument is in `cache` and the clock is less than `expiry_time`, it returns the saved result. The wrapped function does not run.
  3. Otherwise, it runs the wrapped function.
  4. It reads the clock again. It saves the result with `expiry_time = finish time + 3`.
  5. It returns the result.

### When cache entries expire

- The 3 seconds start when the wrapped function **finishes**. They do not start when the function starts. A slow function does not use up its own 3 seconds.
- An entry is valid while `clock < finish time + 3`.
- At exactly 3 seconds, the entry is expired. The function runs again.
- A cache hit does **not** make the entry last longer. The expiry time stays the same.
- Each argument has its own expiry time.

### Other design choices

- I use `arg in cache` to check for a saved result. I do not check if the result is true or false. So results such as `0`, `None`, and `""` are also saved.
- If the wrapped function raises an error, the error goes to the caller. The line that saves the result comes after `func(arg)`, so a failed call is never saved.
- I call `time.monotonic()` through the `time` module. The tests can then replace the clock. `time.monotonic()` is also not changed when the computer clock changes.
