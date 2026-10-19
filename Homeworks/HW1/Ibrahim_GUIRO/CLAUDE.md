## HW1 Submission Context

I am Ibrahim GUIRO, a student in the UCB MFE Python Pre-Program 2026. This folder contains my Homework 1 submission for the UCB MFE Python Pre-Program.

## contents

The homework covers:

1. **`git.txt`**: my name and favorite movie (Git exercise: branch, commit, pull request).
2. **`uv.txt`**: the uv commands I used to create the `HW1` environment and to install, downgrade and upgrade `requests`.
3. **`game_of_life.py`**: my implementation of the `evolve` function of Conway's Game of Life.

## Notes

- `HW1/` is my local uv virtual environment (Python 3.12.9). It is gitignored and must not be committed.
- `game_of_life.py` implements `evolve`. Cells outside the board count as dead. The function returns a new grid and does not change the input.
- To run the tests: `source HW1/bin/activate && python game_of_life.py`. No output means both assertions passed.
