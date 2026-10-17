# HW1 Submission - Gobind Singh

Homework 1 for the UCB MFE Python pre-program (Winter 2026).

## Files

- `git.txt`: my name and favorite movie (Gobind Singh, Inception).
- `uv.txt`: the uv commands I ran in this folder. They create the `HW1` virtual environment with Python 3.12.9, install `../requirements.txt`, downgrade `requests` to 2.24.0, then upgrade it to the latest version (2.34.2).
- `game_of_life.py`: my implementation of `evolve` for Conway's Game of Life. It counts the live neighbors of each cell first, then applies the rules to the board:
  - A live cell with fewer than 2 or more than 3 live neighbors dies.
  - A dead cell with exactly 3 live neighbors becomes alive.
  - Every other cell keeps its state.

## Notes

- Run the tests from this folder with the `HW1` environment activated: `python game_of_life.py`. Both supplied assertions pass.
- The `HW1/` virtual environment is local only and is not part of the submission.
