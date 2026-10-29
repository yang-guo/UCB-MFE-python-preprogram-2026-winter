# CLAUDE.md

This file gives Claude Code context for Shravani Koli's Homework 1 submission.

## Assignment Overview

Homework 1 for the UCB MFE Python Pre-Program covers three topics:

1. **Git**: create a personal submission folder and a `hw_1_Shravani_Koli` branch, commit the work, and open a pull request into `main`. `git.txt` holds my name and favorite movie.
2. **uv package management**: set up a separate homework environment with Python 3.12.9, install the shared `requirements.txt` (`requests==2.25`), then do the required downgrade and upgrade. `uv.txt` records the commands I used and the latest `requests` version.
3. **Conway's Game of Life**: implement `evolve(initial_state)` in `game_of_life.py`. It takes a 2D grid of 0s (dead) and 1s (alive) and returns the next generation:
   - A live cell with 2 or 3 live neighbors survives.
   - A dead cell with exactly 3 live neighbors becomes alive.
   - Every other cell is dead in the next generation.

## Submission Files

All work lives in `Homeworks/HW1/Shravani_Koli/`:

- `git.txt`: my name and favorite movie
- `uv.txt`: package-management commands and the latest `requests` version
- `CLAUDE.md`: this file
- `game_of_life.py`: my `evolve` implementation with both supplied tests

## Rules

- **Only edit files inside my personal folder** (`Homeworks/HW1/Shravani_Koli/`). Do not modify any other files in the repository.
- **Preserve the original starter files.** The shared `Homeworks/HW1/game_of_life.py`, `requirements.txt`, and `README.md` must stay unchanged; my implementation goes in my own copy of `game_of_life.py`.
- **Keep the supplied test cases unchanged** and make sure both pass:
  - `test_case_1` (a 2x2 block) must stay the same after one step.
  - `test_case_2` (a vertical blinker) must become the horizontal blinker `test_case_2_next`.
- Do not commit the local Python environment (`.venv/` or similar).

## Verifying

With the homework environment activated, run:

```bash
python game_of_life.py
```

If it exits with no output and no `AssertionError`, both tests passed.
