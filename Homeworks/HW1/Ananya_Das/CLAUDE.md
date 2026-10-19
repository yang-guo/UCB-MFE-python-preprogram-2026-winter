# HW1 — Ananya Das

Homework 1 submission for the UC Berkeley MFE Python Pre-Program (Winter 2026).
It covers Git, Claude Code, uv package management, and Conway's Game of Life.

## Files

- `git.txt`: name and favorite movie (`Ananya Das, The Perks of Being a Wallflower`).
- `uv.txt`: uv commands for the package-management exercise.
- `CLAUDE.md`: this file, which shows Claude Code is set up and gives it context.
- `game_of_life.py`: `evolve` implementation with the two supplied test cases.

## Git

- Branch: `hw_1_Ananya_Das`.
- Only this folder is changed. Shared files in `Homeworks/HW1/` are untouched.

## uv

- Created a virtual environment `HW1` with Python 3.12.9 in this folder.
- Installed `../requirements.txt` (`requests==2.25`).
- Downgraded `requests` to 2.24.0, then upgraded it to the latest: **2.34.2**.
- The `HW1/` environment is git-ignored and is not part of the submission.

## Game of Life

`evolve(initial_state)` returns the next generation as a new board:

- Each cell counts its 8 neighbors. Cells outside the board count as dead.
- A live cell survives with 2 or 3 live neighbors. Otherwise it dies.
- A dead cell becomes alive with exactly 3 live neighbors.
- The input board is not modified.

Test cases:
- `test_case_1` (block): a still life, so the board stays the same.
- `test_case_2` (blinker): a vertical line of three becomes a horizontal one.

Run from this folder (no output means both assertions passed):

```bash
source HW1/bin/activate
python game_of_life.py
```

## Notes for Claude Code

- Keep changes inside `Homeworks/HW1/Ananya_Das/`.
- Never commit the `HW1/` environment.
- Keep both supplied test cases unchanged.
- Ask the author before staging, committing, pushing, or opening a PR.
