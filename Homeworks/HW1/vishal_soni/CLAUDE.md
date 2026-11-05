# CLAUDE.md

Context for Claude Code when working in this folder.

## Submission

This is Vishal Soni's Homework 1 submission for the UCB MFE Python Pre-Program (Fall 2026).
It is due Monday, October 19, 2026, at 11:59 p.m. PT. The full assignment is in `../../../Lectures/Lecture 1/hw1.md`.

The homework covers Git, Claude Code, uv package management with Python 3.12.9, and Conway's Game of Life.

## Files

- `git.txt`: my name and favorite movie.
- `uv.txt`: the uv commands used to create the `HW1` environment (Python 3.12.9), install `requests==2.25`, downgrade, then upgrade to the latest version.
- `CLAUDE.md`: this file. It also serves as proof that Claude Code is installed.
- `game_of_life.py`: my implementation of `evolve` for Conway's Game of Life.

## Game of Life notes

- `evolve(initial_state)` takes a 2D list of 0s and 1s and returns the next generation as a new list.
- Rules: a live cell with 2 or 3 live neighbors survives. A dead cell with exactly 3 live neighbors becomes alive. All other cells are dead.
- Cells outside the grid count as dead. The grid does not wrap around.
- Do not mutate the input grid.
- Keep the two supplied assertions (block still life and blinker) at the bottom of the file.

## Running

```bash
source HW1/bin/activate   # or .venv
python game_of_life.py    # no output means both assertions passed
```

## Conventions

- Only files in this folder belong in the submission. Do not edit the shared starter files in `Homeworks/HW1/`.
- The local environment (`HW1/` or `.venv/`) is not committed.
- Branch: `hw_1_vishal_soni`. Open the PR into `main` and assign `kevinramlal` and `elfiasco`.
- Keep code clear and commented, since this is a learning exercise.
