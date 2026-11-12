# CLAUDE.md

This folder is Min Jin's Homework 1 submission for the UCB MFE Python Pre-Program (Fall 2026).
The full assignment is in `Lectures/Lecture 1/hw1.md`; the shared starter files are in the parent `Homeworks/HW1/` folder.

**Due:** Monday, October 19, 2026, 11:59 p.m. PT
**Branch:** `hw_1_Min_Jin` (PR into `main`, assigned to `kevinramlal` and `elfiasco`)

## Files

| File | Purpose |
| --- | --- |
| `git.txt` | Name and favorite movie, in the format `<FIRST_NAME> <LAST_NAME>, <YOUR_FAV_MOVIE>` |
| `uv.txt` | uv commands to create a Python 3.12.9 virtual environment named `HW1`, install `../requirements.txt`, downgrade `requests` to 2.24.0, upgrade it to the latest release, and the resulting latest version |
| `CLAUDE.md` | This file: a description of the submission and proof that Claude Code is installed and initialized |
| `game_of_life.py` | Implementation of `evolve(initial_state)` for Conway's Game of Life, with both supplied test assertions kept |

## Game of Life notes

- `evolve` takes a board (a list of lists of `0`/`1`) and returns the next generation as a new board.
- Rules: a live cell with 2 or 3 live neighbors survives; a dead cell with exactly 3 live neighbors becomes alive; every other cell is dead in the next generation.
- Each cell has 8 neighbors (horizontal, vertical, diagonal). Cells outside the board count as dead; the board does not wrap around.
- All cells update at the same time, based on the original board, so the input is not changed in place.
- Test 1 (2x2 block) is a still life and should be unchanged; test 2 (vertical blinker) should turn into a horizontal blinker.

Run the tests from this folder with the `HW1` environment activated:

```bash
source HW1/bin/activate
python game_of_life.py   # no output means both assertions passed
```

## Constraints

- Only files in this folder are part of the submission. Do not modify the shared starter files or other students' folders.
- The `HW1/` virtual environment (or `.venv/`) stays local and must not be committed.
- Do not commit `.env` files or credentials.
