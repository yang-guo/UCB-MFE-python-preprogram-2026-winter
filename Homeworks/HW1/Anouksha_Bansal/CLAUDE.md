# CLAUDE.md

Context for Claude Code when working in this folder.

## About this submission

This is my Homework 1 submission for the UCB MFE Python Pre-Program (Fall 2026).

## Files

| File | Purpose |
| --- | --- |
| `git.txt` | My name and favorite movie |
| `uv.txt` | The `uv` commands I used to create the `HW1` environment (Python 3.12.9), install `requests==2.25`, downgrade and upgrade it, plus the latest `requests` version |
| `CLAUDE.md` | This file: submission description and proof that Claude Code is set up |
| `game_of_life.py` | My implementation of `evolve`, which computes the next generation of Conway's Game of Life |

## Notes for Claude Code

- **Game of Life rules:** a live cell (`1`) with 2 or 3 live neighbors survives; a dead
  cell (`0`) with exactly 3 live neighbors becomes alive; every other cell is dead in the
  next generation. Neighbors are the 8 surrounding cells. Cells outside the grid count as
  dead (no wrap-around).
- `evolve` must return a **new** grid (a list of lists) and must not change the input.
- Keep the two supplied `assert` test cases at the bottom of `game_of_life.py` (the block
  and the blinker). Running `python game_of_life.py` with no output means both pass.
- Run the code with the `HW1` virtual environment (created with `uv` in this folder)
  activated. The environment folder (`HW1/` or `.venv/`) is not committed.
- Only files in this folder belong in my PR. Do not edit the shared starter files in
  `Homeworks/HW1/`.
- Branch: `hw_1_Anouksha_Bansal`. The PR goes into `main`, with reviewers `kevinramlal`
  and `raghuramkowdeed`.
- Explain changes before making them
