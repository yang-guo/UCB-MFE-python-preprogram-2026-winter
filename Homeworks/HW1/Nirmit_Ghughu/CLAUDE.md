# CLAUDE.md — HW1 (Nirmit Ghughu)

This folder is my Homework 1 submission for the UCB MFE Python Pre-Program (Fall 2026).
The full assignment is in `Lectures/Lecture 1/hw1.md`; the shared starter files are in `Homeworks/HW1/`.

**Due:** Monday, October 19, 2026, 11:59 p.m. PT
**Branch:** `hw_1_Nirmit_Ghughu` → PR into `main`, titled `HW1: Nirmit Ghughu`, assigned to `kevinramlal` and `elfiasco`

## Files in this folder

| File | Purpose |
|------|---------|
| `game_of_life.py` | Implementation of `evolve(initial_state)`, one generation of Conway's Game of Life, with both supplied test assertions kept |
| `git.txt` | One line: `Nirmit Ghughu, <favorite movie>` |
| `uv.txt` | The `uv` commands used to create the `HW1` environment (Python 3.12.9), install `../requirements.txt`, downgrade `requests` to 2.24.0, upgrade it to the latest, and the latest version number |
| `CLAUDE.md` | This file: context for Claude Code and proof that it was set up in this directory |

## Game of Life notes

- The board is a list of lists of `0` (dead) and `1` (alive) on a finite rectangular grid.
- Cells outside the board count as dead (no wrap-around).
- Rules: a live cell with 2 or 3 live neighbors survives; a dead cell with exactly 3 live neighbors becomes alive; every other cell is dead in the next generation.
- All cells update simultaneously: the next board is built from the original, and `evolve` returns a new board without changing its input.
- Run the tests from this folder with the homework environment activated: `python game_of_life.py` (no output means both assertions passed).

## Guidelines for Claude Code

- Only edit files inside `Homeworks/HW1/Nirmit_Ghughu/`. Do not change the shared starter files (`Homeworks/HW1/game_of_life.py`, `requirements.txt`) or other students' folders.
- Keep both supplied test cases in `game_of_life.py`.
- Never commit the virtual environment (`HW1/` or `.venv/`), `.env` files, or credentials.
- Keep the code plain and simple, competitive-programming style, with no comments.
