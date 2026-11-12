# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this folder is

Dhruv Chopra's Homework 1 submission for the UCB MFE Python Pre-Program (Fall 2026). The full assignment is in `../../../Lectures/Lecture 1/hw1.md`; the shared starter files and submission layout are in `../README.md`. Repo-wide guidance is in `../../../AGENTS.md`.

HW1 is due **Monday, October 19, 2026, 11:59 p.m. PDT**. The course allows three free late days and grants no extensions.

## Submission files

The submission needs exactly these four files in this folder:

- `git.txt`: one line, `<FIRST_NAME> <LAST_NAME>, <FAVORITE_MOVIE>`
- `uv.txt`: the **commands** used to (1) create a uv virtual environment named `HW1` with Python 3.12.9 inside this folder, (2) install `../requirements.txt` (`requests==2.25`), (3) downgrade `requests` to 2.24.0, (4) upgrade `requests` to the latest version, and (5) the latest `requests` version number
- `CLAUDE.md`: this file, which also serves as proof that Claude Code was set up
- `game_of_life.py`: an implementation of `evolve(initial_state)`, with both supplied assertions kept

## Running and testing

There is no build, lint, or test framework. The tests are the two `assert` statements at the bottom of `game_of_life.py`. Run them from this folder with the homework environment activated:

```bash
source HW1/bin/activate        # the HW1 uv environment, created per the uv.txt exercise
python game_of_life.py         # no output means both assertions passed
```

The `requests==2.24.0` downgrade is only a package-management exercise. Its old dependencies won't import on Python 3.12, so use `uv pip show requests` to check the installed version.

## Implementation requirements for `evolve`

- Apply one generation of Conway's Game of Life: a live cell with 2 or 3 live neighbors survives, a dead cell with exactly 3 live neighbors becomes alive, and every other cell is dead in the next generation. Count all 8 neighbors, including diagonals.
- The board is a finite list of lists of 0/1 values. Cells outside the board count as dead. Don't assume the board is square.
- Compute every cell from the original board so all cells update at once. Return a new board and leave the input unchanged.
- The two supplied tests (a still-life block and a blinker) are symmetric, so some wrong implementations, such as a matrix transpose, also pass them. Check any change against extra boards, for example a glider or a non-square grid.

## Git and submission rules

- Branch: `hw_1_Dhruv_Chopra`. Open the PR into `main` with the title `HW1: Dhruv Chopra` and assign it to `kevinramlal` and `raghuramkowdeed`.
- Stage only this folder (`git add Homeworks/HW1/Dhruv_Chopra/`) and review the result with `git diff --cached`.
- Don't modify the shared `../game_of_life.py`, `../requirements.txt`, `../README.md`, or another student's folder.
- Don't commit virtual environments (`HW1/`, `.venv/`), `.env` files, or `data.db`.
