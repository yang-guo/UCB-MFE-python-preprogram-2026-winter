# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this folder is

Romir Kant's Homework 1 submission for the UCB MFE Python Pre-Program (Fall 2026). The full assignment is `Lectures/Lecture 1/hw1.md`; the shared starter files and required layout are described in `Homeworks/HW1/README.md`.

**Due:** Monday, October 19, 2026, 11:59 p.m. Pacific.

## Submission files

This folder must contain exactly these four files:

| File | Contents |
| ---- | -------- |
| `git.txt` | `<FIRST_NAME> <LAST_NAME>, <YOUR_FAV_MOVIE>` (done: `Romir Kant, The Odyssey`) |
| `uv.txt` | The uv commands used to create a Python 3.12.9 virtual environment named `HW1` in this folder, install `../requirements.txt` (`requests==2.25`), downgrade `requests` to 2.24.0, upgrade it to the latest, plus the latest `requests` version number |
| `CLAUDE.md` | This file; proof that Claude Code was installed and initialized |
| `game_of_life.py` | Copy of `../game_of_life.py` with `evolve` implemented |

## Game of Life

- `evolve(initial_state)` takes a board as a list of lists of 0/1 and returns the next generation (Conway's rules: a live cell with 2 or 3 live neighbors survives, a dead cell with exactly 3 becomes alive, all others are dead). All cells update at once, so compute each new cell from the original board.
- The two supplied `assert` statements at the bottom of the file are the tests (a still-life block and an oscillating blinker). Keep both unchanged.
- Run the tests from this folder with the `HW1` environment activated; no output means both pass:

```bash
python game_of_life.py
```

## Constraints

- Only change files inside `Homeworks/HW1/Romir_Kant/`. Leave the shared `../game_of_life.py`, `../requirements.txt`, and other students' folders untouched.
- The `HW1/` virtual environment (and any `.venv/`) is local only and must not be committed.
- `requests==2.24.0` installs but cannot be imported on Python 3.12; check it with `uv pip show requests` instead, and upgrade before using the library.
- Keep the homework environment separate from the course-wide environment at the repository root (`uv sync --locked`).

## Submitting

Work on the branch `hw_1_Romir_Kant` (not `main`). From the repository root:

```bash
git add Homeworks/HW1/Romir_Kant/
git diff --cached
git commit -m "Submit HW1 for Romir Kant"
git push -u origin hw_1_Romir_Kant
```

Open a PR into `main` titled `HW1: Romir Kant`, assigned to `kevinramlal` and `raghuramkowdeed`.
