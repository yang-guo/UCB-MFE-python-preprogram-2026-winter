# CLAUDE.md

This file gives Claude Code context for my submission folder.

## About this submission

This is Jing Jian's Homework 1 for the UC Berkeley MFE Python Pre-Program (Fall 2026),
part of the Basic Python module. The full assignment is in
`Lectures/Lecture 1/hw1.md`, and the starter files and submission layout are in
`Homeworks/HW1/README.md`.

- **Branch:** `hw_1_Jing_Jian`
- **Due:** Monday, October 19, 2026, at 11:59 p.m. Pacific Time (PDT)
- **Submission:** a pull request into `main` titled `HW1: Jing Jian`, assigned to
  `kevinramlal` and `raghuramkowdeed`

## What the assignment covers

1. **Git**: create the `hw_1_Jing_Jian` branch and this personal folder, then add
   `git.txt` containing my name and favorite movie.
2. **Claude Code**: install and initialize Claude Code in this folder. This
   `CLAUDE.md` is the proof that it is set up.
3. **uv package management**: in `uv.txt`, record the commands used to:
   - create a virtual environment named `HW1` with Python 3.12.9 in this folder
   - install the packages from `Homeworks/HW1/requirements.txt` (`requests==2.25`)
   - downgrade `requests` to 2.24.0
   - upgrade `requests` to the latest version
   - note what the latest version of `requests` is
4. **Python, Conway's Game of Life**: implement `evolve(initial_state)` in
   `game_of_life.py`. It takes a 2D list of 0s (dead) and 1s (alive) and returns
   the next generation using the standard rules:
   - A live cell with 2 or 3 live neighbors survives.
   - A dead cell with exactly 3 live neighbors becomes alive.
   - Every other cell dies or stays dead.
   - Neighbors are the 8 surrounding cells. Cells outside the board count as dead.
   - Return a new board and leave the input unchanged.

## Files in this folder

| File | Purpose |
| --- | --- |
| `git.txt` | My name and favorite movie (Git exercise) |
| `uv.txt` | uv commands and the latest `requests` version |
| `CLAUDE.md` | This file: submission description and Claude Code context |
| `game_of_life.py` | My `evolve` implementation plus the two supplied test cases |

## Notes for Claude Code

- Only change files inside `Homeworks/HW1/Jing_Jian/`. Do not edit the shared
  starter files (`Homeworks/HW1/game_of_life.py`, `Homeworks/HW1/requirements.txt`)
  or any other student's folder.
- Keep both supplied `assert` test cases in `game_of_life.py` (a 2x2 block that
  stays the same, and a blinker that flips from vertical to horizontal).
- Test from this folder with the homework environment activated:
  ```bash
  source HW1/bin/activate
  python game_of_life.py
  ```
  No output means both assertions passed.
- The `HW1/` virtual environment stays on my machine only. Never commit it, a
  `.venv/` folder, or any `.env` file.
- `requests==2.24.0` installs but cannot be imported on Python 3.12. Use
  `uv pip show requests` to check the installed version, and upgrade before using it.
- This is a learning exercise, so prefer clear, well-commented code and explain
  changes rather than just making them.
