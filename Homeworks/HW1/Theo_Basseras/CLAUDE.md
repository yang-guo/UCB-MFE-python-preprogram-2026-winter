# HW1 – Theo Basseras

Homework 1 of the UC Berkeley MFE Python pre-program (fall 2026): Git, Claude Code, uv and Conway's Game of Life.

## What is in this folder

- `git.txt`: my name and favourite movie (Git exercise).
- `uv.txt`: the uv commands I used: a Python 3.12.9 environment called `HW1`, the install from `../requirements.txt`, the downgrade of `requests` to 2.24.0, the upgrade to the latest version, and that version.
- `game_of_life.py`: `evolve(board)` returns the next generation of the board; the two supplied test cases pass.
- `CLAUDE.md`: this file, written with Claude Code in this folder.

## Notes for Claude Code

- Python 3.12.9, in the local environment `HW1/` inside this folder. It is not part of the submission (the repository ignores it).
- Run the tests from this folder with the environment activated: `python game_of_life.py`.
- In `evolve`, cells outside the board count as dead: the board does not wrap around, and the input board is never modified.
- Only the four files above belong to the pull request; the shared starter files in `Homeworks/HW1/` stay unchanged.
