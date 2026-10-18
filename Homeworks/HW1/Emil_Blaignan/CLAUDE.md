# CLAUDE.md

## Submission description

HW1 submission for Emil Blaignan (UCB MFE Python Pre-Program, Fall 2026). This folder contains:

- `git.txt` - name and favorite movie
- `uv.txt` - the `uv` commands used to create the `HW1` environment (Python 3.12.9), install `requirements.txt`, downgrade `requests` to 2.24.0, upgrade it to the latest, and the latest `requests` version
- `CLAUDE.md` - this file, as proof that Claude Code is installed and initialized
- `game_of_life.py` - implementation of `evolve` for Conway's Game of Life; passes both supplied test cases

## Notes for Claude Code

- `evolve(board)` returns the next generation as a new board and does not mutate its input. Cells outside the board count as dead.
- Run `python game_of_life.py` from this folder with the `HW1` environment activated. The two asserts at the bottom of the file must keep passing.
- Do not commit the `HW1/` virtual environment (it is gitignored). Never edit the shared starter files in `Homeworks/HW1/` or any other student's folder.
- Keep the code simple and well commented; this is a learning exercise.
