# CLAUDE.md

This file gives Claude Code context for my HW1 submission in the UCB MFE Python Pre-Program (Fall 2026).

## About this submission

- **Student**: Oscar Sallusti
- **Branch**: `hw_1_Oscar_Sallusti`
- **Assignment spec**: `Lectures/Lecture 1/hw1.md`
- **Due**: Monday, October 19, 2026, 11:59 p.m. PT, submitted as a pull request assigned to `kevinramlal` and `elfiasco`

HW1 covers the course tooling: Git, Claude Code, uv, and a short Python exercise (Conway's Game of Life).

## Files in this directory

| File | Purpose |
| ---- | ------- |
| `git.txt` | Name and favorite movie (`<FIRST_NAME> <LAST_NAME>, <YOUR_FAV_MOVIE>`) |
| `uv.txt` | uv commands: create the `HW1` venv on Python 3.12.9, install `Homeworks/HW1/requirements.txt`, downgrade `requests` to 2.24.0, upgrade it to the latest, and record that version |
| `CLAUDE.md` | This file; shows Claude Code is installed and initialized |
| `game_of_life.py` | Copy of `Homeworks/HW1/game_of_life.py` with `evolve(board)` implemented |

## Notes for Claude Code

- `HW1/` is the local uv virtual environment. It must **not** be committed or edited. The repo's `.gitignore` (`/Homeworks/HW1/*/HW1/`) and uv's own `HW1/.gitignore` both keep it out of git.
- Don't commit `.DS_Store` files either.
- Leave the shared files in `Homeworks/HW1/` (`game_of_life.py`, `requirements.txt`, `README.md`) unchanged. Only this folder is submitted.
- `requests==2.24.0` installs but can't be imported on Python 3.12. Use `uv pip show requests` to check the installed version, and upgrade before importing it.
- `evolve` should return the next generation of the board under the standard Game of Life rules:
  - A live cell with 2 or 3 live neighbors survives.
  - A dead cell with exactly 3 live neighbors becomes alive.
  - Every other cell is dead in the next generation.
- Don't mutate the input board. Build the new board from the old state.
- Check the work with `python game_of_life.py` (run from this directory with the `HW1` venv active). Both test cases must pass before submitting.
- I'm learning, so keep code clear and commented, and explain changes instead of only making them.
