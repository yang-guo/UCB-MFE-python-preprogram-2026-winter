# HW1 - Pierre-Francois Pinelli

UC Berkeley MFE Python pre-program, Homework 1 (Basic Python).

## Contents
- `git.txt`: name and favorite movie (Git exercise)
- `uv.txt`: uv commands to create the `HW1` environment (Python 3.12.9), install `requirements.txt`, downgrade `requests` to 2.24.0, then upgrade it to the latest version
- `game_of_life.py`: implementation of `evolve` for Conway's Game of Life, with the two supplied test cases

## Notes for Claude Code
- `evolve` takes a 2D list of 0/1 and returns a new board (input not modified); cells outside the board count as dead.
- Run the tests from this folder with the homework environment active: `python game_of_life.py` (no output means both assertions pass).
- The `HW1/` virtual environment is local only and must not be committed.
