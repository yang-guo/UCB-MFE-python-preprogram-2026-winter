# CLAUDE.md

## About me

I am Ibrahim GUIRO, a student in the UCB MFE Python Pre-Program (2026).

## About this folder

This folder contains my HW1 submission for the UCB MFE Python course. The homework covers:

1. **Tooling**: managing my work with Git (branches, commits, pull requests) and uv (Python environment and dependency management).
2. **Game of Life**: implementing the `evolve` function of Conway's Game of Life.

## Notes

- `HW1/` is my local uv virtual environment (Python 3.12.9). It is gitignored and must not be committed.
- `uv.txt` lists the uv commands I used: create the environment, install `../requirements.txt`, downgrade `requests` to 2.24.0, then upgrade it to the latest version (2.34.2).
- `game_of_life.py` implements `evolve`. Cells outside the board count as dead. The function returns a new grid and does not change the input.
- To run the tests: `source HW1/bin/activate && python game_of_life.py`. No output means both assertions passed.
