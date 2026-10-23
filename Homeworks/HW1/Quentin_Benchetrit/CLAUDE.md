# HW1 - Quentin Benchetrit

Submission for HW1 of the UC Berkeley MFE Python Pre-program (2026).

## Files
- `git.txt`: name and favorite movie.
- `uv.txt`: uv commands used to create the `HW1` environment (Python 3.12.9), install `requirements.txt`, downgrade `requests` to 2.24.0, then upgrade it to the latest version (2.34.2).
- `game_of_life.py`: implementation of `evolve` for Conway's Game of Life. Each cell looks at its 8 neighbors, with no wrap-around at the edges. A live cell survives with 2 or 3 live neighbors, a dead cell becomes alive with exactly 3, every other cell is dead. The result is written to a new board so the input is never modified.

## Notes
- Run `python game_of_life.py` from this folder with the `HW1` environment activated. Both supplied assertions pass.
- The `HW1/` virtual environment stays local and is ignored by git.
