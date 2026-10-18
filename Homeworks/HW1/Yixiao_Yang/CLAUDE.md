# HW1 - Yixiao Yang

This folder contains the four files for the Basic Python homework:

- `git.txt`: name and favorite movie.
- `uv.txt`: commands used to create the Python 3.12.9 homework environment,
  install the requirements, and downgrade and upgrade requests.
- `game_of_life.py`: Conway's Game of Life implementation and both original
  test cases.
- `CLAUDE.md`: submission description and context for Claude Code.

## Implementation notes

- The board is a rectangular list of lists containing 0 (dead) or 1 (alive).
- Count the eight surrounding neighbors, treating positions outside the board
  as dead; do not wrap across the edges.
- Apply all updates simultaneously using a new board, leaving the input intact.
- Preserve both test cases from the shared starter file.
- From this folder, activate `HW1` and run `python game_of_life.py`.
- Keep the environment, credentials, and local assistant state out of Git.
- Leave shared files and other students' submissions unchanged.

## Claude Code setup

Claude Code 2.1.302 was installed in the preparation environment, and
`claude --version` was verified. The CLI was launched from this homework
directory and its initial terminal theme was configured. This `CLAUDE.md`
provides the directory's project context.

The startup reached the account sign-in screen. Account authentication and
model use were not completed in the preparation environment.
