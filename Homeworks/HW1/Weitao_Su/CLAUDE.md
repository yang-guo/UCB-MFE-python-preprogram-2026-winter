# Homework 1

This directory contains my Homework 1 submission for the UCB MFE Python
Pre-Program course, covering Git, Claude Code setup, uv package management,
and basic Python through Conway's Game of Life.

## Submission files

- `game_of_life.py`: Conway's Game of Life exercise, with examples for a
  stable block and an oscillating blinker. The `evolve(initial_state)` function
  is currently a placeholder and still needs to be implemented.
- `git.txt`: My name and favorite movie, Iron Man, for the Git exercise.
- `uv.txt` (still to be created): Commands actually used to create the homework
  environment, install the shared requirements, downgrade `requests` to 2.24.0,
  upgrade it to the latest version, and report that version.
- `CLAUDE.md`: This submission description and working context. The assignment
  also requires installing Claude Code and running `claude` from this directory;
  those steps have not been verified here.

## Notes for Claude Code

- Keep code clear and beginner-friendly, with explanations of non-obvious logic.
- The Game of Life examples use nested Python lists containing `0` and `1`.
- Use a separate local `HW1` virtual environment with Python 3.12.9 inside this
  directory for the uv exercise. Install dependencies from `../requirements.txt`.
  Keep the environment out of the submission and separate from the course-wide
  environment. Inspect the downgraded `requests` version with `uv pip show requests`; upgrade before importing it because its old dependencies do not
  support Python 3.12.
- Once `evolve` is implemented, activate the homework environment and run
  `python game_of_life.py` from this directory to check both supplied assertions.
  Keep both tests; they do not pass with the placeholder.
- Keep homework changes scoped to this directory unless asked otherwise. Leave
  shared starter and requirements files unchanged, and exclude credentials.
