# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# HW1 - Yixiao Yang

This folder contains the four files for the Basic Python homework:

- `git.txt`: name and favorite movie.
- `uv.txt`: commands used to create the Python 3.12.9 homework environment,
  install the requirements, and downgrade and upgrade requests.
- `game_of_life.py`: Conway's Game of Life implementation and both original
  test cases.
- `CLAUDE.md`: submission description and context for Claude Code.

## Commands

- Setup (once): `uv venv HW1 --python 3.12.9 && source HW1/bin/activate && uv pip install -r ../requirements.txt`
  (`requirements.txt` is shared and lives one level up; it may be missing from
  a standalone checkout.)
- Run tests: `python game_of_life.py`. The tests are bare `assert`s at module
  level, so passing prints nothing and exits 0. A failure raises
  `AssertionError`.
- Check one case: `python -c "from game_of_life import evolve, test_case_2, test_case_2_next; print(evolve(test_case_2) == test_case_2_next)"`
  (importing runs both asserts first).
- No linter, formatter or test framework is configured. Don't add one.
- After the downgrade step in `uv.txt`, don't import `requests==2.24.0` on
  Python 3.12, because its old dependencies don't work there. That step only
  practises package management.

## Implementation notes

- The board is a rectangular list of lists containing 0 (dead) or 1 (alive).
- Count the eight surrounding neighbors, treating positions outside the board
  as dead; do not wrap across the edges.
- Apply all updates simultaneously using a new board, leaving the input intact.
- Preserve both test cases from the shared starter file.
- The tests call `evolve(initial_state)`, so its name and signature must stay
  the same. The tests and fixtures live in `game_of_life.py` with the
  implementation, and they should stay there.
- Keep the environment, credentials, and local assistant state out of Git.
- Leave shared files and other students' submissions unchanged.

## Claude Code setup

Claude Code 2.1.302 was installed and verified on my Mac. I launched it in Homeworks/HW1/Yixiao_Yang and ran /init.
