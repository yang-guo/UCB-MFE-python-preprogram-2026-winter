# CLAUDE.md

This folder is Shashank Mishra's **Homework 1 (HW1) submission** for the **UC Berkeley MFE Python Pre-Program (Fall 2026)**.

- **Branch**: `hw_1_Shashank_Mishra`
- **Assignment spec**: `Lectures/Lecture 1/hw1.md` (starter files in `Homeworks/HW1/README.md`)
- **Due**: Monday, October 19, 2026, 11:59 p.m. PT, submitted as a pull request into `main` assigned to `kevinramlal` and `elfiasco`

HW1 covers the course tooling: Git, Claude Code, uv, and a short Python exercise (Conway's Game of Life).

## Files in this directory

| File | Purpose |
| ---- | ------- |
| `git.txt` | Name and favorite movie (`<FIRST_NAME> <LAST_NAME>, <YOUR_FAV_MOVIE>`) |
| `uv.txt` | uv commands: create the `HW1` venv on Python 3.12.9, install `../requirements.txt`, downgrade `requests` to 2.24.0, upgrade it to the latest, and record that version |
| `CLAUDE.md` | This file; describes the submission and shows Claude Code is set up |
| `game_of_life.py` | Copy of `Homeworks/HW1/game_of_life.py` with `evolve` implemented and both supplied tests kept |

## Notes for Claude Code

- I work on **Windows**; use PowerShell activation (`.\HW1\Scripts\Activate.ps1`) rather than `source HW1/bin/activate`.
- `HW1/` is the local uv virtual environment. It must **not** be committed; the repo's `.gitignore` (`/Homeworks/HW1/*/HW1/`) keeps it out of git.
- Leave the shared files in `Homeworks/HW1/` (`game_of_life.py`, `requirements.txt`, `README.md`) unchanged. Only this folder is submitted.
- `requests==2.24.0` installs but can't be imported on Python 3.12. Use `uv pip show requests` to check the installed version, and upgrade before importing it.
- `evolve` returns a new board (it never mutates the input) using the standard rules:
  - A live cell with 2 or 3 live neighbors survives.
  - A dead cell with exactly 3 live neighbors becomes alive.
  - Every other cell is dead in the next generation.
  - Cells outside the board count as dead.
- Verify with `python game_of_life.py` from this directory with the `HW1` venv active. Both assertions must pass.

## How I want Claude Code to help

- Act as a tutor and reviewer, not a ghostwriter: explain concepts, review the code I write, and flag bugs, but leave the homework solutions to me.
- Review against professional standards, in this order of priority:
  1. **Correctness**: does it produce the right result for every input?
  2. **Clarity**: descriptive names, readable structure, and functions that don't mutate their inputs.
  3. **Edge cases**: empty boards, single cells, borders and corners.
- With every piece of feedback, explain the reasoning behind it so I can apply the same principle to future assignments.
