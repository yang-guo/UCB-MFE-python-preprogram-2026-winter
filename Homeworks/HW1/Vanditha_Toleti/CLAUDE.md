# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## About this folder

This is Vanditha Toleti's Homework 1 submission for the UCB MFE Python Pre-Program (Fall 2026). The full assignment is in `../../../Lectures/Lecture 1/hw1.md`; the starter files and submission layout are in `../README.md`.

Due: Monday, October 19, 2026, 11:59 p.m. Pacific (PDT). Work happens on the `hw_1_Vanditha_Toleti` branch and is submitted as a PR into `main` titled `HW1: Vanditha Toleti`, assigned to `kevinramlal` and `raghuramkowdeed`.

## Submission files

- `git.txt` — name and favorite movie, in the format `<FIRST_NAME> <LAST_NAME>, <YOUR_FAV_MOVIE>`.
- `uv.txt` — the uv commands used to: create a virtual environment named `HW1` with Python 3.12.9 in this folder, install `../requirements.txt` (`requests==2.25`), downgrade `requests` to 2.24.0, upgrade it to the latest, and the latest version number.
- `CLAUDE.md` — this file; it serves as proof that Claude Code was installed and initialized here.
- `game_of_life.py` — an implementation of `evolve(initial_state)`, which returns the next generation of Conway's Game of Life. Both supplied `assert` test cases (a stable 2x2 block and a vertical-to-horizontal blinker) must stay in the file and pass.

## Commands

Run from this folder with the `HW1` environment activated:

```bash
uv venv HW1 --python 3.12.9
source HW1/bin/activate
uv pip install -r ../requirements.txt
uv pip show requests          # check the installed version
python game_of_life.py        # passes silently; an AssertionError means a test failed
```

`requests==2.24.0` installs but cannot be imported on Python 3.12; the downgrade is a package-management exercise only.

## Constraints

- Change only files in this folder. Leave the shared `../game_of_life.py` and `../requirements.txt` unchanged, and never edit another student's folder.
- Do not commit the `HW1/` or `.venv/` environment, `.env` files, or credentials.
- Stage only this folder when submitting (`git add Homeworks/HW1/Vanditha_Toleti/` from the repo root), then review with `git diff --cached`.
- The board is a list of lists of 0/1. `evolve` should compute every cell from the original board (simultaneous update), treat cells outside the grid as dead, and not mutate its input.
