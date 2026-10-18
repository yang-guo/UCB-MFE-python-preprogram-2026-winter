# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this folder is

Antoine Amiel's Homework 1 submission for the UCB MFE Python Pre-Program (Fall 2026). It lives at `Homeworks/HW1/Antoine_Amiel/` inside the course repo; the repo root's `AGENTS.md` and `README.md` cover course-wide context. Full assignment: `Lectures/Lecture 1/hw1.md`; layout rules: `Homeworks/HW1/README.md`.

**Due:** Monday, October 19, 2026, 11:59 p.m. PDT. Branch: `hw_1_Antoine_Amiel`.

## Required submission files

All four must be in this folder, and nothing else from the repo should change:

- `git.txt` — one line: `Antoine Amiel, <favorite movie>`.
- `uv.txt` — the uv commands used to: create a venv named `HW1` with Python 3.12.9 in this folder, install `../requirements.txt` (`requests==2.25`), downgrade `requests` to 2.24.0, upgrade it to latest, plus the latest version number.
- `CLAUDE.md` — this file (proof Claude Code was initialized here).
- `game_of_life.py` — copy of the shared starter `../game_of_life.py` with `evolve(initial_state)` implemented.

## game_of_life.py

- `evolve` takes a rectangular list-of-lists board of 0/1 and returns the next generation (Conway's rules) as a new list of lists, so it compares with `==` against the expected boards.
- Keep both supplied module-level `assert`s unchanged (still-life block; blinker oscillator). The file is its own test: it passes when it runs with no output.
- Cells outside the board are dead (no wrap-around); compute all cells from the original board so updates are simultaneous.

Run from this folder with the homework env active:

```bash
source HW1/bin/activate
python game_of_life.py
```

## Environments

- The `HW1/` venv here is for the uv exercise only and must never be committed (it's gitignored). After the 2.24.0 downgrade, `requests` cannot be imported on Python 3.12 — check it with `uv pip show requests` rather than importing.
- The course-wide environment is separate: `uv sync --locked` at the repo root (Python >=3.12, deps in the root `pyproject.toml`).

## Submitting

From the repo root:

```bash
git add Homeworks/HW1/Antoine_Amiel/
git diff --cached   # only this folder's four files; no venv, no shared-file edits
git commit -m "Submit HW1 for Antoine Amiel"
git push -u origin hw_1_Antoine_Amiel
```

Open a PR into `main` titled `HW1: Antoine Amiel`, assigned to `kevinramlal` and `raghuramkowdeed`.
