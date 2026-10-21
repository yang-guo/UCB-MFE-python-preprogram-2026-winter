# CLAUDE.md

Context for Claude Code when working in this folder.

## Submission

- **Student:** Sterling Couts
- **Assignment:** UCB MFE Python Pre-Program, Homework 1
- **Branch:** `hw_1_Sterling_Couts`
- **Topics covered:** Git, uv package management, and Conway's Game of Life

## Files in this folder

| File | Purpose |
| --- | --- |
| `git.txt` | My name and favorite movie (Git exercise) |
| `uv.txt` | The package-management commands I ran and the latest `requests` version |
| `CLAUDE.md` | This file: submission description and context for Claude Code |
| `game_of_life.py` | My implementation of `evolve` for Conway's Game of Life |

## Python environment

- Use a separate homework environment named `HW1`, created with uv and Python 3.12.9:

  ```bash
  uv venv HW1 --python 3.12.9
  ```

- Activate it before installing packages or running code:
  - macOS/Linux: `source HW1/bin/activate`
  - Windows (PowerShell): `HW1\Scripts\Activate.ps1`
- Install the shared dependencies from `../requirements.txt` into this environment with `uv pip install -r ../requirements.txt`, then do the required downgrade and upgrade of `requests` there.
- Do not commit the virtual environment. `HW1/` (and `.venv/`) must stay out of Git; the repository `.gitignore` already ignores `Homeworks/HW1/*/HW1/`.

## Game of Life

- `game_of_life.py` must keep **both** supplied tests exactly as given:
  - `assert evolve(test_case_1) == test_case_1` (a block stays a block)
  - `assert evolve(test_case_2) == test_case_2_next` (a vertical blinker becomes horizontal)
- Do not modify or remove the test cases or assertions to make them pass; fix `evolve` instead.
- Verify with the `HW1` environment active: `python game_of_life.py` should exit without an `AssertionError`.

## Rules for working in this repository

- Keep all of my work inside this personal folder: `Homeworks/HW1/Sterling_Couts/`.
- Leave shared course files unchanged, including `Homeworks/HW1/README.md`, `Homeworks/HW1/requirements.txt`, `Homeworks/HW1/game_of_life.py`, `Lectures/`, and the root `README.md`.
- Before committing, review staged changes (`git status`, `git diff --staged`) to confirm they contain only my submission files.
