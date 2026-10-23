# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## About this folder

This is Guillaume Estepa's Homework 1 submission for the UCB MFE Python Pre-Program (Fall 2026). The full assignment is in `../../../Lectures/Lecture 1/hw1.md`, and the submission layout is in `../README.md`. It is due **Monday, October 19, 2026, at 11:59 p.m. PDT**.

The submission contains exactly four files:

| File | Purpose |
| ---- | ------- |
| `git.txt` | One line: `Guillaume Estepa, The Big Short` |
| `uv.txt` | The uv commands used for the package-management exercise, plus the latest `requests` version |
| `CLAUDE.md` | This file: a description of the submission and proof that Claude Code is installed |
| `game_of_life.py` | A copy of `../game_of_life.py` with `evolve` implemented and both supplied assertions kept |

## Constraints

- Only change files in this folder. Leave the shared `../game_of_life.py`, `../requirements.txt`, `../README.md`, and other students' folders unchanged.
- Keep both supplied test cases (`assert evolve(...)`) in `game_of_life.py`. The script is the test, so there is no pytest setup.
- Don't commit the local virtual environment `HW1/` (or `.venv/`). The repository `.gitignore` already ignores `Homeworks/HW1/*/HW1/`.
- The homework environment is separate from the course-wide uv project at the repository root (`pyproject.toml`, `uv.lock`).

## Game of Life implementation notes

- `evolve(initial_state)` takes a list of lists of 0/1 values and returns the next generation as a new list of lists, in the same shape. Return a list, not a NumPy array, because the assertions use `==`.
- Compute every cell from the original board, so all cells update simultaneously, and don't change the input.
- Treat cells outside the finite board as dead. Don't wrap around the edges.
- Rules: a live cell with 2 or 3 live neighbors survives, a dead cell with exactly 3 live neighbors becomes alive, and every other cell is dead in the next generation.

## Commands

Run these from this folder (`Homeworks/HW1/Guillaume_Estepa/`):

```bash
# Create and activate the homework environment (Python 3.12.9)
uv venv HW1 --python 3.12.9
source HW1/bin/activate

# Package-management exercise; record these commands in uv.txt
uv pip install -r ../requirements.txt     # requests==2.25
uv pip install requests==2.24.0           # downgrade
uv pip install --upgrade requests         # upgrade to latest
uv pip show requests                      # check the installed version

# Run the tests: no output and exit code 0 means both assertions passed
python game_of_life.py
```

`requests==2.24.0` installs but can't be imported on Python 3.12. The downgrade only practices package management, so upgrade again before you use the library.

## Submitting

The branch is `hw_1_Guillaume_Estepa`. From the repository root:

```bash
git add Homeworks/HW1/Guillaume_Estepa/
git diff --cached        # check that only the four files are staged
git commit -m "Submit HW1 for Guillaume Estepa"
git push -u origin hw_1_Guillaume_Estepa
```

Open a PR into `main` titled `HW1: Guillaume Estepa` and assign it to `kevinramlal` and `raghuramkowdeed`.
