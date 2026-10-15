# CLAUDE.md

This document provides context for my HW1 submission for the UC Berkeley MFE Python Pre-Program (Winter 2026).

## Submission overview

- **Student:** Philip Szkudlarski
- **Git branch:** `hw_1_Philip_Szkudlarski`
- **Homework instructions:** `Lectures/Lecture 1/hw1.md`
- **Deadline:** Monday, October 19, 2026, at 11:59 p.m. PT
- **Submission method:** Open a pull request from my personal branch into `main`, assigning `kevinramlal` and `elfiasco`.

This assignment focuses on setting up the development workflow using Git, Claude Code, and `uv`, along with implementing Conway's Game of Life in Python.

## Submission files

My personal submission folder contains the following four files:

| File | Description |
|---|---|
| `git.txt` | My name and favorite movie |
| `uv.txt` | Commands used to create the homework environment and manage the Requests package |
| `CLAUDE.md` | Project context and instructions for Claude Code |
| `game_of_life.py` | My implementation of the `evolve(initial_state)` function and the provided test cases |

## Development guidelines

- **Virtual environment:** Use the `HW1/` environment inside my personal submission folder. Do not include the virtual environment or its generated files in the Git submission.
- **Shared starter files:** Keep the original `Homeworks/HW1/README.md`, `Homeworks/HW1/game_of_life.py`, and `Homeworks/HW1/requirements.txt` unchanged.
- **Package management:** Use `uv` with Python 3.12.9 to install the course requirements, downgrade Requests to version `2.24.0`, verify the installed version, and then upgrade Requests to the latest available version.
- **Platform compatibility:** Requests 2.24.0 may have compatibility issues when imported under Python 3.12. Use package metadata commands to verify the required version before upgrading it.
- **Git hygiene:** Commit only the four files in my personal submission folder. Do not commit `.venv`, `HW1/`, `.DS_Store`, or unrelated repository files.

## Game of Life implementation

Implement `evolve(initial_state)` to return a new grid representing one generation of Conway's Game of Life.

Apply these rules to every cell:

1. A live cell survives if it has two or three live neighbors.
2. A dead cell becomes alive if it has exactly three live neighbors.
3. All other cells are dead in the next generation.

Count the eight surrounding positions, excluding the cell itself. Positions outside the grid should not count as neighbors.

Build a separate output grid rather than modifying the input. Preserve the dimensions of the original grid.

## Validation checklist

Before submitting:

1. Confirm that the homework virtual environment uses Python 3.12.9.
2. Verify the Requests downgrade and subsequent upgrade using `uv pip show requests`.
3. Run `python game_of_life.py` from my personal submission directory with the homework environment activated.
4. Confirm that both provided assertions pass.
5. Review the staged Git changes and ensure that only my four submission files are included.
6. Push `hw_1_Philip_Szkudlarski` and open a pull request into `main` with the required reviewers.

When assisting with this assignment, explain the reasoning behind code changes and prefer readable, well-commented Python over unnecessarily complicated implementations.
