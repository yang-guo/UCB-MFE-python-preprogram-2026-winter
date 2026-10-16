# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Submission

This is Farshid Gilak's HW1 submission for the UCB MFE Python Pre-Program (Fall 2026).

### Files in this folder
- `git.txt` — Name and favorite movie (Mulholland Drive)
- `uv.txt` — Commands used for uv package management exercises
- `game_of_life.py` — Implementation of Conway's Game of Life `evolve` function
- `CLAUDE.md` — This file (proof of Claude Code installation)

## Running the code

Run the Game of Life implementation from this folder with the homework environment activated:

```bash
python game_of_life.py
```

Both assertions at the bottom of `game_of_life.py` must pass before submission.

## Context

The `evolve` function in `game_of_life.py` takes a 2D grid (list of lists of 0s and 1s) and returns the next generation according to Conway's Game of Life rules:
- A live cell with 2 or 3 live neighbors survives.
- A dead cell with exactly 3 live neighbors becomes alive.
- All other cells die or stay dead.

The shared starter file at `Homeworks/HW1/game_of_life.py` must remain unchanged; only the copy in this personal folder is modified.
