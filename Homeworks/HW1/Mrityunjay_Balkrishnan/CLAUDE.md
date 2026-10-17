# HW1 Submission Context

This folder contains my Homework 1 submission for the UCB MFE Python Pre-Program.

## Contents

- `game_of_life.py` implements `evolve(initial_state)` for one generation of Conway's Game of Life and retains both supplied assertions.
- `git.txt` records my name and favorite movie.
- `uv.txt` records the uv environment and package-management commands for this assignment.

## Notes

The board is a finite rectangular grid. Cells outside the provided board are treated as dead. A generation is calculated from the original board so cell updates happen simultaneously, and `evolve` does not mutate its input.

To complete the assignment's Claude Code setup step, initialize Claude Code from this directory with `claude`.
