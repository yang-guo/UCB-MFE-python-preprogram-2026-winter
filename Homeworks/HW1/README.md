# Homework 1

**Due: Monday, October 19, 2026, at 11:59 p.m. Pacific Time (PDT, UTC-7).**

The [full assignment](../../Lectures/Lecture%201/hw1.md) covers Git, Claude Code, uv package management with Python 3.12.9, and Conway's Game of Life.

## Shared starter files

- [game_of_life.py](game_of_life.py): copy this into your personal folder and implement `evolve`. It includes the two required test cases; they intentionally fail before you implement the function.
- [requirements.txt](requirements.txt): the initial dependencies for the package-management exercise (`requests==2.25`). Install these in your separate homework environment, then perform the required downgrade and upgrade there.

## Submission layout

Create your own folder using your first and last names, separated by an underscore, with no spaces:

```text
Homeworks/
└── HW1/
    ├── README.md                  # Shared instructions
    ├── requirements.txt           # Shared dependency file
    ├── game_of_life.py             # Shared starter; leave unchanged
    └── <FIRST_NAME>_<LAST_NAME>/   # Your submission
        ├── git.txt                # Your name and favorite movie
        ├── uv.txt                 # Your package-management commands and latest requests version
        ├── CLAUDE.md              # Submission description and context for Claude Code
        └── game_of_life.py         # Your implementation with both supplied tests
```

Your local Python environment (`HW1/` inside your personal folder, or `.venv/`) is not part of the submission.

## Before submitting

1. Complete all four files listed above.
2. Run your `game_of_life.py` with your homework environment activated and confirm that both supplied assertions pass.
3. Review your staged changes to make sure they contain only your submission files.
4. Push your `hw_1_<FIRST_NAME>_<LAST_NAME>` branch and open a PR into `main`, assigning `kevinramlal` and `raghuramkowdeed`.

The full assignment includes the folder-creation and Git submission commands. The course's [late policy](../../README.md#late-policy) applies.
